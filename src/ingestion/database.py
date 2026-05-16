"""
Database indexing module for the DIEM Chatbot.
Handles the splitting and vectorization of documents using a Parent-Child strategy.
"""

import os
import time
from typing import Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    PARENT_STORE_DIR,
    MAX_CHILD_CHUNKS_PER_BATCH,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    EMBEDDING_DIMENSION
)
from src.ingestion.enrichment import generate_context_header
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_context_header(doc: Document) -> str:
    header = doc.metadata.get("context_header", "")
    return header.strip() if isinstance(header, str) else ""


def _strip_context_header(text: str, header: str) -> str:
    if not header:
        return text

    stripped = text.lstrip()
    if stripped.startswith(header):
        return stripped[len(header):].lstrip()
    return text


def _add_context_header(text: str, header: str) -> str:
    if not header:
        return text

    body = _strip_context_header(text, header).lstrip()
    if not body:
        return header
    return f"{header}\n\n{body}"


class ContextHeaderTextSplitter(RecursiveCharacterTextSplitter):
    """
    Text splitter that prepends each generated child chunk with its context header.
    """

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        split_docs: List[Document] = []

        for doc in documents:
            header = _get_context_header(doc)
            if not header:
                split_docs.extend(super().split_documents([doc]))
                continue

            body_doc = Document(
                page_content=_strip_context_header(doc.page_content, header),
                metadata=doc.metadata.copy(),
            )
            chunks = super().split_documents([body_doc])
            for chunk in chunks:
                chunk.page_content = _add_context_header(chunk.page_content, header)
            split_docs.extend(chunks)

        return split_docs


class DocumentIndexer:
    """
    Manages the ingestion and indexing of documents into a vector database
    using a Parent-Child Document Retriever strategy.
    """

    def __init__(self, embedding_model: Embeddings) -> None:
        """
        Initializes the indexer with the given embedding model and sets up the storage.

        Args:
            embedding_model (Embeddings): The model to generate embeddings.
        """
        self._embedding_model = embedding_model
        self._setup_storage()
        self._setup_splitters()
        
        self._child_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_model,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
        )
        
        self._retriever = ParentDocumentRetriever(
            vectorstore=self._child_vectorstore,
            docstore=self._parent_doc_store,
            child_splitter=self._child_splitter,
        )

    def _setup_storage(self) -> None:
        """
        Ensures the local storage directories exist and initializes the document store.
        """
        try:
            os.makedirs(PARENT_STORE_DIR, exist_ok=True)
            parent_store = LocalFileStore(str(PARENT_STORE_DIR))
            self._parent_doc_store = create_kv_docstore(parent_store)
            logger.info(f"Initialized parent document store at {PARENT_STORE_DIR}")
        except Exception as e:
            logger.error(f"Failed to setup storage: {e}")
            raise

    def _setup_splitters(self) -> None:
        """
        Configures the parent and child document splitters.
        """
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, 
            chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        self._child_splitter = ContextHeaderTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, 
            chunk_overlap=CHILD_CHUNK_OVERLAP
        )

    def _get_collection_count(self) -> Optional[int]:
        """
        Safely retrieves the number of items in the underlying Chroma collection.
        
        Returns:
            Optional[int]: The count of items if successful, None otherwise.
        """
        try:
            # TODO (Code Refactorer): Avoid accessing private `_collection` attribute directly. Provide a better API or check for `__len__`.
            return self._child_vectorstore._collection.count()
        except Exception as e:
            logger.warning(f"Could not retrieve collection count: {e}")
            return None

    @staticmethod
    def _add_context_headers_to_parent_documents(docs: List[Document]) -> tuple[int, int]:
        """
        Generate parent-specific context headers and store them only in metadata.
        """
        generated = 0
        missing = 0
        total = len(docs)
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("PHASE 2B - Generating parent context headers")
        logger.info("=" * 60)

        for index, doc in enumerate(docs, 1):
            previous_header = _get_context_header(doc)
            if previous_header:
                doc.page_content = _strip_context_header(doc.page_content, previous_header)

            source = doc.metadata.get("source", "")
            title = str(doc.metadata.get("title", ""))[:120]
            parent_chars = len(doc.page_content or "")
            logger.debug(
                "Header generation parent start: "
                f"{index}/{total}; chars={parent_chars}; source={source}; title={title}"
            )
            try:
                header = generate_context_header(doc.page_content, source, doc.metadata)
            except Exception as e:
                logger.warning(f"Could not generate context header for parent from {source}: {e}")
                header = ""

            if header:
                doc.metadata["context_header"] = header
                generated += 1
            else:
                doc.metadata.pop("context_header", None)
                missing += 1

            if index % 25 == 0 or index == total:
                elapsed_mins = (time.time() - start_time) / 60
                logger.info(
                    "  -> Header generation progress: "
                    f"{index}/{total} parents "
                    f"({index / total:.1%}); generated={generated}; missing={missing}; "
                    f"elapsed={elapsed_mins:.1f} min; last_source={source}"
                )

        return generated, missing

    def index(self, all_docs: List[Document]) -> Chroma:
        """
        Executes the parent-child splitting and indexes the documents into the vector store.
        Processes documents in batches to avoid overwhelming the embedding model or database.

        Args:
            all_docs (List[Document]): The raw documents to be indexed.

        Returns:
            Chroma: The populated vector store.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2 - Starting Parent-Child document splitting")
        logger.info("=" * 60)
        
        parent_docs = self._parent_splitter.split_documents(all_docs)
        total_parents = len(parent_docs)
        logger.info(f"  -> Generated {total_parents} parent documents from {len(all_docs)} sources")

        headers_generated, parents_without_header = self._add_context_headers_to_parent_documents(parent_docs)
        logger.info(f"  -> Context headers generated for {headers_generated} parent documents")
        logger.info(f"  -> Parent documents without context header: {parents_without_header}")
        logger.info("  -> Context headers stored in metadata only; parent content remains clean")

        logger.info("=" * 60)
        logger.info("PHASE 3 - Embedding child chunks and indexing parents")
        logger.info("=" * 60)

        indexed_parent_docs: int = 0
        start_time: float = time.time()
        
        batch: List[Document] = []
        batch_child_chunks: int = 0

        def _process_batch(current_batch: List[Document], current_chunks: int) -> None:
            """Inner helper to process a single batch of documents."""
            nonlocal indexed_parent_docs
            if not current_batch:
                return

            try:
                self._retriever.add_documents(current_batch)
                indexed_parent_docs += len(current_batch)
                
                elapsed_mins = (time.time() - start_time) / 60
                progress_pct = indexed_parent_docs / total_parents
                
                child_count = self._get_collection_count()
                child_info = f", total child chunks in Chroma: {child_count}" if child_count is not None else ""

                logger.info(
                    f"  -> {indexed_parent_docs}/{total_parents} parent docs indexed "
                    f"({progress_pct:.1%}); "
                    f"[Batch chunks: {current_chunks}{child_info}] "
                    f"- Elapsed time: {elapsed_mins:.1f} min"
                )
            except Exception as e:
                logger.error(f"Failed to index batch: {e}")
                raise

        # Iterate over parent docs and batch them based on estimated child chunks
        for parent_doc in parent_docs:
            # Estimate child chunks for this specific parent
            doc_child_chunks = len(self._child_splitter.split_documents([parent_doc]))

            # If adding this doc exceeds the batch limit, process the current batch first
            if batch and (batch_child_chunks + doc_child_chunks > MAX_CHILD_CHUNKS_PER_BATCH):
                _process_batch(batch, batch_child_chunks)
                batch.clear()
                batch_child_chunks = 0

            batch.append(parent_doc)
            batch_child_chunks += doc_child_chunks

        # Process any remaining documents in the final batch
        _process_batch(batch, batch_child_chunks)

        logger.info(f"Indexing completed successfully. Total Parent documents indexed: {total_parents}")
        return self._child_vectorstore


def index_documents(all_docs: List[Document], embedding_model: Embeddings) -> Chroma:
    """
    Main entry point for document indexing. Wraps the DocumentIndexer class 
    to maintain backward compatibility with the existing functional API.

    Args:
        all_docs (List[Document]): The raw documents to index.
        embedding_model (Embeddings): The model to generate embeddings.

    Returns:
        Chroma: The initialized and populated Chroma vector store.
    """
    indexer = DocumentIndexer(embedding_model)
    return indexer.index(all_docs)