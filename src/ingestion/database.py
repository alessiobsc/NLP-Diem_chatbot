"""
Database indexing module for the DIEM Chatbot.
Handles the splitting and vectorization of documents using QdrantRAG
and LocalFileStore for Parent Documents.
"""

import os
import time
from typing import Iterable, List
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore, create_kv_docstore

from config import (
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_STORE_DIR
)
from src.ingestion.enrichment import generate_context_header
from src.utils.logger import get_logger
from src.rag_hybrid import QdrantRAG

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
    Text splitter that prepends each generated chunk with its context header.
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
    Manages the ingestion and indexing of documents into Qdrant using QdrantRAG
    and LocalFileStore for Parent-Child Strategy.
    """

    def __init__(self, in_memory: bool = True) -> None:
        """
        Initializes the indexer and connects to Qdrant via QdrantRAG.
        """
        self._setup_storage()
        self._setup_splitters()
        
        self._qdrant_rag = QdrantRAG(in_memory=in_memory)

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

    def index(self, all_docs: List[Document]) -> QdrantRAG:
        """
        Executes the parent-child splitting and indexes the documents into the vector store.
        
        Args:
            all_docs (List[Document]): The raw documents to be indexed.

        Returns:
            QdrantRAG: The initialized and populated vector store.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2 - Starting Parent-Child document splitting")
        logger.info("=" * 60)
        
        # 1. Split into parents
        parent_docs = self._parent_splitter.split_documents(all_docs)
        total_parents = len(parent_docs)
        logger.info(f"  -> Generated {total_parents} parent documents from {len(all_docs)} sources")

        # 2. Generate context headers for parents
        headers_generated, parents_without_header = self._add_context_headers_to_parent_documents(parent_docs)
        logger.info(f"  -> Context headers generated for {headers_generated} parent documents")
        logger.info(f"  -> Parent documents without context header: {parents_without_header}")
        logger.info("  -> Context headers stored in metadata only; parent content remains clean")

        logger.info("=" * 60)
        logger.info("PHASE 3 - Recreating Qdrant Collection")
        logger.info("=" * 60)
        
        self._qdrant_rag.create_collection()

        logger.info("=" * 60)
        logger.info("PHASE 4 - Embedding child chunks and indexing parents")
        logger.info("=" * 60)

        indexed_parent_docs: int = 0
        total_child_chunks: int = 0
        start_time: float = time.time()
        
        for parent_doc in parent_docs:
            try:
                # Generate a stable ID for the parent based on its content and source
                parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, parent_doc.metadata.get("source", "") + parent_doc.page_content[:100]))
                
                # Store parent in LocalFileStore
                self._parent_doc_store.mset([(parent_id, parent_doc)])
                indexed_parent_docs += 1
                
                # Split parent into children
                child_docs = self._child_splitter.split_documents([parent_doc])
                
                # Index children in Qdrant
                for child_doc in child_docs:
                    # Numeric ID for Qdrant
                    child_id = uuid.uuid4().int & (1<<64)-1
                    
                    metadata = child_doc.metadata.copy()
                    metadata["content"] = child_doc.page_content
                    # IMPORTANT: Link child to parent
                    metadata["parent_id"] = parent_id
                    
                    self._qdrant_rag.upsert_document(
                        document_id=child_id,
                        text=child_doc.page_content,
                        metadata=metadata
                    )
                    total_child_chunks += 1
                
                if indexed_parent_docs % 50 == 0 or indexed_parent_docs == total_parents:
                    elapsed_mins = (time.time() - start_time) / 60
                    progress_pct = indexed_parent_docs / total_parents
                    
                    logger.info(
                        f"  -> {indexed_parent_docs}/{total_parents} parent docs indexed "
                        f"({progress_pct:.1%}); "
                        f"[Total child chunks: {total_child_chunks}] "
                        f"- Elapsed time: {elapsed_mins:.1f} min"
                    )
            except Exception as e:
                logger.error(f"Failed to index parent document {indexed_parent_docs}: {e}")
                raise

        logger.info(f"Indexing completed successfully.")
        logger.info(f"Total Parent documents: {indexed_parent_docs}")
        logger.info(f"Total Child chunks: {total_child_chunks}")
        return self._qdrant_rag


def index_documents(all_docs: List[Document], embedding_model=None, in_memory: bool = True) -> QdrantRAG:
    """
    Main entry point for document indexing. Wraps the DocumentIndexer class.

    Args:
        all_docs (List[Document]): The raw documents to index.
        embedding_model: Ignored.
        in_memory: If True, runs Qdrant in-memory.

    Returns:
        QdrantRAG: The initialized and populated vector store.
    """
    indexer = DocumentIndexer(in_memory=in_memory)
    return indexer.index(all_docs)
