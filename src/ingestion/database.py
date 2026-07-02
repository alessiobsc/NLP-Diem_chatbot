"""
Database indexing module for the DIEM Chatbot.
Handles the splitting and vectorization of documents using a Parent-Child strategy.
"""

import hashlib
import os
import time
from collections import defaultdict
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
        self.last_indexed_parent_ids_by_source: dict[str, list[str]] = {}
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
            chunk_overlap=PARENT_CHUNK_OVERLAP,
            separators=["\n\n", ".\n", ". ", "\n", " ", ""],
        )
        self._child_splitter = ContextHeaderTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            separators=["\n\n", ".\n", ". ", "\n", " ", ""],
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

    def delete_sources(self, sources: Iterable[str], known_parent_ids: Optional[dict[str, list[str]]] = None) -> None:
        """
        Delete child chunks from Chroma and matching parent documents from the docstore.

        This is used by incremental updates before re-indexing changed source URLs.
        Parent IDs are primarily recovered from Chroma child metadata ("doc_id");
        stored IDs from crawl_state are used as a fallback.
        """
        known_parent_ids = known_parent_ids or {}

        for source in sorted(set(s for s in sources if s)):
            child_ids: list[str] = []
            parent_ids: set[str] = set(known_parent_ids.get(source, []))

            try:
                existing = self._child_vectorstore.get(
                    where={"source": source},
                    include=["metadatas"],
                )
                child_ids = list(existing.get("ids") or [])
                for metadata in existing.get("metadatas") or []:
                    if not isinstance(metadata, dict):
                        continue
                    parent_id = metadata.get("doc_id") or metadata.get("chunk_id")
                    if parent_id:
                        parent_ids.add(parent_id)
            except Exception as e:
                logger.warning(f"Could not inspect existing Chroma chunks for {source}: {e}")

            if child_ids:
                try:
                    self._child_vectorstore.delete(ids=child_ids)
                    logger.info(f"Deleted {len(child_ids)} child chunks from Chroma for {source}")
                except Exception as e:
                    logger.warning(f"Could not delete Chroma chunks for {source}: {e}")

            if parent_ids:
                try:
                    self._parent_doc_store.mdelete(list(parent_ids))
                    logger.info(f"Deleted {len(parent_ids)} parent docs for {source}")
                except Exception as e:
                    logger.warning(f"Could not delete parent docs for {source}: {e}")

    @staticmethod
    def _add_context_headers_to_parent_documents(docs: List[Document]) -> tuple[int, int]:
        """
        Generate context headers per source URL (one LLM call per URL for docs with
        fewer than LARGE_DOC_THRESHOLD parent chunks). Large docs keep per-chunk headers
        since their chunks cover semantically distinct sections.
        """
        LARGE_DOC_THRESHOLD = 10

        # Group doc indices by source URL, preserving order
        url_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(docs):
            url_to_indices[doc.metadata.get("source", "")].append(i)

        large_doc_sources: set[str] = set()
        url_header_cache: dict[str, str] = {}  # source → header for small docs

        n_small = sum(1 for idxs in url_to_indices.values() if len(idxs) < LARGE_DOC_THRESHOLD)
        n_large = len(url_to_indices) - n_small
        total = len(docs)
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("PHASE 2B - Generating parent context headers")
        logger.info(f"  {n_small} URLs -> per-URL header | {n_large} URLs -> per-chunk (large doc)")
        logger.info("=" * 60)

        # Pass 1: generate one header per small-doc URL using first chunk
        n_urls = len(url_to_indices)
        for url_idx, (source, indices) in enumerate(url_to_indices.items(), 1):
            if len(indices) >= LARGE_DOC_THRESHOLD:
                large_doc_sources.add(source)
                continue
            first_doc = docs[indices[0]]
            text = first_doc.page_content
            prev = _get_context_header(first_doc)
            if prev:
                text = _strip_context_header(text, prev)
            try:
                header = generate_context_header(text, source, first_doc.metadata)
            except Exception as e:
                logger.warning(f"Could not generate context header for {source}: {e}")
                header = ""
            url_header_cache[source] = header

            if url_idx % 50 == 0 or url_idx == n_urls:
                elapsed_mins = (time.time() - start_time) / 60
                logger.info(
                    f"  -> URL header pass: {url_idx}/{n_urls} URLs; "
                    f"elapsed={elapsed_mins:.1f} min; last={source}"
                )

        # Pass 2: strip old headers and assign to all docs
        generated = 0
        missing = 0
        for index, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "")
            prev = _get_context_header(doc)
            if prev:
                doc.page_content = _strip_context_header(doc.page_content, prev)

            if source in large_doc_sources:
                # Per-chunk for large docs
                logger.debug(f"Header (per-chunk): {index}/{total}; source={source}")
                try:
                    header = generate_context_header(doc.page_content, source, doc.metadata)
                except Exception as e:
                    logger.warning(f"Could not generate context header for parent from {source}: {e}")
                    header = ""
            else:
                header = url_header_cache.get(source, "")

            if header:
                doc.metadata["context_header"] = header
                generated += 1
            else:
                doc.metadata.pop("context_header", None)
                missing += 1

            if index % 25 == 0 or index == total:
                elapsed_mins = (time.time() - start_time) / 60
                logger.info(
                    "  -> Header assign progress: "
                    f"{index}/{total} ({index / total:.1%}); "
                    f"generated={generated}; missing={missing}; "
                    f"elapsed={elapsed_mins:.1f} min; last={source}"
                )

        return generated, missing

    @staticmethod
    def _assign_chunk_indices(parent_docs: List[Document]) -> None:
        """Assign chunk_index, chunk_total, chunk_id to each parent in source order.

        chunk_id is a deterministic 16-char hex derived from (source, chunk_index),
        used as the docstore key so adjacent chunks can be fetched at retrieval time.
        """
        by_source: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(parent_docs):
            by_source[doc.metadata.get("source", "")].append(i)

        for source, indices in by_source.items():
            total = len(indices)
            for seq, idx in enumerate(indices):
                chunk_id = hashlib.md5(f"{source}:{seq}".encode()).hexdigest()[:16]
                parent_docs[idx].metadata["chunk_index"] = seq
                parent_docs[idx].metadata["chunk_total"] = total
                parent_docs[idx].metadata["chunk_id"] = chunk_id

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

        self._assign_chunk_indices(parent_docs)
        logger.info("  -> chunk_index/chunk_id assigned to all parent documents")
        parent_ids_by_source: dict[str, list[str]] = defaultdict(list)
        for doc in parent_docs:
            source = doc.metadata.get("source", "")
            chunk_id = doc.metadata.get("chunk_id")
            if source and chunk_id:
                parent_ids_by_source[source].append(chunk_id)
        self.last_indexed_parent_ids_by_source = dict(parent_ids_by_source)

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
                batch_ids = [doc.metadata["chunk_id"] for doc in current_batch]
                self._retriever.add_documents(current_batch, ids=batch_ids)
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
