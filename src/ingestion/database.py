"""
Database indexing module for the DIEM Chatbot.
Handles the splitting and vectorization of documents using a Parent-Child strategy.
"""

import os
import time
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    PARENT_STORE_DIR,
    MAX_CHILD_CHUNKS_PER_BATCH,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP
)

class DocumentIndexer:
    """
    Manages the ingestion and indexing of documents into a vector database
    using a Parent-Child Document Retriever strategy.
    """

    def __init__(self, embedding_model: HuggingFaceEmbeddings) -> None:
        """
        Initializes the indexer with the given embedding model and sets up the storage.

        Args:
            embedding_model (HuggingFaceEmbeddings): The model to generate embeddings.
        """
        self._embedding_model = embedding_model
        self._setup_storage()
        self._setup_splitters()
        
        self._child_vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_model,
            persist_directory=str(CHROMA_DIR),
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
            print(f"Initialized parent document store at {PARENT_STORE_DIR}")
        except Exception as e:
            print(f"Failed to setup storage: {e}")
            raise

    def _setup_splitters(self) -> None:
        """
        Configures the parent and child document splitters.
        """
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, 
            chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
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
            return self._child_vectorstore._collection.count()
        except Exception as e:
            print(f"Could not retrieve collection count: {e}")
            return None

    def index(self, all_docs: List[Document]) -> Chroma:
        """
        Executes the parent-child splitting and indexes the documents into the vector store.
        Processes documents in batches to avoid overwhelming the embedding model or database.

        Args:
            all_docs (List[Document]): The raw documents to be indexed.

        Returns:
            Chroma: The populated vector store.
        """
        print("\n" + "=" * 60)
        print("PHASE 2 - Starting Parent-Child document splitting")
        print("=" * 60)
        
        parent_docs = self._parent_splitter.split_documents(all_docs)
        total_parents = len(parent_docs)
        print(f"  -> Generated {total_parents} parent documents from {len(all_docs)} sources")

        print("\n" + "=" * 60)
        print("PHASE 3 - Embedding child chunks and indexing parents")
        print("=" * 60)

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

                print(
                    f"  -> {indexed_parent_docs}/{total_parents} parent docs indexed "
                    f"({progress_pct:.1%}); "
                    f"[Batch chunks: {current_chunks}{child_info}] "
                    f"- Elapsed time: {elapsed_mins:.1f} min",
                    flush=True
                )
            except Exception as e:
                print(f"Failed to index batch: {e}")
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

        print(f"\nIndexing completed successfully. Total Parent documents indexed: {total_parents}")
        return self._child_vectorstore


def index_documents(all_docs: List[Document], embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Main entry point for document indexing. Wraps the DocumentIndexer class 
    to maintain backward compatibility with the existing functional API.

    Args:
        all_docs (List[Document]): The raw documents to index.
        embedding_model (HuggingFaceEmbeddings): The model to generate embeddings.

    Returns:
        Chroma: The initialized and populated Chroma vector store.
    """
    indexer = DocumentIndexer(embedding_model)
    return indexer.index(all_docs)