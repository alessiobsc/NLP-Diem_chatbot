"""
This module implements a retrieval system using Qdrant and HuggingFace Embeddings.

It supports dense vector search with BGE-M3 model.
"""

from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document

from config import PARENT_STORE_DIR, COLLECTION_NAME, QDRANT_STORAGE_DIR

class QdrantRAG:
    """
    A class to manage a retrieval system with Qdrant and BGE-M3,
    including a Parent-Child retrieval strategy.
    """

    def __init__(
        self, 
        in_memory: bool = True,
        **kwargs  # Consumes unused openrouter_api_key
    ) -> None:
        """
        Initializes the Qdrant client and the embedding model.

        Args:
            in_memory: If True, runs Qdrant in-memory. Otherwise, uses a local file-based storage.
        """
        if in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=str(QDRANT_STORAGE_DIR))

        # Initialize HuggingFaceEmbeddings for BAAI/bge-m3
        self.model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'}, # Or 'cuda' if available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.collection_name = COLLECTION_NAME
        
        # Initialize parent document store
        parent_store = LocalFileStore(str(PARENT_STORE_DIR))
        self.parent_doc_store = create_kv_docstore(parent_store)

    def create_collection(self) -> None:
        """
        Creates the collection in Qdrant with a single dense vector configuration.
        """
        vector_size = self.model.client.get_sentence_embedding_dimension()
        if vector_size is None:
            # Fallback for some models, bge-m3 should have this defined
            vector_size = 1024

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{self.collection_name}' created successfully.")

    def upsert_document(self, document_id: int, text: str, metadata: Dict[str, Any]) -> None:
        """
        Generates a dense embedding and upserts a child document into Qdrant.
        """
        embedding = self.model.embed_documents([text])[0]

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=document_id,
                    vector=embedding,
                    payload=metadata
                )
            ],
            wait=True
        )

    def search_and_retrieve_parents(self, query_text: str, limit: int = 10) -> List[Document]:
        """
        Performs a dense search on child documents and retrieves the corresponding
        parent documents.
        """
        child_results = self.search_children(query_text, limit=limit)
        
        parent_ids = []
        for point in child_results:
            parent_id = point.payload.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
        
        if not parent_ids:
            return []
            
        parent_docs = self.parent_doc_store.mget(parent_ids)
        return [doc for doc in parent_docs if doc is not None]

    def search_children(self, query_text: str, limit: int = 10) -> List[models.ScoredPoint]:
        """
        Performs a dense vector search on child documents in Qdrant.
        """
        query_embedding = self.model.embed_query(query_text)

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
