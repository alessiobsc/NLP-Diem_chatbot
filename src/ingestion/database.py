import os
import time

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "chroma_diem"
COLLECTION = "diem_knowledge"
PARENT_STORE_DIR = os.path.join(CHROMA_DIR, "parent_store")
MAX_CHILD_CHUNKS_PER_BATCH = 4000


def index_documents(all_docs: list, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    print("\n" + "=" * 60)
    print("PHASE 2 – Parent-Child splitters")
    print("=" * 60)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    parent_docs = parent_splitter.split_documents(all_docs)
    print(f"  -> {len(parent_docs)} parent documents from {len(all_docs)} sources")

    print("\n" + "=" * 60)
    print("PHASE 3 – Embedding child chunks and indexing parents")
    print("=" * 60)

    child_vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    parent_store = LocalFileStore(PARENT_STORE_DIR)
    parent_docstore = create_kv_docstore(parent_store)

    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=parent_docstore,
        child_splitter=child_splitter,
    )

    indexed_parent_docs = 0
    start_time = time.time()
    batch: list = []
    batch_child_chunks = 0

    def index_batch(batch: list, batch_child_chunks: int) -> None:
        nonlocal indexed_parent_docs
        if not batch:
            return

        retriever.add_documents(batch)
        indexed_parent_docs += len(batch)
        elapsed = time.time() - start_time

        try:
            child_chunks = child_vectorstore._collection.count()
            child_info = f", child chunks in Chroma: {child_chunks}"
        except Exception:
            child_info = ""

        print(
            f"  -> {indexed_parent_docs}/{len(parent_docs)} parent docs indexed "
            f"({indexed_parent_docs / len(parent_docs):.1%}); "
            f", batch child chunks: {batch_child_chunks}"
            f"{child_info}; elapsed: {elapsed / 60:.1f} min",
            flush=True,
        )

    for parent_doc in parent_docs:
        doc_child_chunks = len(child_splitter.split_documents([parent_doc]))

        if batch and batch_child_chunks + doc_child_chunks > MAX_CHILD_CHUNKS_PER_BATCH:
            index_batch(batch, batch_child_chunks)
            batch = []
            batch_child_chunks = 0

        batch.append(parent_doc)
        batch_child_chunks += doc_child_chunks

    index_batch(batch, batch_child_chunks)

    print(f"\nIndexing complete. Parent documents indexed: {len(parent_docs)}")
    return child_vectorstore
