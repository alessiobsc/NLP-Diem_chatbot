"""
Check child chunks stored in Chroma and verify context header propagation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from src.brain import embedding_model
from config import CHROMA_DIR, COLLECTION_NAME


def fetch_child_chunks(vectorstore: Chroma) -> tuple[list[str], list[dict]]:
    """Retrieve child chunks from the Chroma vectorstore."""
    data = vectorstore.get()
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not docs:
        print("Nessun documento trovato nel vector store Chroma.")
        return [], []

    return docs, metas


def main() -> None:
    print("=== DIEM Chroma Child Chunk Context Header Check ===")

    if not CHROMA_DIR.exists():
        print(f"ERROR: {CHROMA_DIR} not found. Run the ingestion pipeline first.")
        return

    print(f"Chroma dir: {CHROMA_DIR}")
    print("Loading vectorstore...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    docs, metas = fetch_child_chunks(vectorstore)
    total_docs = len(docs)

    if total_docs == 0:
        return

    print(f"\nFound {total_docs} child chunks in Chroma.")
    print("-" * 50)

    # Check propagation of context_header into page_content
    chunks_with_header_in_metadata = 0
    chunks_with_header_in_content = 0

    sample_size = min(5, total_docs)
    print(f"\nAnalyzing first {sample_size} chunks in detail:\n")

    for i in range(total_docs):
        doc_content = docs[i]
        meta = metas[i] if metas else {}

        header_in_meta = meta.get("context_header", "")
        if header_in_meta:
            chunks_with_header_in_metadata += 1
            if doc_content.startswith(header_in_meta):
                chunks_with_header_in_content += 1

        # Print first few samples
        if i < sample_size:
            print(f"--- Chunk {i+1} ---")
            print(f"Source URL: {meta.get('source', 'Unknown')}")
            print(f"Header in metadata: {repr(header_in_meta)}")

            # Show the beginning of the actual content
            content_start = doc_content[:150].replace('\n', '\\n')
            print(f"Content start: {content_start}...")
            print("-" * 30)

    print("\n=== Summary ===")
    print(f"Total chunks checked: {total_docs}")
    print(f"Chunks with context_header in metadata: {chunks_with_header_in_metadata}")
    print(f"Chunks where context_header was properly prepended to content: {chunks_with_header_in_content}")

    if chunks_with_header_in_metadata > 0:
        success_rate = (chunks_with_header_in_content / chunks_with_header_in_metadata) * 100
        print(f"Header propagation success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
