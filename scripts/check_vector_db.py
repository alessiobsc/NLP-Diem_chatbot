"""
Check child chunks stored in Chroma and verify context header propagation.
Run after rebuilding the index with main_ingestion.py --full.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from langchain_chroma import Chroma
from src.brain import embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIR, COLLECTION_NAME

BATCH_SIZE = 5000


def fetch_child_chunks(vectorstore: Chroma) -> tuple[list[str], list[dict]]:
    collection = vectorstore._collection
    total = collection.count()
    docs: list[str] = []
    metas: list[dict] = []

    offset = 0
    while offset < total:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=BATCH_SIZE,
            offset=offset,
        )
        docs.extend(batch["documents"])
        metas.extend(batch["metadatas"])
        offset += BATCH_SIZE

    return docs, metas


def print_samples(label: str, rows: list[tuple[str, dict]], limit: int = 5) -> None:
    if not rows:
        return

    print(f"\n{label} (first {min(limit, len(rows))}):")
    for text, meta in rows[:limit]:
        source = meta.get("source", "?") if meta else "?"
        header = meta.get("context_header", "") if meta else ""
        snippet = (text or "")[:180].replace("\n", " ")
        print(f"  source: {source}")
        print(f"  header: {header or '<missing>'}")
        print(f"  text: {snippet}...")


def main() -> None:
    if not CHROMA_DIR.exists():
        print(f"ERROR: {CHROMA_DIR} not found. Run the ingestion pipeline first.")
        sys.exit(1)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    docs, metas = fetch_child_chunks(vectorstore)
    total = len(docs)

    with_header_metadata = 0
    header_present = 0
    header_as_prefix = 0
    duplicate_header = []
    missing_header_metadata = []
    missing_header_text = []
    missing_source = []

    for text, meta in zip(docs, metas):
        meta = meta or {}
        source = meta.get("source", "")
        header = meta.get("context_header", "")
        text = text or ""

        if not source:
            missing_source.append((text, meta))

        if not header:
            missing_header_metadata.append((text, meta))
            continue

        with_header_metadata += 1
        occurrences = text.count(header)
        if occurrences > 0:
            header_present += 1
        else:
            missing_header_text.append((text, meta))
        if text.lstrip().startswith(header):
            header_as_prefix += 1
        if occurrences > 1:
            duplicate_header.append((text, meta))

    pct = (header_present / total * 100) if total else 0
    prefix_pct = (header_as_prefix / total * 100) if total else 0

    print("=== DIEM Chroma Child Chunk Context Header Check ===")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma dir: {CHROMA_DIR}")
    print(f"Total child chunks: {total}")
    print(f"Child chunks with context_header metadata: {with_header_metadata}")
    print(f"Child chunks with context_header in page_content: {header_present} ({pct:.1f}%)")
    print(f"Child chunks starting with context_header: {header_as_prefix} ({prefix_pct:.1f}%)")
    print(f"Child chunks missing context_header metadata: {len(missing_header_metadata)}")
    print(f"Child chunks missing context_header in page_content: {len(missing_header_text)}")
    print(f"Child chunks with duplicate context_header text: {len(duplicate_header)}")
    print(f"Child chunks missing source metadata: {len(missing_source)}")

    print_samples("Chunks missing context_header metadata", missing_header_metadata)
    print_samples("Chunks missing context_header in page_content", missing_header_text)
    print_samples("Chunks with duplicate context_header text", duplicate_header)


if __name__ == "__main__":
    main()
