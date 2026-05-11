"""
Analisi dei chunk salvati nel Chroma DB e nel parent store.
Eseguire DOPO il completamento della pipeline --full.
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("PYTHONUNBUFFERED", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import (
    CHROMA_DIR, COLLECTION_NAME, PARENT_STORE_DIR,
)

SEP = "=" * 70


def load_stores():
    from langchain_chroma import Chroma
    from langchain_classic.storage import LocalFileStore, create_kv_docstore

    print("Connecting to Chroma...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("Loading parent store...")
    parent_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))

    return vectorstore, parent_store


def analyze_child_chunks(vectorstore):
    print(f"\n{SEP}")
    print("CHILD CHUNKS (Chroma vector store)")
    print(SEP)

    collection = vectorstore._collection
    total = collection.count()
    print(f"Total child chunks: {total}")

    # Pull all with metadata in batches (SQLite variable limit ~32k)
    BATCH = 5000
    docs, metas = [], []
    offset = 0
    while offset < total:
        batch = collection.get(include=["documents", "metadatas"], limit=BATCH, offset=offset)
        docs.extend(batch["documents"])
        metas.extend(batch["metadatas"])
        offset += BATCH
    print(f"Fetched {len(docs)} chunks in batches of {BATCH}")

    # Source domain breakdown
    domain_counts = Counter()
    section_counts = Counter()
    char_lengths = []

    for text, meta in zip(docs, metas):
        source = meta.get("source", "") if meta else ""
        parsed = urlparse(source)
        domain = parsed.netloc or "unknown"
        domain_counts[domain] += 1

        path_parts = parsed.path.strip("/").split("/")
        section = path_parts[1] if len(path_parts) > 1 else path_parts[0] if path_parts else ""
        section_counts[f"{domain}/{section}"] += 1

        char_lengths.append(len(text) if text else 0)

    print(f"\nBy domain:")
    for domain, cnt in domain_counts.most_common():
        print(f"  {cnt:5d}  {domain}")

    print(f"\nTop 15 sections:")
    for sec, cnt in section_counts.most_common(15):
        print(f"  {cnt:5d}  /{sec}")

    avg_len = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    print(f"\nChunk length — avg: {avg_len:.0f} chars, "
          f"min: {min(char_lengths)}, max: {max(char_lengths)}")

    print(f"\n--- Sample child chunks (first 5) ---")
    for i, (text, meta) in enumerate(zip(docs[:5], metas[:5])):
        source = meta.get("source", "?") if meta else "?"
        print(f"\n[{i+1}] source: {source}")
        snippet = (text or "")[:300].replace("\n", " ")
        print(f"     text: {snippet}...")

    return docs, metas


def analyze_parent_docs(parent_store):
    print(f"\n{SEP}")
    print("PARENT DOCS (LocalFileStore)")
    print(SEP)

    try:
        # LocalFileStore keys are stored as files in PARENT_STORE_DIR
        parent_files = list(PARENT_STORE_DIR.iterdir()) if PARENT_STORE_DIR.exists() else []
        print(f"Parent doc files on disk: {len(parent_files)}")
    except Exception as e:
        print(f"Could not count parent files: {e}")

    # Fetch a sample via known keys
    try:
        keys = [f.name for f in PARENT_STORE_DIR.iterdir() if PARENT_STORE_DIR.exists()][:10]
        sample_docs = parent_store.mget(keys)
        valid = [d for d in sample_docs if d is not None]
        print(f"Sample fetch (10 keys): {len(valid)} valid docs")

        print(f"\n--- Sample parent docs (first 3) ---")
        for i, doc in enumerate(valid[:3]):
            source = doc.metadata.get("source", "?")
            text_preview = doc.page_content[:400].replace("\n", " ")
            print(f"\n[{i+1}] source: {source}")
            print(f"     chars: {len(doc.page_content)}")
            print(f"     text: {text_preview}...")
    except Exception as e:
        print(f"Could not sample parent docs: {e}")


def quality_checks(docs, metas):
    print(f"\n{SEP}")
    print("QUALITY CHECKS")
    print(SEP)

    # Empty/very short chunks
    short = [(t, m) for t, m in zip(docs, metas) if len(t or "") < 50]
    print(f"Very short chunks (<50 chars): {len(short)}")
    for text, meta in short[:5]:
        src = meta.get("source", "?") if meta else "?"
        print(f"  [{len(text or '')} chars] {src}: '{text}'")

    # Check for pre-2020 sources slipped through
    import re
    pre2020 = []
    for meta in metas:
        source = meta.get("source", "") if meta else ""
        normalized = source.replace("_", "/")
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20[01]\d)\b", normalized)]
        if years and max(years) < 2020:
            pre2020.append(source)
    print(f"\nPre-2020 sources in child chunks: {len(pre2020)}")
    for s in pre2020[:10]:
        print(f"  {s}")

    # Check for /zh/ or /en/ slipped through
    zh_en = [m.get("source","") for m in metas if m and ("/zh/" in m.get("source","") or "/en/" in m.get("source",""))]
    print(f"\n/zh/ or /en/ sources in child chunks: {len(zh_en)}")
    for s in zh_en[:10]:
        print(f"  {s}")

    # Publication pages (should have text, not PDFs)
    pub_chunks = [(t, m) for t, m in zip(docs, metas)
                  if m and "/pubblicazioni" in m.get("source", "")]
    print(f"\nPubblicazioni chunks: {len(pub_chunks)}")
    if pub_chunks:
        text, meta = pub_chunks[0]
        print(f"  Sample: {meta.get('source','?')}")
        print(f"  Text: {(text or '')[:200].replace(chr(10),' ')}...")

    # Temporal filter metadata
    temporal = Counter(m.get("temporal_filter", "missing") for m in metas if m)
    print(f"\nTemporal filter reasons (top 10):")
    for reason, cnt in temporal.most_common(10):
        print(f"  {cnt:5d}  {reason}")


def main():
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_diem/ not found. Run --full first.")
        sys.exit(1)

    vectorstore, parent_store = load_stores()
    docs, metas = analyze_child_chunks(vectorstore)
    analyze_parent_docs(parent_store)
    quality_checks(docs, metas)

    print(f"\n{SEP}")
    print("ANALYSIS COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
