"""
Check child chunks stored in Chroma: context header propagation, domain breakdown,
and trafilatura content quality.
"""

import io
import sys

from app import embedding_model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_DIMENSION


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


BATCH_SIZE = 2000


def fetch_child_chunks(vectorstore: Chroma) -> tuple[list[str], list[dict]]:
    """Paginated fetch to avoid SQLite 'too many SQL variables' on large collections."""
    docs: list[str] = []
    metas: list[dict] = []
    offset = 0
    while True:
        batch = vectorstore.get(limit=BATCH_SIZE, offset=offset)
        batch_docs = batch.get("documents") or []
        batch_metas = batch.get("metadatas") or []
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch_metas)
        offset += len(batch_docs)
        if len(batch_docs) < BATCH_SIZE:
            break
    if not docs:
        print("Nessun documento trovato nel vector store Chroma.")
    return docs, metas


def main() -> None:
    print("=== DIEM Chroma Child Chunk Inspection ===")

    if not CHROMA_DIR.exists():
        print(f"ERROR: {CHROMA_DIR} not found. Run the ingestion pipeline first.")
        return

    print(f"Chroma dir: {CHROMA_DIR}")
    print("Loading vectorstore...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
    )

    docs, metas = fetch_child_chunks(vectorstore)
    total_docs = len(docs)

    if total_docs == 0:
        return

    print(f"\nFound {total_docs} child chunks in Chroma.")

    # ── Context header stats ──────────────────────────────────────────────────
    chunks_with_header_in_metadata = 0
    chunks_with_header_in_content = 0
    domain_counter: Counter = Counter()
    empty_or_tiny = 0
    total_chars = 0
    garbage_nav_hits = 0
    NAV_MARKERS = ("privacy policy", "cookie", "home page", "vai al contenuto",
                   "accedi", "login", "registrati", "menu")

    for i, (doc_content, meta) in enumerate(zip(docs, metas)):
        meta = meta or {}
        header_in_meta = meta.get("context_header", "")
        if header_in_meta:
            chunks_with_header_in_metadata += 1
            if doc_content.startswith(header_in_meta):
                chunks_with_header_in_content += 1

        source = meta.get("source", "")
        domain_counter[domain_of(source)] += 1

        length = len(doc_content.strip())
        total_chars += length
        if length < 80:
            empty_or_tiny += 1

        lowered = doc_content[:300].lower()
        if sum(m in lowered for m in NAV_MARKERS) >= 2:
            garbage_nav_hits += 1

    avg_len = total_chars / total_docs if total_docs else 0

    # ── Domain breakdown ──────────────────────────────────────────────────────
    print("\n--- Domain breakdown ---")
    for domain, count in sorted(domain_counter.items(), key=lambda x: -x[1]):
        pct = count / total_docs * 100
        print(f"  {domain:<40s} {count:>6d}  ({pct:.1f}%)")

    # ── Content quality summary ───────────────────────────────────────────────
    print("\n--- Content quality ---")
    print(f"  Avg chunk length (chars)      : {avg_len:.0f}")
    print(f"  Chunks < 80 chars (suspicious): {empty_or_tiny}  ({empty_or_tiny/total_docs*100:.1f}%)")
    print(f"  Nav/garbage heuristic hits    : {garbage_nav_hits}  ({garbage_nav_hits/total_docs*100:.1f}%)")

    # ── Context header propagation ────────────────────────────────────────────
    print("\n--- Context header propagation ---")
    print(f"  Total chunks                  : {total_docs}")
    print(f"  Chunks with header in metadata: {chunks_with_header_in_metadata}")
    print(f"  Chunks with header prepended  : {chunks_with_header_in_content}")
    if chunks_with_header_in_metadata > 0:
        rate = chunks_with_header_in_content / chunks_with_header_in_metadata * 100
        print(f"  Header propagation rate       : {rate:.1f}%")

    # ── Detailed samples (2 per domain) ──────────────────────────────────────
    print("\n--- Content samples (2 per domain, trafilatura output) ---")
    shown: Counter = Counter()
    MAX_PER_DOMAIN = 2
    SAMPLE_LEN = 400

    for doc_content, meta in zip(docs, metas):
        meta = meta or {}
        source = meta.get("source", "")
        domain = domain_of(source)
        if shown[domain] >= MAX_PER_DOMAIN:
            continue
        shown[domain] += 1

        header = meta.get("context_header", "")
        print(f"\n  [{domain}]")
        print(f"  URL    : {source}")
        if header:
            print(f"  Header : {header[:120]}")
        snippet = doc_content.strip()[:SAMPLE_LEN].replace("\n", " ↵ ")
        print(f"  Content: {snippet}")
        print(f"  Length : {len(doc_content.strip())} chars")
        if shown[domain] < MAX_PER_DOMAIN:
            print("  " + "." * 50)

        if all(v >= MAX_PER_DOMAIN for v in shown.values()) and len(shown) >= len(domain_counter):
            break


if __name__ == "__main__":
    main()
