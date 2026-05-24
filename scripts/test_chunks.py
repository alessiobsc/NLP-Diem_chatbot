"""
Analysis of chunks stored in Qdrant.
Verifies chunk separation, metadata, and content quality.
"""

import io
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    QDRANT_STORAGE_DIR,
    COLLECTION_NAME,
    CHILD_CHUNK_SIZE,
)
from qdrant_client import QdrantClient

BATCH_SIZE = 1000

def fetch_all_chunks(client: QdrantClient, collection_name: str) -> tuple[list[str], list[dict]]:
    """Paginated fetch to get all documents from a Qdrant collection."""
    docs: list[str] = []
    metas: list[dict] = []
    next_offset = None
    
    print("Fetching all chunks from Qdrant...")
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=BATCH_SIZE,
            offset=next_offset,
            with_payload=True,
            with_vectors=False
        )
        
        for point in points:
            payload = point.payload or {}
            docs.append(payload.get("content", ""))
            metas.append(payload)
            
        if not next_offset:
            break
            
    if not docs:
        print("No documents found in the Qdrant collection.")
    return docs, metas


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def word_count(text: str) -> int:
    return len(text.split())


def looks_like_nav(text: str) -> bool:
    """Heuristic: nav/boilerplate chunk has many short lines and few real words."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    avg_line_len = sum(len(l) for l in lines) / len(lines)
    short_line_ratio = sum(1 for l in lines if len(l) < 20) / len(lines)
    return avg_line_len < 25 and short_line_ratio > 0.6


def repetition_score(texts: list[str]) -> float:
    """Fraction of chunks whose content is a near-duplicate of another."""
    fingerprints: Counter = Counter()
    for t in texts:
        words = t.lower().split()[:100]
        fp = " ".join(words[:6]) if len(words) >= 6 else " ".join(words)
        fingerprints[fp] += 1
    duplicates = sum(v - 1 for v in fingerprints.values() if v > 1)
    return duplicates / len(texts) if texts else 0.0


def main() -> None:
    try:
        client = QdrantClient(path=str(QDRANT_STORAGE_DIR))
        print(f"Connecting to Qdrant at {QDRANT_STORAGE_DIR}...")
    except Exception as e:
        print(f"ERROR: Could not connect to Qdrant. Is it running? ({e})")
        return

    print("\n" + "=" * 60)
    print("QDRANT CHUNK ANALYSIS")
    print("=" * 60)

    docs, metas = fetch_all_chunks(client, COLLECTION_NAME)
    total_chunks = len(docs)
    print(f"Total chunks in collection '{COLLECTION_NAME}': {total_chunks}")

    if not docs:
        print("No chunks found.")
        return

    lengths = [len(d) for d in docs]
    words = [word_count(d) for d in docs]
    nav_hits = sum(1 for d in docs if looks_like_nav(d))
    tiny = sum(1 for l in lengths if l < 80)
    over_limit = sum(1 for l in lengths if l > CHILD_CHUNK_SIZE * 1.2)
    rep_score = repetition_score(docs)

    domain_counter: Counter = Counter()
    domain_docs: dict[str, list[str]] = defaultdict(list)
    for d, m in zip(docs, metas):
        dom = domain_of((m or {}).get("source", ""))
        domain_counter[dom] += 1
        domain_docs[dom].append(d)

    print(f"\n--- Size distribution ---")
    print(f"  Min / Avg / Max (chars)   : {min(lengths)} / {sum(lengths)//total_chunks} / {max(lengths)}")
    print(f"  Min / Avg / Max (words)   : {min(words)} / {sum(words)//total_chunks} / {max(words)}")
    print(f"  Chunks < 80 chars (tiny)  : {tiny}  ({tiny/total_chunks*100:.1f}%)")
    print(f"  Chunks > {CHILD_CHUNK_SIZE*1.2:.0f} chars (oversized): {over_limit}  ({over_limit/total_chunks*100:.1f}%)")

    print(f"\n--- Content quality ---")
    print(f"  Nav/boilerplate heuristic : {nav_hits}  ({nav_hits/total_chunks*100:.1f}%)")
    print(f"  Near-duplicate rate       : {rep_score*100:.1f}%")

    print(f"\n--- Domain breakdown ---")
    for dom, cnt in sorted(domain_counter.items(), key=lambda x: -x[1]):
        pct = cnt / total_chunks * 100
        dom_docs_list = domain_docs[dom]
        avg_len = sum(len(d) for d in dom_docs_list) // len(dom_docs_list)
        dom_nav = sum(1 for d in dom_docs_list if looks_like_nav(d))
        print(f"  {dom:<40s} {cnt:>6d} ({pct:.1f}%)  avg={avg_len}ch  nav={dom_nav}")

    print(f"\n--- Sample chunks (1 per domain) ---")
    shown: set = set()
    for d, m in zip(docs, metas):
        m = m or {}
        dom = domain_of(m.get("source", ""))
        if dom in shown:
            continue
        shown.add(dom)
        header = m.get("context_header", "")
        snippet = d.strip()[:300].replace("\n", " | ")
        print(f"\n  [{dom}]")
        print(f"  URL    : {m.get('source', '')}")
        print(f"  Header : {header[:100]}")
        print(f"  Content: {snippet}")
        print(f"  Chars  : {len(d.strip())}  Words: {word_count(d)}  Nav: {looks_like_nav(d)}")

if __name__ == "__main__":
    main()
