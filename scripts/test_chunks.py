"""
Analisi dei chunk salvati nel Chroma DB e nel parent store.
Verifica corretta separazione, metadati, e qualità del contenuto (trafilatura output).
"""

import io
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from app import embedding_model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHROMA_DIR, COLLECTION_NAME, PARENT_STORE_DIR,
    CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE, EMBEDDING_DIMENSION
)

BATCH_SIZE = 2000


def fetch_all_children(vectorstore) -> tuple[list[str], list[dict]]:
    """Paginated fetch — avoids SQLite 'too many SQL variables' on large collections."""
    docs, metas = [], []
    offset = 0
    while True:
        batch = vectorstore.get(limit=BATCH_SIZE, offset=offset)
        bdocs = batch.get("documents") or []
        bmetas = batch.get("metadatas") or []
        if not bdocs:
            break
        docs.extend(bdocs)
        metas.extend(bmetas)
        offset += len(bdocs)
        if len(bdocs) < BATCH_SIZE:
            break
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
        # 6-word shingle fingerprint of first 100 words
        words = t.lower().split()[:100]
        fp = " ".join(words[:6]) if len(words) >= 6 else " ".join(words)
        fingerprints[fp] += 1
    duplicates = sum(v - 1 for v in fingerprints.values() if v > 1)
    return duplicates / len(texts) if texts else 0.0


def main() -> None:
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_diem/ not found. Run --full first.")
        return

    from langchain_chroma import Chroma
    from langchain_classic.storage import LocalFileStore, create_kv_docstore

    print("Connecting to Chroma...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
    )

    # ── Child chunks ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHILD CHUNKS (Chroma vector store)")
    print("=" * 60)

    docs, metas = fetch_all_children(vectorstore)
    total_children = len(docs)
    print(f"Total child chunks: {total_children}")

    if not docs:
        print("Nessun chunk trovato.")
    else:
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
        print(f"  Min / Avg / Max (chars)   : {min(lengths)} / {sum(lengths)//total_children} / {max(lengths)}")
        print(f"  Min / Avg / Max (words)   : {min(words)} / {sum(words)//total_children} / {max(words)}")
        print(f"  Chunks < 80 chars (tiny)  : {tiny}  ({tiny/total_children*100:.1f}%)")
        print(f"  Chunks > {CHILD_CHUNK_SIZE*1.2:.0f} chars (oversized): {over_limit}  ({over_limit/total_children*100:.1f}%)")

        print(f"\n--- Content quality ---")
        print(f"  Nav/boilerplate heuristic : {nav_hits}  ({nav_hits/total_children*100:.1f}%)")
        print(f"  Near-duplicate rate       : {rep_score*100:.1f}%")

        print(f"\n--- Domain breakdown ---")
        for dom, cnt in sorted(domain_counter.items(), key=lambda x: -x[1]):
            pct = cnt / total_children * 100
            dom_docs = domain_docs[dom]
            avg_len = sum(len(d) for d in dom_docs) // len(dom_docs)
            dom_nav = sum(1 for d in dom_docs if looks_like_nav(d))
            print(f"  {dom:<40s} {cnt:>6d} ({pct:.1f}%)  avg={avg_len}ch  nav={dom_nav}")

        print(f"\n--- Sample child chunks (1 per domain) ---")
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

    # ── Parent chunks ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARENT CHUNKS (LocalFileStore)")
    print("=" * 60)

    fs = LocalFileStore(str(PARENT_STORE_DIR))
    parent_store = create_kv_docstore(fs)

    try:
        keys = list(parent_store.yield_keys())
        total_parents = len(keys)
        print(f"Total parent chunks: {total_parents}")

        if keys:
            ratio = total_children / total_parents if total_parents else 0
            print(f"Child/Parent ratio  : {ratio:.2f}  (expected ~{CHILD_CHUNK_SIZE and PARENT_CHUNK_SIZE // CHILD_CHUNK_SIZE})")

            # Sample first 3 parents
            sample_keys = keys[:3]
            sample_docs = parent_store.mget(sample_keys)
            print(f"\n--- Sample parent chunks ---")
            for key, pdoc in zip(sample_keys, sample_docs):
                if not pdoc:
                    continue
                snippet = pdoc.page_content.strip()[:400].replace("\n", " | ")
                print(f"\n  ID     : {key}")
                print(f"  URL    : {pdoc.metadata.get('source', '')}")
                print(f"  Chars  : {len(pdoc.page_content.strip())}")
                print(f"  Content: {snippet}")

    except Exception as e:
        print(f"Errore parent store: {e}")


if __name__ == "__main__":
    main()