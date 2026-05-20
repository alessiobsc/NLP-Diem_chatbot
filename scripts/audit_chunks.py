"""
Audit all child chunks in the Chroma vector store for text quality issues.

Checks per chunk:
  - U+FFFD replacement characters (encoding corruption)
  - Raw PDF artifact (binary stream leaked into text)
  - Symbol density (non-alphanumeric ratio above threshold)
  - Minimum content length

Usage:
    # Audit completo: esporta tutti i chunk problematici (con testo intero) in JSON
    venv/Scripts/python scripts/audit_chunks.py --export bad_chunks.json

    # Varianti
    venv/Scripts/python scripts/audit_chunks.py --min-chars 50 --max-symbol-ratio 0.4 --export bad_chunks.json
    venv/Scripts/python scripts/audit_chunks.py --max-bad 0          # mostra tutti a console senza export
    venv/Scripts/python scripts/audit_chunks.py --limit 500          # audit solo primi 500 chunk (test rapido)
    venv/Scripts/python scripts/audit_chunks.py --batch-size 1000    # batch Chroma più grandi/piccoli
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION
from src.ingestion.parser import is_raw_pdf_artifact
from src.encoders.embedding_init import build_embedding_model

REPLACEMENT_CHAR = "�"
DEFAULT_MIN_CHARS = 50
DEFAULT_MAX_SYMBOL_RATIO = 0.45
DEFAULT_BATCH_SIZE = 500


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    alphanum = sum(1 for c in text if c.isalnum() or c.isspace())
    return 1.0 - (alphanum / len(text))


def audit_chunk(text: str, min_chars: int, max_symbol_ratio: float) -> list[str]:
    issues = []
    if len(text) < min_chars:
        issues.append(f"too_short ({len(text)} chars)")
    if REPLACEMENT_CHAR in text:
        count = text.count(REPLACEMENT_CHAR)
        issues.append(f"replacement_chars ({count}x U+FFFD)")
    if is_raw_pdf_artifact(text):
        issues.append("raw_pdf_artifact")
    ratio = symbol_ratio(text)
    if ratio > max_symbol_ratio:
        issues.append(f"high_symbol_ratio ({ratio:.2f})")
    return issues


def iter_chroma_chunks(vectorstore: Chroma, batch_size: int, limit: int = 0):
    """Fetch Chroma in pages to avoid SQLite 'too many SQL variables'."""
    offset = 0
    fetched = 0
    while True:
        remaining = limit - fetched if limit else batch_size
        current_limit = min(batch_size, remaining) if limit else batch_size
        if current_limit <= 0:
            break

        batch = vectorstore.get(
            limit=current_limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        if not ids:
            break

        for doc_id, text, meta in zip(ids, documents, metadatas):
            yield doc_id, text or "", meta or {}

        batch_count = len(ids)
        fetched += batch_count
        offset += batch_count

        if batch_count < current_limit:
            break


def main():
    parser = argparse.ArgumentParser(description="Audit all Chroma child chunks for text quality")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-symbol-ratio", type=float, default=DEFAULT_MAX_SYMBOL_RATIO)
    parser.add_argument("--show-bad-only", action="store_true", help="Print only problematic chunks")
    parser.add_argument("--limit", type=int, default=0, help="Audit only first N chunks (0=all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Chroma fetch batch size")
    parser.add_argument("--max-bad", type=int, default=30, help="Max bad chunks printed to console (0=all)")
    parser.add_argument("--export", type=str, default="", metavar="FILE", help="Export all bad chunks to JSON file")
    args = parser.parse_args()

    print(f"Loading Chroma collection '{COLLECTION_NAME}' from {CHROMA_DIR_NAME}/...")
    embedding_model = build_embedding_model()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
    )

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    total_in_collection = vectorstore._collection.count()
    target_total = min(args.limit, total_in_collection) if args.limit else total_in_collection
    if args.limit:
        print(f"Auditing {target_total}/{total_in_collection} chunks (--limit {args.limit}, batch {args.batch_size})")
    else:
        print(f"Auditing {target_total} chunks (batch {args.batch_size})...")

    issue_counts: dict[str, int] = {}
    bad_chunks: list[dict] = []
    audited = 0

    for doc_id, text, meta in iter_chroma_chunks(vectorstore, args.batch_size, args.limit):
        audited += 1
        issues = audit_chunk(text, args.min_chars, args.max_symbol_ratio)
        if issues:
            for iss in issues:
                key = iss.split(" ")[0]
                issue_counts[key] = issue_counts.get(key, 0) + 1
            bad_chunks.append({
                "id": doc_id,
                "source": meta.get("source", "unknown"),
                "issues": issues,
                "length": len(text),
                "text": text,
            })

    n_bad = len(bad_chunks)
    n_ok = audited - n_bad

    print()
    print(f"=== AUDIT RESULTS ===")
    print(f"Total chunks audited : {audited}")
    print(f"OK                   : {n_ok} ({100*n_ok/max(audited,1):.1f}%)")
    print(f"Problematic          : {n_bad} ({100*n_bad/max(audited,1):.1f}%)")

    if issue_counts:
        print()
        print("Issue breakdown:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue:<30} {count}")

    if args.export and bad_chunks:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(bad_chunks, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(bad_chunks)} bad chunks to {args.export}")

    if bad_chunks:
        cap = len(bad_chunks) if args.max_bad == 0 else args.max_bad
        label = "ALL" if args.max_bad == 0 else f"first {cap}"
        print()
        print(f"=== PROBLEMATIC CHUNKS ({label}) ===")
        for chunk in bad_chunks[:cap]:
            print(f"  [{', '.join(chunk['issues'])}]")
            print(f"  source  : {chunk['source']}")
            print(f"  length  : {chunk['length']} chars")
            print(f"  preview : {chunk['text'][:120].replace(chr(10), ' ')}")
            print()
        if cap < len(bad_chunks):
            print(f"  ... {len(bad_chunks) - cap} more (use --max-bad 0 or --export FILE to see all)")
    elif not args.show_bad_only:
        print()
        print("No problematic chunks found.")


if __name__ == "__main__":
    main()
