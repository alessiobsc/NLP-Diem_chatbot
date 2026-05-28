"""
Inspect child chunks and parent docs stored in Chroma for a given source URL.

Usage:
  venv/Scripts/python scripts/inspect_chroma_url.py "https://www.diem.unisa.it/dipartimento/strutture?id=2"
  venv/Scripts/python scripts/inspect_chroma_url.py "strutture?id=2" --partial
  venv/Scripts/python scripts/inspect_chroma_url.py "strutture" --partial --children-only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from config import CHROMA_DIR_NAME, COLLECTION_NAME, PARENT_STORE_DIR


def main():
    parser = argparse.ArgumentParser(description="Inspect Chroma chunks for a source URL")
    parser.add_argument("url", help="Source URL (exact) or substring with --partial")
    parser.add_argument("--partial", action="store_true", help="Substring match on source URL")
    parser.add_argument("--children-only", action="store_true", help="Skip parent doc lookup")
    parser.add_argument("--keyword", help="Filter chunks containing this substring (case-insensitive)")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR_NAME)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    print(f"\nChroma: {CHROMA_DIR_NAME} / {COLLECTION_NAME} ({total} child chunks)\n")

    if args.partial:
        ids, docs, metas = [], [], []
        batch_size = 5000
        offset = 0
        while offset < total:
            batch = col.get(include=["documents", "metadatas"], limit=batch_size, offset=offset)
            for i, meta in enumerate(batch["metadatas"]):
                if args.url.lower() in meta.get("source", "").lower():
                    ids.append(batch["ids"][i])
                    docs.append(batch["documents"][i])
                    metas.append(meta)
            offset += batch_size
    else:
        res = col.get(where={"source": args.url}, include=["documents", "metadatas"])
        ids, docs, metas = res["ids"], res["documents"], res["metadatas"]

    if args.keyword:
        kw = args.keyword.lower()
        filtered = [(i, d, m) for i, d, m in zip(ids, docs, metas) if kw in d.lower()]
        print(f"Child chunks found: {len(ids)} — keyword '{args.keyword}' matches: {len(filtered)}")
        ids, docs, metas = zip(*filtered) if filtered else ([], [], [])
    else:
        print(f"Child chunks found: {len(ids)}")

    if not ids:
        print("  -> URL not in index. Try --partial for substring match.")
        return

    parent_store = None
    if not args.children_only:
        parent_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))

    seen_parents = set()

    for i, (chunk_id, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        print(f"\n{'='*70}")
        print(f"CHILD #{i}")
        print(f"  Source : {meta.get('source', '?')}")
        print(f"  Header : {meta.get('context_header', '?')}")
        print(f"  Title  : {meta.get('title', '?')}")
        print(f"  Chars  : {len(doc)}")
        print(f"  Content:\n")
        for line in doc.splitlines():
            print(f"    {line}")

        if args.children_only:
            continue

        parent_id = meta.get("doc_id")
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        parent_docs = parent_store.mget([parent_id])
        parent = parent_docs[0] if parent_docs else None

        print(f"\n  --- PARENT doc_id={parent_id} ---")
        if parent:
            print(f"  Chars  : {len(parent.page_content)}\n")
            for line in parent.page_content.splitlines():
                print(f"    {line}")
        else:
            print("  -> Parent not found (orphaned child chunk)")

    print(f"\n{'='*70}")
    print(f"Summary: {len(ids)} child chunks | {len(seen_parents)} unique parent docs\n")


if __name__ == "__main__":
    main()
