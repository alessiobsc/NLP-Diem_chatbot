"""
Retrieval debug script.

Shows exactly where a document falls in each pipeline stage:
  Stage 1 — raw Chroma similarity (no threshold, no k cap)
  Stage 2 — after score_threshold filter
  Stage 3 — after bi-encoder top-k
  Stage 4 — after cross-encoder reranking

Usage:
  venv/Scripts/python scripts/debug_retrieval.py "info sui laboratori"
  venv/Scripts/python scripts/debug_retrieval.py "info sui laboratori" --keyword laboratori

Flags:
  --keyword  WORD   highlight rows whose content contains WORD (case-insensitive)
  --raw-k    N      how many raw results to pull from Chroma (default: 60)
  --bi-k     N      bi-encoder k (default: BI_ENCODER_K from config)
  --cross-k  N      cross-encoder top-n (default: CROSS_ENCODER_K from config)
  --threshold F     score threshold (default: RETRIEVER_SCORE_THRESHOLD from config)
"""

import argparse
import os
import sys

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION,
    PARENT_STORE_DIR, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    BI_ENCODER_K, CROSS_ENCODER_K, RETRIEVER_SCORE_THRESHOLD,
)
from src.encoders.embedding_init import build_embedding_model
from src.encoders.reranker import rerank as cross_rerank


def _mark(flag: bool) -> str:
    return "*** PASS ***" if flag else ""


def _highlight(text: str, keyword: str) -> str:
    if not keyword:
        return text[:120]
    lower = text.lower()
    idx = lower.find(keyword.lower())
    if idx == -1:
        return text[:120]
    start = max(0, idx - 40)
    end = min(len(text), idx + 80)
    snippet = text[start:end].replace("\n", " ")
    return f"...{snippet}..."


def main():
    parser = argparse.ArgumentParser(description="Debug retrieval pipeline stages")
    parser.add_argument("query", help="Query string to test")
    parser.add_argument("--keyword", default="", help="Highlight docs containing this word")
    parser.add_argument("--raw-k", type=int, default=60, help="Raw Chroma results (no threshold)")
    parser.add_argument("--bi-k", type=int, default=BI_ENCODER_K)
    parser.add_argument("--cross-k", type=int, default=CROSS_ENCODER_K)
    parser.add_argument("--threshold", type=float, default=RETRIEVER_SCORE_THRESHOLD)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Query   : {args.query}")
    print(f"Keyword : {args.keyword or '(none)'}")
    print(f"Config  : threshold={args.threshold} | bi-k={args.bi_k} | cross-k={args.cross_k}")
    print(f"{'='*70}\n")

    print("Loading embedding model...")
    embedding_model = build_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
        collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
    )
    print(f"Chroma loaded — {vectorstore._collection.count()} child chunks\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: raw similarity, no threshold, large k
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"STAGE 1 — Raw Chroma similarity (top {args.raw_k}, no threshold)")
    print(f"{'─'*70}")

    raw_results = vectorstore.similarity_search_with_relevance_scores(
        args.query, k=args.raw_k
    )

    pass_threshold = []
    pass_bik = []

    for rank, (doc, score) in enumerate(raw_results, start=1):
        above_thresh = score >= args.threshold
        in_bik = rank <= args.bi_k
        kw_hit = args.keyword and args.keyword.lower() in doc.page_content.lower()

        if above_thresh:
            pass_threshold.append((rank, doc, score))
        if in_bik and above_thresh:
            pass_bik.append((rank, doc, score))

        marker = ""
        if kw_hit:
            marker = "  <<< KEYWORD HIT"
        elif above_thresh and in_bik:
            marker = "  [bi-encoder pass]"
        elif above_thresh:
            marker = "  [above thresh, beyond k]"
        else:
            marker = "  [BELOW threshold]"

        src = doc.metadata.get("source", "?")[-60:]
        snippet = _highlight(doc.page_content, args.keyword)
        print(f"  #{rank:3d}  score={score:.4f}{marker}")
        print(f"         src : {src}")
        print(f"         text: {snippet}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: after threshold filter
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"STAGE 2 — After threshold={args.threshold}: {len(pass_threshold)} child chunks pass")
    kw_in_stage2 = sum(
        1 for _, doc, _ in pass_threshold
        if args.keyword and args.keyword.lower() in doc.page_content.lower()
    )
    if args.keyword:
        print(f"  Keyword '{args.keyword}' hits in stage 2: {kw_in_stage2}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: after bi-encoder k cap
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"STAGE 3 — After bi-encoder k={args.bi_k}: {len(pass_bik)} child chunks remain")
    kw_in_stage3 = sum(
        1 for _, doc, _ in pass_bik
        if args.keyword and args.keyword.lower() in doc.page_content.lower()
    )
    if args.keyword:
        print(f"  Keyword '{args.keyword}' hits in stage 3: {kw_in_stage3}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3b: resolve parents via ParentDocumentRetriever
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print("STAGE 3b — Resolving parent docs via ParentDocumentRetriever...")

    parent_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        search_type=SearchType.similarity_score_threshold,
        search_kwargs={"k": args.bi_k, "score_threshold": args.threshold},
    )
    parent_docs = retriever.invoke(args.query)
    print(f"  Parent docs resolved: {len(parent_docs)}")
    kw_in_parents = sum(
        1 for d in parent_docs
        if args.keyword and args.keyword.lower() in d.page_content.lower()
    )
    if args.keyword:
        print(f"  Keyword '{args.keyword}' hits in parents: {kw_in_parents}")
    for i, doc in enumerate(parent_docs, start=1):
        src = doc.metadata.get("source", "?")[-70:]
        kw_hit = args.keyword and args.keyword.lower() in doc.page_content.lower()
        marker = "  <<< KEYWORD" if kw_hit else ""
        print(f"  Parent #{i:2d}: {src}{marker}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: cross-encoder reranking
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{'─'*70}")
    print(f"STAGE 4 — Cross-encoder reranking (top {args.cross_k})...")
    if not parent_docs:
        print("  No parent docs to rerank.\n")
        return

    reranked = cross_rerank(args.query, parent_docs, top_n=args.cross_k)
    print(f"  Final docs kept: {len(reranked)}\n")
    for i, doc in enumerate(reranked, start=1):
        score = doc.metadata.get("relevance_score", "?")
        src = doc.metadata.get("source", "?")[-70:]
        kw_hit = args.keyword and args.keyword.lower() in doc.page_content.lower()
        marker = "  <<< KEYWORD" if kw_hit else ""
        print(f"  Rank #{i}: score={score:.4f}  {src}{marker}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    if args.keyword:
        kw_in_final = sum(
            1 for d in reranked
            if args.keyword.lower() in d.page_content.lower()
        )
        print(f"{'='*70}")
        print(f"SUMMARY for keyword '{args.keyword}':")
        print(f"  Stage 1 (raw {args.raw_k})      : {sum(1 for _, d, _ in raw_results if args.keyword.lower() in d.page_content.lower())} hits")
        print(f"  Stage 2 (threshold)         : {kw_in_stage2} pass")
        print(f"  Stage 3 (bi-k={args.bi_k:3d})        : {kw_in_stage3} pass")
        print(f"  Stage 3b (parents)          : {kw_in_parents} pass")
        print(f"  Stage 4 (cross-k={args.cross_k:2d})      : {kw_in_final} pass")
        if kw_in_stage2 == 0:
            print(f"\n  >> DROP POINT: score threshold ({args.threshold}) — child chunks score too low")
        elif kw_in_stage3 == 0:
            print(f"\n  >> DROP POINT: bi-encoder k cap ({args.bi_k}) — docs rank beyond k after threshold")
        elif kw_in_parents == 0:
            print(f"\n  >> DROP POINT: parent store lookup failed")
        elif kw_in_final == 0:
            print(f"\n  >> DROP POINT: cross-encoder — docs outside top-{args.cross_k}")
        else:
            print(f"\n  >> Keyword present in final context!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
