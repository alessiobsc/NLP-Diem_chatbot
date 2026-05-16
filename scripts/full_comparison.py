"""
Full corpus comparison: trafilatura vs Crawl4AI.

Five phases, each independently runnable (intermediate files saved to disk):
  --phase crawl    Crawl all DIEM domains → evaluation/full_comparison/raw_docs.pkl
  --phase extract  Load raw HTML, apply both extractors + Ollama enrichment → docs_*.pkl
  --phase index    Load extracted docs, build Parent-Child index, persist to disk
  --phase qa       Load persisted index, run golden set Q/A with cross-encoder reranking
  --phase test     Run index + qa sequentially (mirrors production pipeline)
  --phase all      Run all five phases sequentially

Usage:
  python scripts/full_comparison.py --phase crawl
  python scripts/full_comparison.py --phase extract
  python scripts/full_comparison.py --phase index   # one-time, ~20 min
  python scripts/full_comparison.py --phase qa      # fast, reuses persisted index

Prerequisites:
  Phase crawl:   network access to diem/docenti/corsi.unisa.it
  Phase extract: Crawl4AI installed; Ollama running with llama3.2:3b for enrichment
  Phase index:   EMBEDDING_MODEL_NAME weights; docs_*.pkl from extract phase
  Phase qa:      persisted stores from index phase; OpenRouter API key (or Ollama fallback)
"""

import argparse
import asyncio
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

# ── bootstrap sys.path ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── output paths ─────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "evaluation" / "full_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DOCS_PATH   = OUT_DIR / "raw_docs.pkl"
TRAF_DOCS_PATH  = OUT_DIR / "docs_trafilatura.pkl"
C4AI_DOCS_PATH  = OUT_DIR / "docs_crawl4ai.pkl"
GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "golden_set_it.json"
MIGRATION_TXT   = PROJECT_ROOT / "migration.txt"

# ── persisted index paths (written by phase_index, read by phase_qa) ─────────
STORES_DIR       = OUT_DIR / "stores"
TRAF_CHROMA_DIR  = STORES_DIR / "traf_chroma"
C4AI_CHROMA_DIR  = STORES_DIR / "c4ai_chroma"
TRAF_PARENTS_DIR = STORES_DIR / "traf_parents"
C4AI_PARENTS_DIR = STORES_DIR / "c4ai_parents"


# =============================================================================
# PHASE 1: CRAWL
# =============================================================================

def phase_crawl() -> None:
    _section("PHASE 1: CRAWL — full DIEM domain (diem + docenti + corsi)")

    from main_ingestion import crawl_phase

    raw_html_docs, pdf_docs = crawl_phase()

    print(f"\nCrawl complete: {len(raw_html_docs)} HTML docs, {len(pdf_docs)} PDF docs")
    print(f"Saving to {RAW_DOCS_PATH} ...")
    with open(RAW_DOCS_PATH, "wb") as f:
        pickle.dump({"html": raw_html_docs, "pdf": pdf_docs}, f)
    print("Saved.")


# =============================================================================
# PHASE 2: EXTRACT
# =============================================================================

async def _c4ai_extract_batch(raw_html_docs: list) -> list:
    """Run Crawl4AI filtered markdown extraction on pre-fetched HTML docs."""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from langchain_core.documents import Document

    _md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
        options={"ignore_links": True, "ignore_images": True},
    )
    _excluded = ["nav", "footer", "header", "aside", "iframe", "noscript", "script", "style"]
    cfg = CrawlerRunConfig(
        excluded_tags=_excluded,
        markdown_generator=_md_gen,
        word_count_threshold=10,
    )

    extracted = []
    total = len(raw_html_docs)
    async with AsyncWebCrawler() as crawler:
        for i, doc in enumerate(raw_html_docs, 1):
            url = doc.metadata.get("source", f"doc_{i}")
            try:
                result = await crawler.aprocess_html(
                    html=doc.page_content, url=url, config=cfg
                )
                md = result.markdown if (result and result.success) else None
                text = (md.fit_markdown or md.raw_markdown) if md else ""
            except Exception as e:
                print(f"  WARNING: Crawl4AI failed for {url}: {e}")
                text = ""
            extracted.append(Document(page_content=text, metadata=dict(doc.metadata)))
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] Crawl4AI extraction progress")
    return extracted


def phase_extract() -> None:
    _section("PHASE 2: EXTRACT — trafilatura + Crawl4AI (same HTML) + Ollama enrichment")

    if not RAW_DOCS_PATH.exists():
        print(f"ERROR: {RAW_DOCS_PATH} not found. Run --phase crawl first.")
        return

    with open(RAW_DOCS_PATH, "rb") as f:
        data = pickle.load(f)
    raw_html_docs: list = data["html"]
    pdf_docs: list      = data["pdf"]
    print(f"Loaded: {len(raw_html_docs)} HTML docs + {len(pdf_docs)} PDF docs")

    from langchain_core.documents import Document
    from src.ingestion.parser import (
        extract_html_metadata, html_extractor,
        filter_recent_documents, NON_ITALIAN_LANG_PREFIXES,
    )
    from src.ingestion.enrichment import add_context_headers

    def lang_filter(docs: list, label: str) -> list:
        kept, dropped = [], 0
        for doc in docs:
            if doc.metadata.get("language", "").startswith(NON_ITALIAN_LANG_PREFIXES):
                dropped += 1
            else:
                kept.append(doc)
        print(f"  [{label}] Language filter: kept {len(kept)}, dropped {dropped}")
        return kept

    def stats(docs: list, label: str) -> None:
        n = len(docs)
        total = sum(len(d.page_content) for d in docs)
        avg = total // n if n else 0
        print(f"  [{label}] docs={n}, total_chars={total:,}, avg={avg}")

    # ── Shared: extract HTML metadata once from raw HTML ─────────────────────
    print("\n-- Extracting HTML metadata (shared) --")
    for doc in raw_html_docs:
        meta = extract_html_metadata(doc.page_content)
        doc.metadata.update(meta)

    # ── Trafilatura path ─────────────────────────────────────────────────────
    print("\n-- Trafilatura extraction --")
    traf_docs = [
        Document(page_content=html_extractor(doc.page_content), metadata=dict(doc.metadata))
        for doc in raw_html_docs
    ]
    traf_docs = lang_filter(traf_docs, "traf")
    traf_docs = filter_recent_documents(traf_docs)
    traf_docs = traf_docs + pdf_docs
    stats(traf_docs, "traf")

    # ── Crawl4AI path (same HTML, async extraction) ──────────────────────────
    print("\n-- Crawl4AI extraction (async) --")
    c4ai_docs = asyncio.run(_c4ai_extract_batch(raw_html_docs))
    c4ai_docs = lang_filter(c4ai_docs, "c4ai")
    c4ai_docs = filter_recent_documents(c4ai_docs)
    c4ai_docs = c4ai_docs + pdf_docs
    stats(c4ai_docs, "c4ai")

    # ── Enrichment: Ollama context headers (applied equally to both) ──────────
    print("\n-- Enrichment: trafilatura docs (Ollama llama3.2:3b) --")
    add_context_headers(traf_docs)
    print("-- Enrichment: Crawl4AI docs (Ollama llama3.2:3b) --")
    add_context_headers(c4ai_docs)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(TRAF_DOCS_PATH, "wb") as f:
        pickle.dump(traf_docs, f)
    with open(C4AI_DOCS_PATH, "wb") as f:
        pickle.dump(c4ai_docs, f)
    print(f"\nSaved: {TRAF_DOCS_PATH}")
    print(f"Saved: {C4AI_DOCS_PATH}")


# =============================================================================
# PHASE 3: INDEX  (Parent-Child, persisted to disk)
# =============================================================================

def phase_index() -> None:
    _section("PHASE 3: INDEX — Parent-Child retrieval (trafilatura + Crawl4AI)")

    for path, label in [(TRAF_DOCS_PATH, "trafilatura"), (C4AI_DOCS_PATH, "Crawl4AI")]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run --phase extract first.")
            return

    with open(TRAF_DOCS_PATH, "rb") as f:
        traf_docs = pickle.load(f)
    with open(C4AI_DOCS_PATH, "rb") as f:
        c4ai_docs = pickle.load(f)
    print(f"Loaded: {len(traf_docs)} traf docs, {len(c4ai_docs)} c4ai docs")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_classic.storage import LocalFileStore, create_kv_docstore
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        return

    from config import (
        EMBEDDING_MODEL_NAME,
        PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
        BI_ENCODER_K,
    )
    from src.ingestion.database import ContextHeaderTextSplitter

    class _E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])
        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME} ...")
    emb = _E5Embeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = ContextHeaderTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    STORES_DIR.mkdir(parents=True, exist_ok=True)
    BATCH = 100

    def _build_and_persist(docs: list, chroma_dir: Path, parent_dir: Path, name: str) -> None:
        print(f"\nSplitting [{name}] into parent chunks (size={PARENT_CHUNK_SIZE}) ...")
        parent_docs = parent_splitter.split_documents(docs)
        print(f"  {len(parent_docs)} parent chunks")

        chroma_dir.mkdir(parents=True, exist_ok=True)
        parent_dir.mkdir(parents=True, exist_ok=True)

        child_vectorstore = Chroma(
            collection_name=f"full_{name}",
            embedding_function=emb,
            persist_directory=str(chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        parent_store = create_kv_docstore(LocalFileStore(str(parent_dir)))
        retriever = ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_store,
            child_splitter=child_splitter,
            search_kwargs={"k": BI_ENCODER_K},
        )

        total = len(parent_docs)
        for i in range(0, total, BATCH):
            batch = parent_docs[i : i + BATCH]
            retriever.add_documents(batch)
            print(f"  [{name}] indexed {min(i + BATCH, total)}/{total} parent docs")

        print(f"  [{name}] index persisted → {chroma_dir}")

    _build_and_persist(traf_docs, TRAF_CHROMA_DIR, TRAF_PARENTS_DIR, "traf")
    _build_and_persist(c4ai_docs, C4AI_CHROMA_DIR, C4AI_PARENTS_DIR, "c4ai")

    print("\nIndexing complete. Run --phase qa to evaluate.")


# =============================================================================
# PHASE 4: Q/A  (ParentDocumentRetriever + Cross-Encoder reranking)
# =============================================================================

_WORD_RE = re.compile(r"\b\w{4,}\b")


def _word_overlap(a: str, b: str) -> float:
    """Fraction of words in b that appear in a (recall-style overlap)."""
    b_words = set(_WORD_RE.findall(b.lower()))
    if not b_words:
        return 0.0
    a_words = set(_WORD_RE.findall(a.lower()))
    return round(len(a_words & b_words) / len(b_words), 4)


def phase_qa() -> None:
    _section("PHASE 4: Q/A — ParentDocumentRetriever + Cross-Encoder reranking")

    missing = [
        d for d in [TRAF_CHROMA_DIR, C4AI_CHROMA_DIR, TRAF_PARENTS_DIR, C4AI_PARENTS_DIR]
        if not d.exists()
    ]
    if missing:
        for d in missing:
            print(f"ERROR: {d} not found.")
        print("Run --phase index first.")
        return

    if not GOLDEN_SET_PATH.exists():
        print(f"ERROR: {GOLDEN_SET_PATH} not found.")
        return

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_classic.storage import LocalFileStore, create_kv_docstore
        from langchain_core.messages import HumanMessage
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        return

    from config import (
        EMBEDDING_MODEL_NAME,
        CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
        LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL,
        OLLAMA_CHAT_MODEL, LLM_TEMPERATURE,
        BI_ENCODER_K, CROSS_ENCODER_K, CROSS_ENCODER_MODEL_NAME,
    )
    from src.ingestion.database import ContextHeaderTextSplitter
    from src.prompts import SYSTEM_PROMPT, REJECTION_TAGS

    # ── Embeddings ────────────────────────────────────────────────────────────
    class _E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])
        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME} ...")
    emb = _E5Embeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={"normalize_embeddings": True})

    child_splitter = ContextHeaderTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    def _load_retriever(chroma_dir: Path, parent_dir: Path, name: str) -> ParentDocumentRetriever:
        child_vectorstore = Chroma(
            collection_name=f"full_{name}",
            embedding_function=emb,
            persist_directory=str(chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        parent_store = create_kv_docstore(LocalFileStore(str(parent_dir)))
        return ParentDocumentRetriever(
            vectorstore=child_vectorstore,
            docstore=parent_store,
            child_splitter=child_splitter,
            search_kwargs={"k": BI_ENCODER_K},
        )

    print("Loading trafilatura retriever ...")
    traf_retriever = _load_retriever(TRAF_CHROMA_DIR, TRAF_PARENTS_DIR, "traf")
    print("Loading Crawl4AI retriever ...")
    c4ai_retriever = _load_retriever(C4AI_CHROMA_DIR, C4AI_PARENTS_DIR, "c4ai")

    # ── Cross-Encoder reranker ────────────────────────────────────────────────
    print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL_NAME} ...")
    reranker = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def _rerank(query: str, docs: list, top_n: int = CROSS_ENCODER_K) -> list:
        if not docs:
            return []
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]

    # ── LLM ──────────────────────────────────────────────────────────────────
    def _build_llm():
        if LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                print(f"LLM: OpenRouter ({OPENROUTER_MODEL})")
                return ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                    model=OPENROUTER_MODEL,
                    temperature=LLM_TEMPERATURE,
                )
            except Exception as e:
                print(f"OpenRouter init failed ({e}), falling back to Ollama")
        from langchain_ollama import ChatOllama
        print(f"LLM: Ollama ({OLLAMA_CHAT_MODEL})")
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)

    llm = _build_llm()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _format_ctx(docs: list) -> str:
        if not docs:
            return "Nessun documento trovato."
        parts = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            parts.append(
                f"<document>\n<source>{src}</source>\n"
                f"<content>\n{doc.page_content}\n</content>\n</document>"
            )
        return "\n\n".join(parts)

    def _ask(question: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            for tag in REJECTION_TAGS:
                text = text.replace(tag, "").strip()
            return text
        except Exception as e:
            return f"ERROR: {e}"

    # ── Golden set ────────────────────────────────────────────────────────────
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        gs = json.load(f)
    questions = [q for q in gs.get("in_scope", []) if "reference" in q]
    print(f"\nLoaded {len(questions)} in_scope questions")

    # ── Q/A loop ──────────────────────────────────────────────────────────────
    results = []
    print()
    for i, q in enumerate(questions, 1):
        qid, qtext, ref = q["id"], q["question"], q["reference"]

        traf_raw     = traf_retriever.invoke(qtext)
        c4ai_raw     = c4ai_retriever.invoke(qtext)
        traf_reranked = _rerank(qtext, traf_raw)
        c4ai_reranked = _rerank(qtext, c4ai_raw)

        traf_ctx = _format_ctx(traf_reranked)
        c4ai_ctx = _format_ctx(c4ai_reranked)

        print(f"  [{i:02d}/{len(questions)}] {qid} "
              f"(traf={len(traf_raw)}->{len(traf_reranked)}, "
              f"c4ai={len(c4ai_raw)}->{len(c4ai_reranked)}) - querying LLM ...")
        traf_answer = _ask(qtext, traf_ctx)
        c4ai_answer = _ask(qtext, c4ai_ctx)

        results.append({
            "id": qid,
            "question": qtext,
            "reference": ref,
            "trafilatura": {
                "answer": traf_answer,
                "answer_overlap": _word_overlap(traf_answer, ref),
                "context_overlap": _word_overlap(traf_ctx, ref),
                "retrieved_sources": [d.metadata.get("source", "") for d in traf_reranked],
            },
            "crawl4ai": {
                "answer": c4ai_answer,
                "answer_overlap": _word_overlap(c4ai_answer, ref),
                "context_overlap": _word_overlap(c4ai_ctx, ref),
                "retrieved_sources": [d.metadata.get("source", "") for d in c4ai_reranked],
            },
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(results)
    agg = {
        "trafilatura": {
            "mean_answer_overlap": round(sum(r["trafilatura"]["answer_overlap"] for r in results) / n, 4),
            "mean_context_overlap": round(sum(r["trafilatura"]["context_overlap"] for r in results) / n, 4),
        },
        "crawl4ai": {
            "mean_answer_overlap": round(sum(r["crawl4ai"]["answer_overlap"] for r in results) / n, 4),
            "mean_context_overlap": round(sum(r["crawl4ai"]["context_overlap"] for r in results) / n, 4),
        },
    }

    # ── Console table ─────────────────────────────────────────────────────────
    _section("Q/A COMPARISON RESULTS")
    print(f"  {'ID':<22} {'Traf ans':>10} {'C4AI ans':>10} {'Traf ctx':>10} {'C4AI ctx':>10} {'Winner':>8}")
    print("  " + "-" * 74)
    for r in results:
        t_a = r["trafilatura"]["answer_overlap"]
        c_a = r["crawl4ai"]["answer_overlap"]
        t_c = r["trafilatura"]["context_overlap"]
        c_c = r["crawl4ai"]["context_overlap"]
        winner = "=" if abs(t_a - c_a) <= 0.02 else ("C4AI" if c_a > t_a else "TRAF")
        print(f"  {r['id']:<22} {t_a:>10.4f} {c_a:>10.4f} {t_c:>10.4f} {c_c:>10.4f} {winner:>8}")
    print("  " + "-" * 74)
    t_ans = agg["trafilatura"]["mean_answer_overlap"]
    c_ans = agg["crawl4ai"]["mean_answer_overlap"]
    t_ctx = agg["trafilatura"]["mean_context_overlap"]
    c_ctx = agg["crawl4ai"]["mean_context_overlap"]
    print(f"  {'AGGREGATE':<22} {t_ans:>10.4f} {c_ans:>10.4f} {t_ctx:>10.4f} {c_ctx:>10.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    llm_name = OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else OLLAMA_CHAT_MODEL
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": n,
        "bi_encoder_k": BI_ENCODER_K,
        "cross_encoder_k": CROSS_ENCODER_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
        "llm": llm_name,
        "aggregate": agg,
        "per_question": results,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")

    _append_to_migration(output, out_path)


def _append_to_migration(data: dict, json_path: Path) -> None:
    """Append Q/A comparison results to migration.txt."""
    agg = data["aggregate"]
    t_ans = agg["trafilatura"]["mean_answer_overlap"]
    c_ans = agg["crawl4ai"]["mean_answer_overlap"]
    t_ctx = agg["trafilatura"]["mean_context_overlap"]
    c_ctx = agg["crawl4ai"]["mean_context_overlap"]
    overall_winner = "Crawl4AI" if c_ans > t_ans + 0.005 else ("Trafilatura" if t_ans > c_ans + 0.005 else "TIE")

    bi_k  = data.get("bi_encoder_k", "?")
    ce_k  = data.get("cross_encoder_k", "?")
    ce_m  = data.get("cross_encoder_model", "?")

    lines = [
        "",
        "=" * 80,
        "FULL CORPUS Q/A COMPARISON (production pipeline)",
        f"   run: {data['timestamp']}",
        f"   json: {json_path.name}",
        "=" * 80,
        "",
        f"  Corpus:        full DIEM crawl (diem.unisa.it + docenti + corsi)",
        f"  Enrichment:    Ollama context headers (llama3.2:3b), applied equally to both",
        f"  Indexing:      ParentDocumentRetriever (parent=2000, child=400, overlap=50)",
        f"  Embedding:     {data['embedding_model']}",
        f"  Retrieval:     bi-encoder k={bi_k} → cross-encoder top {ce_k}",
        f"  Cross-Encoder: {ce_m}",
        f"  Q/A LLM:       {data['llm']}",
        f"  Questions:     {data['n_questions']} in_scope from golden_set_it.json",
        "",
        "  AGGREGATE METRICS",
        "  " + "-" * 60,
        f"  {'Metric':<28} {'Trafilatura':>12} {'Crawl4AI':>12} {'Delta':>10}",
        f"  {'mean answer overlap':<28} {t_ans:>12.4f} {c_ans:>12.4f} {c_ans - t_ans:>+10.4f}",
        f"  {'mean context overlap':<28} {t_ctx:>12.4f} {c_ctx:>12.4f} {c_ctx - t_ctx:>+10.4f}",
        "",
        "  PER-QUESTION ANSWER OVERLAP",
        "  " + "-" * 60,
        f"  {'ID':<22} {'Traf ans':>10} {'C4AI ans':>10} {'Traf ctx':>10} {'C4AI ctx':>10} {'Winner':>8}",
    ]
    for r in data["per_question"]:
        t = r["trafilatura"]["answer_overlap"]
        c = r["crawl4ai"]["answer_overlap"]
        tc = r["trafilatura"]["context_overlap"]
        cc = r["crawl4ai"]["context_overlap"]
        w = "=" if abs(t - c) <= 0.02 else ("C4AI" if c > t else "TRAF")
        lines.append(f"  {r['id']:<22} {t:>10.4f} {c:>10.4f} {tc:>10.4f} {cc:>10.4f} {w:>8}")
    lines += [
        "",
        f"  CONCLUSION: {overall_winner} wins on answer quality (traf={t_ans:.4f}, c4ai={c_ans:.4f})",
        f"  See {json_path.name} for full answers and retrieved sources.",
        "=" * 80,
    ]

    if MIGRATION_TXT.exists():
        with open(MIGRATION_TXT, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Appended to: {MIGRATION_TXT}")


# =============================================================================
# MAIN
# =============================================================================

def _section(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full corpus comparison: trafilatura vs Crawl4AI"
    )
    parser.add_argument(
        "--phase",
        choices=["crawl", "extract", "index", "qa", "test", "all"],
        required=True,
        help=(
            "Phase to run: "
            "crawl | extract | index | qa | "
            "test (index+qa) | all"
        ),
    )
    args = parser.parse_args()

    if args.phase in ("crawl", "all"):
        phase_crawl()
    if args.phase in ("extract", "all"):
        phase_extract()
    if args.phase in ("index", "test", "all"):
        phase_index()
    if args.phase in ("qa", "test", "all"):
        phase_qa()


if __name__ == "__main__":
    main()
