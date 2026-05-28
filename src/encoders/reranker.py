# ─────────────────────────────────────────────────────────────────────────────
# Reranker model
# ─────────────────────────────────────────────────────────────────────────────
"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept
so ingestion scripts and tester.py continue to import without modification.
"""
import json
import re
from typing import List
import requests
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config import LOCAL_RERANKER_MODEL, OPENROUTER_API_KEY, OPENROUTER_RERANKER_MODEL, RERANKER_PROVIDER
from src.utils.logger import get_logger


logger = get_logger(__name__)

# Matches [AY 2025/2026] or [2021] in context_header
_YEAR_RE = re.compile(r'\[AY (\d{4})|\[(\d{4})\]')


def _extract_year(header: str) -> int | None:
    m = _YEAR_RE.search(header)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def _apply_recency_boost(docs: List[Document]) -> List[Document]:
    """Re-sort docs by relevance_score + small recency bonus extracted from context_header.

    Bonus = (year - 2020) * 0.001 → max 0.005 for year 2025.
    Acts as tiebreaker only; docs without a detectable year get no boost.
    """
    for doc in docs:
        year = _extract_year(doc.metadata.get("context_header", ""))
        is_pdf = doc.metadata.get("source", "").lower().endswith(".pdf")
        bonus = max(0.0, (year - 2020) * 0.001) if (year and is_pdf) else 0.0
        base = doc.metadata.get("relevance_score", 0.0)
        doc.metadata["relevance_score_boosted"] = round(base + bonus, 6)
        if bonus > 0:
            logger.debug(
                f"Recency boost +{bonus:.4f} (year={year}): "
                f"{doc.metadata.get('source', '')[-60:]}"
            )
    return sorted(docs, key=lambda d: d.metadata["relevance_score_boosted"], reverse=True)

# Load once at module level — avoids 20s reload on every retrieve call
_local_reranker: CrossEncoder | None = None

def _get_local_reranker() -> CrossEncoder:
    global _local_reranker
    if _local_reranker is None:
        logger.info(f"Loading local reranker model: {LOCAL_RERANKER_MODEL}")
        _local_reranker = CrossEncoder(LOCAL_RERANKER_MODEL)
    return _local_reranker


def _rerank_with_openrouter(query: str, documents: List[Document], top_n: int) -> List[Document]:
    """Reranks documents using the official OpenRouter rerank endpoint."""
    if not documents:
        return []

    logger.debug(f"Reranking {len(documents)} documents for query: '{query}' with OpenRouter: {OPENROUTER_RERANKER_MODEL}")

    docs_content = [d.page_content for d in documents]

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_RERANKER_MODEL,
                "query": query,
                "documents": docs_content,
                "top_n": top_n
            }),
            timeout=15
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            raise ValueError("No results returned from OpenRouter rerank API.")

        reranked_docs = []
        for result in results:
            doc = documents[result["index"]]
            score = result["relevance_score"]
            doc.metadata["relevance_score"] = score
            logger.debug(f"Reranked doc (score={score:.4f}): {doc.metadata.get('source', 'Unknown')}")
            reranked_docs.append(doc)

        reranked_docs = _apply_recency_boost(reranked_docs)
        logger.info(f"Selected top {len(reranked_docs)} documents after OpenRouter reranking")
        return reranked_docs

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OpenRouter rerank API: {e}")
        raise e


def _rerank_local(query: str, documents: List[Document], top_n: int) -> List[Document]:
    """Reranks documents using a local Cross-Encoder model."""
    if not documents:
        return []

    logger.debug(f"Reranking {len(documents)} documents for query: '{query}' with local model: {LOCAL_RERANKER_MODEL}")

    reranker = _get_local_reranker()
    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    out = []
    for d, s in ranked:
        d.metadata["relevance_score"] = float(s)
        out.append(d)

    out = _apply_recency_boost(out)[:top_n]
    for i, d in enumerate(out):
        logger.debug(f"Reranked rank {i+1}: score={d.metadata['relevance_score']:.4f} boosted={d.metadata['relevance_score_boosted']:.4f}, source={d.metadata.get('source', 'Unknown')}")

    logger.info(f"Selected top {len(out)} documents after local reranking")
    return out


def rerank(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """Dispatches to the appropriate reranking function based on the provider."""
    if RERANKER_PROVIDER == "openrouter":
        return _rerank_with_openrouter(query, documents, top_n)
    elif RERANKER_PROVIDER == "local":
        return _rerank_local(query, documents, top_n)
    else:
        raise NotImplementedError(f"RERANKER_PROVIDER '{RERANKER_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")
