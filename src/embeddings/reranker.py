# ─────────────────────────────────────────────────────────────────────────────
# Reranker model
# ─────────────────────────────────────────────────────────────────────────────
"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept
so ingestion scripts and tester.py continue to import without modification.
"""
import json
from typing import List
import requests
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from config import EMBEDDING_PROVIDER, LOCAL_RERANKER_MODEL, OPENROUTER_API_KEY, OPENROUTER_RERANKER_MODEL
from src.utils.logger import get_logger


logger = get_logger(__name__)


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

    try:
        reranker = CrossEncoder(LOCAL_RERANKER_MODEL)
    except Exception as e:
        logger.error(f"Failed to load local reranker model '{LOCAL_RERANKER_MODEL}': {e}")
        raise e

    pairs = [[query, d.page_content] for d in documents]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    out = []
    for i, (d, s) in enumerate(ranked[:top_n]):
        d.metadata["relevance_score"] = float(s)
        logger.debug(f"Reranked rank {i+1}: score={s:.4f}, source={d.metadata.get('source', 'Unknown')}")
        out.append(d)

    logger.info(f"Selected top {len(out)} documents after local reranking")
    return out


def rerank(query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
    """Dispatches to the appropriate reranking function based on the provider."""
    if EMBEDDING_PROVIDER == "openrouter":
        return _rerank_with_openrouter(query, documents, top_n)
    elif EMBEDDING_PROVIDER == "local":
        return _rerank_local(query, documents, top_n)
    else:
        raise NotImplementedError(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported. Use 'local' or 'openrouter'.")