"""
RAG tools for the agentic DIEM Chatbot.

Provides four composable tools:
- rewrite: Rewrite ambiguous query into standalone question using history
- retrieve: Search the DIEM knowledge base
- summarize: Summarize long text
- calculate: Apply academic calculations using retrieved formulas
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from config import CROSS_ENCODER_K
from src.logger import get_logger

logger = get_logger(__name__)


def build_tools(retriever, generation_model, brain_ref) -> list:
    """Build the 4 RAG tools. brain_ref._last_docs is updated by retrieve()."""

    @tool
    def rewrite(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Rewrite an ambiguous query into a standalone question using conversation history.
        Call this when the query contains pronouns (lui, lei, suoi, questo) or refers to
        something mentioned earlier. Returns the rewritten query — then call retrieve() with it."""
        from src.rewriter import rewrite_query
        return rewrite_query(generation_model, state.get("messages", []), query)

    @tool
    def retrieve(query: str) -> str:
        """Search the DIEM knowledge base for documents relevant to the query.
        Call this again if current context is insufficient for a multi-part question."""
        from src.brain import rerank, _format_context

        docs = retriever.invoke(query)
        reranked = rerank(query, docs, top_n=CROSS_ENCODER_K) if docs else []
        # brain_ref._last_docs lets DiemBrain access the latest docs after graph completes
        brain_ref._last_docs = reranked
        logger.info(f"retrieve: {len(reranked)} docs after rerank")
        return _format_context({"docs": reranked, "question": query, "history": []})["context"]

    @tool
    def summarize(text: str) -> str:
        """Summarize long text into concise key points. Use when retrieved context is very long."""
        return generation_model.invoke(f"Summarize concisely in Italian:\n{text}").content

    @tool
    def calculate(context: str, operation: str, values: dict) -> str:
        """Apply an academic calculation using a formula already retrieved from the knowledge base.
        Call retrieve() first to fetch the official DIEM formula, then pass its output as context.
        Use for: graduation grade from average, weighted average, TOLC score thresholds."""
        prompt = (
            f"Using only the official formula found in the following context, "
            f"compute the result for: operation='{operation}', values={values}.\n\n"
            f"Context:\n{context}\n\n"
            f"Show the calculation steps and the final result. "
            f"If the formula is not found in the context, say so explicitly."
        )
        return generation_model.invoke(prompt).content

    return [rewrite, retrieve, summarize, calculate]
