"""
Core AI Brain module for the DIEM Chatbot.

Module-level symbols (embedding_model, reranker, rerank, _format_context) are kept
so ingestion scripts and tester.py continue to import without modification.
"""
from typing import Any, Dict, List
from langchain_core.documents import Document
from src.utils.logger import get_logger


logger = get_logger(__name__)

# ── Module-level symbols: unchanged (imported by ingestion scripts and tester) ──


def format_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats retrieved documents into a single context string.

    Args:
        inputs (Dict[str, Any]): Dictionary containing 'docs'.

    Returns:
        Dict[str, Any]: Inputs augmented with the 'context' string.
    """
    docs: List[Document] = inputs.get("docs", [])
    logger.debug(f"Formatting context from {len(docs)} reranked documents")

    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        content = strip_context_header_from_content(doc)
        block = (
            "<document>\n"
            f"<source>{source}</source>\n"
            f"<content>\n{content}\n</content>\n"
            "</document>"
        )
        formatted_docs.append(block)

    context = "\n\n".join(formatted_docs)
    if docs:
        logger.debug(f"Total formatted context length: {len(context)} characters")
    return {**inputs, "context": context}


def strip_context_header_from_content(doc: Document) -> str:
    """
    Remove generated retrieval headers before sending evidence to the answer model.
    """
    content = doc.page_content or ""
    header = doc.metadata.get("context_header", "")

    if isinstance(header, str) and header:
        stripped = content.lstrip()
        if stripped.startswith(header):
            return stripped[len(header):].lstrip()

    stripped = content.lstrip()
    if stripped.lower().startswith("context:"):
        lines = stripped.splitlines()
        if lines:
            return "\n".join(lines[1:]).lstrip()

    return content


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_text(content) -> str:
    """Extract plain text from a message content that may be a string or a list of content blocks.

    LangSmith Studio sends HumanMessage.content as a list of dicts
    (e.g. [{"type": "text", "text": "..."}]) instead of a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
