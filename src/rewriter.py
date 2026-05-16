"""
Query rewriter for the DIEM Chatbot.

Resolves pronouns and references in follow-up questions using conversation history,
producing a standalone query suitable for vector retrieval.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.logger import get_logger
from src.prompts import REWRITE_PROMPT

logger = get_logger(__name__)


def rewrite_query(model, messages: list[BaseMessage], raw_query: str) -> str:
    """Rewrite raw_query into a standalone question using conversation history.

    Returns the rewritten query, or raw_query if rewriting fails or produces empty output.
    """
    history_lines = []
    for msg in messages[:-1]:  # exclude current HumanMessage
        if isinstance(msg, HumanMessage) and not getattr(msg, "tool_calls", None):
            from src.brain import _extract_text
            history_lines.append(f"User: {_extract_text(msg.content)}")
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            from src.brain import _extract_text
            text = _extract_text(msg.content)
            if text:
                history_lines.append(f"AI: {text}")

    history_str = "\n".join(history_lines[-6:])
    prompt = f"<history>\n{history_str}\n</history>\n<user_latest>{raw_query}</user_latest>"

    try:
        rewritten = model.invoke([
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=prompt),
        ]).content.strip()
    except Exception as e:
        logger.warning(f"rewrite_query failed ({e}), using raw query")
        return raw_query

    result = rewritten or raw_query
    if result != raw_query:
        logger.info(f"rewrite: '{raw_query}' → '{result}'")
    return result
