"""
RAG tools for the agentic DIEM Chatbot.

Provides four composable tools:
- rewrite: Rewrite ambiguous query into standalone question using history
- retrieve: Search the DIEM knowledge base
- calculate: Apply academic calculations using retrieved formulas
"""

from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from config import CROSS_ENCODER_K
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_tools(retriever, generation_model, brain_ref) -> list:
    """Build the 4 RAG tools. brain_ref._last_docs is updated by retrieve()."""

    @tool
    def rewrite(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Rewrite the user's latest message into a self-contained, standalone search query.
        Call BEFORE retrieve() when the message:
        - contains pronouns or implicit references (lui, lei, suoi, questo, quale, quel, loro, ne)
        - is a follow-up that omits the subject (e.g. 'e i suoi orari?', 'cosa insegna?', 'quali corsi?')
        - is incomplete or poorly phrased
        - concerns academic content (courses, regulations, exams, degree programs) AND does not
          specify an academic year → the rewrite appends 'anno accademico 2025/2026'
        Returns the rewritten query as a string. After calling this tool you MUST immediately
        call retrieve() with the returned string as the query — do not modify it, do not generate an answer first."""
        from src.agent.brain import extract_text
        from src.prompts import REWRITE_PROMPT, REJECTION_TAGS
        from src.middleware import _SCOPE_REJECTION, _OFFENSIVE_FALLBACK

        _GUARDRAIL_PREFIXES = (
            _SCOPE_REJECTION[:40],
            _OFFENSIVE_FALLBACK[:40],
            "Mi dispiace, non sono riuscito",
        ) + tuple(t[:10] for t in REJECTION_TAGS)

        messages = state.get("messages", [])
        history_lines = []
        for msg in messages[:-1]:  # exclude current HumanMessage
            if isinstance(msg, HumanMessage) and not getattr(msg, "tool_calls", None):
                history_lines.append(f"User: {extract_text(msg.content)}")
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                text = extract_text(msg.content)
                # Skip guardrail-injected messages — they are not real answers and
                # cause the rewrite model to treat the next question as a follow-up
                # to a failed/rejected turn.
                if text and not any(text.startswith(p) for p in _GUARDRAIL_PREFIXES):
                    history_lines.append(f"AI: {text}")
        history_str = "\n".join(history_lines[-6:])

        prompt = f"<history>\n{history_str}\n</history>\n<user_latest>{query}</user_latest>"
        result = generation_model.invoke([
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=prompt),
        ])
        rewritten = result.content.strip()
        logger.info(f"rewrite: '{query}' -> '{rewritten}'")
        return rewritten

    @tool
    def retrieve(query: str) -> str:
        """Search the DIEM knowledge base and return relevant document excerpts.
        ALWAYS call this before generating any answer — context is mandatory.
        If rewrite() was called, pass its output EXACTLY as the query.
        If the returned context is empty or off-topic, retry with a rephrased or broader query
        (never retry with the identical query string).
        Returns formatted document excerpts as a string."""
        from src.agent.brain import rerank, format_context

        docs = retriever.invoke(query)
        reranked = rerank(query, docs, top_n=CROSS_ENCODER_K) if docs else []
        # brain_ref._last_docs lets DiemBrain access the latest docs after graph completes
        brain_ref._last_docs = reranked
        context = format_context({"docs": reranked, "question": query, "history": []})["context"]
        logger.info(
            f"retrieve | query='{query[:80]}' | bi-encoder={len(docs)} | reranked={len(reranked)} "
            f"| context_len={len(context)}"
        )
        return context

    @tool
    def calculate(context: str, operation: str, values: dict) -> str:
        """Apply an academic calculation using the official DIEM formula from retrieved context.
        Always call retrieve() FIRST to fetch the formula, then pass its output as context.
        Use for ANY numeric academic calculation: graduation grade, weighted average, TOLC thresholds.
        Never compute inline — always delegate to this tool.
        Parameters: context (retrieved formula text), operation (what to compute), values (input dict)."""
        from src.prompts import CALCULATE_PROMPT
        logger.info(f"calculate | operation='{operation}' | values={values}")
        user_content = (
            f"Operation: {operation}\n"
            f"Values: {values}\n\n"
            f"Context:\n{context}"
        )
        return generation_model.invoke([
            SystemMessage(content=CALCULATE_PROMPT),
            HumanMessage(content=user_content),
        ]).content

    return [rewrite, retrieve, calculate]
