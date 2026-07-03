"""
RAG tools for the agentic DIEM Chatbot.

Provides two composable tools:
- rewrite: Rewrite ambiguous query into standalone question using history
- retrieve: Search the DIEM knowledge base
"""

import re
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import hashlib

from config import CROSS_ENCODER_K, USE_RERANKER, USE_ADJACENT_RETRIEVAL
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_tools(retriever, generation_model, brain_ref) -> list:
    """Build the RAG tools. brain_ref._last_docs is updated by retrieve()."""

    @tool
    def rewrite(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Rewrite the user's latest message into a self-contained, standalone search query.
        Call BEFORE every retrieve(), including retry retrieves — on retries, automatically
        produces a diversified query. rewrite() resolves pronouns, injects the academic year,
        and adapts phrasing to knowledge base terminology while keeping the query minimal.
        It does not add generic institutional terms unless needed to resolve an implicit reference.
        Returns the rewritten query as a string."""
        from src.agent.brain import extract_text
        from src.prompts import REWRITE_PROMPT, REJECTION_TAGS
        from src.middleware import _SCOPE_REJECTION, _OFFENSIVE_FALLBACK

        _GUARDRAIL_PREFIXES = (
            _SCOPE_REJECTION[:40],
            _OFFENSIVE_FALLBACK[:40],
            "Mi dispiace, non sono riuscito",
        ) + tuple(t[:10] for t in REJECTION_TAGS)

        messages = state.get("messages", [])
        last_human_idx = max(
            (i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)),
            default=-1,
        )
        user_query = (
            extract_text(messages[last_human_idx].content)
            if last_human_idx >= 0
            else query
        )
        history_lines = []
        for msg in messages[:last_human_idx]:
            if isinstance(msg, HumanMessage) and not getattr(msg, "tool_calls", None):
                history_lines.append(f"User: {extract_text(msg.content)}")
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                text = extract_text(msg.content)
                # Skip guardrail-injected messages — they are not real answers and
                # cause the rewrite model to treat the next question as a follow-up
                # to a failed/rejected turn.
                if text and not any(text.startswith(p) for p in _GUARDRAIL_PREFIXES):
                    history_lines.append(f"AI: {text}")
        history_str = "\n".join(history_lines[-4:])

        if query != user_query:
            logger.info(f"rewrite | ignoring agent-expanded query: '{query}'")

        # detect if this is a retry: find last rewrite ToolMessage after the current HumanMessage
        prior_rewrite = next(
            (m.content for m in reversed(messages[last_human_idx:])
             if isinstance(m, ToolMessage) and m.name == "rewrite"),
            None,
        )

        if prior_rewrite:
            prompt = (
                f"<history>\n{history_str}\n</history>\n"
                f"<user_latest>{user_query}</user_latest>\n"
                f"<previous_query>{prior_rewrite}</previous_query>\n"
                "<retry_instruction>The previous query returned insufficient results. "
                "CRITICAL: your output MUST differ from <previous_query>.\n"
                "Strategy (apply in order):\n"
                "1. If <previous_query> differs from <user_latest> (added terms, changed phrasing, "
                "injected institutional scope): generate a query closer to <user_latest>, "
                "removing added elements while preserving entities resolved from history. "
                "Skip to strategy 2 if this would still produce the same output as <previous_query>.\n"
                "2. Use a genuinely different formulation: replace key terms with synonyms, "
                "switch between question form and keyword form, or change the specificity level. "
                "Do NOT output <previous_query> verbatim or with only minor wording changes.</retry_instruction>"
            )
            logger.info(f"rewrite | retry detected | previous_query='{prior_rewrite[:80]}'")
        else:
            prompt = f"<history>\n{history_str}\n</history>\n<user_latest>{user_query}</user_latest>"

        result = generation_model.invoke([
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=prompt),
        ])
        rewritten = result.content.strip()
        logger.info(f"rewrite: '{user_query}' -> '{rewritten}'")
        return rewritten

    @tool
    def retrieve(query: str) -> str:
        """Search the DIEM knowledge base and return relevant document excerpts.
        ALWAYS call this before generating any answer — context is mandatory.
        If the returned context is empty, off-topic, or does not address the exact qualifier
        in the user's question (correct topic but wrong degree level, wrong person, or wrong
        year also counts as insufficient), call rewrite() then retry retrieve().
        Returns formatted document excerpts as a string."""
        from src.agent.brain import rerank, format_context

        docs = retriever.invoke(query)

        if USE_RERANKER and docs:
            final_docs = rerank(query, docs, top_n=CROSS_ENCODER_K)
        else:
            final_docs = docs

        # Fetch the immediately following chunk for each top-k result — done after
        # reranking so adjacent chunks are always included and not subject to reranker scoring.
        adjacent_added = 0
        if USE_ADJACENT_RETRIEVAL and final_docs:
            seen_ids = {doc.metadata.get("chunk_id") for doc in final_docs}
            extra = []
            for doc in final_docs:
                source = doc.metadata.get("source", "")
                chunk_index = doc.metadata.get("chunk_index")
                if chunk_index is None:
                    continue
                next_id = hashlib.md5(f"{source}:{chunk_index + 1}".encode()).hexdigest()[:16]
                if next_id in seen_ids:
                    continue
                result = retriever.docstore.mget([next_id])
                if result and result[0] is not None:
                    seen_ids.add(next_id)
                    extra.append(result[0])
            final_docs = final_docs + extra
            adjacent_added = len(extra)

        # brain_ref._last_docs lets DiemBrain access the latest docs after graph completes
        brain_ref._last_docs = final_docs
        context = format_context({"docs": final_docs, "question": query, "history": []})["context"]

        logger.info(
            f"retrieve | query='{query[:80]}' | bi-encoder={len(docs)} "
            f"| adjacent={adjacent_added} | reranker_used={USE_RERANKER} | final_docs={len(final_docs)} "
            f"| context_len={len(context)}"
        )
        return context

    return [rewrite, retrieve]
