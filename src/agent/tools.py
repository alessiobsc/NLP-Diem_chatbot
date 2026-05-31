"""
RAG tools for the agentic DIEM Chatbot.

Provides four composable tools:
- rewrite: Rewrite ambiguous query into standalone question using history
- retrieve: Search the DIEM knowledge base
- calculate: Apply academic calculations using retrieved formulas
"""

import re
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import hashlib

from config import CROSS_ENCODER_K, USE_RERANKER, USE_ADJACENT_RETRIEVAL
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_tools(retriever, generation_model, brain_ref) -> list:
    """Build the 4 RAG tools. brain_ref._last_docs is updated by retrieve()."""

    @tool
    def rewrite(query: str, state: Annotated[dict, InjectedState]) -> str:
        """Rewrite the user's latest message into a self-contained, standalone search query.
        Call BEFORE the first retrieve() of every turn — even for self-contained queries.
        rewrite() resolves pronouns, injects the academic year, and adapts phrasing to knowledge
        base terminology while keeping the query minimal. It does not add generic institutional
        terms unless needed to resolve an implicit reference. Skip ONLY if retrieve() was already
        called in this turn.
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
        history_str = "\n".join(history_lines[-6:])

        if query != user_query:
            logger.info(f"rewrite | ignoring agent-expanded query: '{query}'")

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
        If the returned context is empty or off-topic, retry with a rephrased or broader query
        (never retry with the identical query string).
        Returns formatted document excerpts as a string."""
        from src.agent.brain import rerank, format_context

        docs = retriever.invoke(query)

        # For each retrieved parent, also fetch the immediately following chunk
        # from the same source — captures content split across page boundaries.
        adjacent_added = 0
        if USE_ADJACENT_RETRIEVAL and docs:
            seen_ids = {doc.metadata.get("chunk_id") for doc in docs}
            extra = []
            for doc in docs:
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
            docs = docs + extra
            adjacent_added = len(extra)

        if USE_RERANKER and docs:
            final_docs = rerank(query, docs, top_n=CROSS_ENCODER_K)
        else:
            final_docs = docs

        # brain_ref._last_docs lets DiemBrain access the latest docs after graph completes
        brain_ref._last_docs = final_docs
        context = format_context({"docs": final_docs, "question": query, "history": []})["context"]

        logger.info(
            f"retrieve | query='{query[:80]}' | bi-encoder={len(docs) - adjacent_added} "
            f"| adjacent={adjacent_added} | reranker_used={USE_RERANKER} | final_docs={len(final_docs)} "
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
        import json
        from simpleeval import simple_eval
        from src.prompts import CALCULATE_PROMPT

        logger.info(f"calculate | operation='{operation}' | values={values}")
        user_content = (
            f"Operation: {operation}\n"
            f"Values: {values}\n\n"
            f"Context:\n{context}"
        )
        raw = generation_model.invoke([
            SystemMessage(content=CALCULATE_PROMPT),
            HumanMessage(content=user_content),
        ]).content.strip()

        # Strip markdown fences if model wraps the JSON
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        try:
            data = json.loads(raw)
        except Exception as e:
            logger.error(f"calculate | JSON parse failed: {e} | raw={raw}")
            return f"Errore nel parsing della risposta del modello: {raw}"

        if "error" in data:
            return data["error"]

        expression = data.get("expression", "")
        # Sanitize common LLM mistakes: Italian decimal comma and curly-brace math notation
        expression = re.sub(r'(\d),(\d)', r'\1.\2', expression)
        expression = expression.replace('{', '(').replace('}', ')')
        variables = {k: float(v) for k, v in data.get("variables", {}).items()}
        unit = data.get("unit", "")

        try:
            result = float(simple_eval(
                expression,
                names=variables,
                functions={"round": round, "min": min, "max": max, "abs": abs},
            ))
            result_str = str(round(result, 2))
            unit_str = f" {unit}" if unit else ""
            logger.info(f"calculate | expr={expression} vars={variables} result={result_str}")
            return f"Risultato: {result_str}{unit_str}\n(Espressione valutata: {expression})"
        except Exception as e:
            logger.error(f"calculate | eval failed: {e} | expr={expression} vars={variables}")
            return f"Errore nel calcolo: {e}"

    return [rewrite, retrieve, calculate]
