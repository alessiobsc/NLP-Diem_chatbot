"""
Node implementations for the DIEM Chatbot LangGraph StateGraph.

DiemNodes is a mixin — DiemBrain inherits from it and provides the instance
attributes (models, retriever, guardrails) that node methods access via self.
"""
import datetime
import re
import uuid
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import CROSS_ENCODER_K
from src.agent.state import DiemState
from src.agent.utils import extract_text, format_context, rewrite_query
from src.encoders.reranker import rerank
from src.middleware import _SCOPE_REJECTION, _OFFENSIVE_FALLBACK, redact_pii
from src.prompts import get_agent_system_prompt, REJECTION_TAGS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prefixes that identify guardrail-injected AIMessages (not real agent responses).
# Used in _node_agent to replace them with a neutral placeholder before sending
# to the model, preserving Q&A pair structure without exposing rejection text.
_GUARDRAIL_PREFIXES = (
    _SCOPE_REJECTION[:40],
    _OFFENSIVE_FALLBACK[:40],
    "Mi dispiace, non sono riuscito",
) + tuple(t[:10] for t in REJECTION_TAGS)

_PLACEHOLDER = "[Risposta non disponibile per questo turno.]"

# matches Italian confirmation challenge phrases; ≤8 word guard prevents
# long dissatisfied messages from being mistaken for challenges
_CHALLENGE_RE = re.compile(r"\b(sicuro|sicura|certo|certa|davvero)\b", re.IGNORECASE)


class DiemNodes:
    """Mixin providing all LangGraph node implementations for DiemBrain."""

    # ── Shared helper ─────────────────────────────────────────────────────────

    def _block_if_offensive(self, content: str, msg_id: str | None = None) -> dict:
        """Return state update that replaces/injects AIMessage if content is offensive, else {}."""
        replacement = self._offensive_guardrail.check(content)
        if replacement is None:
            return {}
        kwargs: dict = {"content": replacement}
        if msg_id is not None:
            kwargs["id"] = msg_id
        return {"messages": [AIMessage(**kwargs)]}

    # ── Guard nodes ───────────────────────────────────────────────────────────

    def _node_input_guard(self, state: DiemState) -> dict:
        """Reject offensive user input before scope check."""
        question = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        return self._block_if_offensive(question)

    def _node_scope_guard(self, state: DiemState) -> dict:
        """Reject out-of-scope queries before retrieval.

        Returns {} (no state change) if in scope → _route_scope sends to reset_state.
        Returns {messages: [AIMessage(rejection)]} if OOT → _route_scope sends to END.
        """
        question = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        # Short follow-ups in multi-turn context (e.g. "Davvero?") have no DIEM keywords
        # and get misclassified as OOS by the LLM — rewrite first so scope check sees
        # the full resolved question instead of the raw fragment.
        has_prior = any(isinstance(m, HumanMessage) for m in state["messages"][:-1])
        check_query = question
        if has_prior:
            check_query = rewrite_query(question, state, self._lightweight_model) or question
            if check_query != question:
                logger.debug(f"scope_guard | rewrote short follow-up for scope check: '{question}' → '{check_query}'")
        if not self._scope_guardrail.check(check_query):
            return {"messages": [AIMessage(content=_SCOPE_REJECTION)]}
        return {}

    # ── Turn lifecycle nodes ──────────────────────────────────────────────────

    def _node_reset_state(self, state: DiemState) -> dict:
        """Reset per-turn state before the agent loop.

        Zeros tool_call_count, retrieved_context, and last_docs so each turn
        starts fresh. The agent will call retrieve() as its first action.
        """
        question = next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        logger.info(f"new_turn | user_question='{question[:120]}'")
        return {"tool_call_count": 0, "retrieved_context": "", "last_docs": []}

    # ── Agent node ────────────────────────────────────────────────────────────

    def _node_agent(self, state: DiemState) -> dict:
        """Agent decides whether to call more tools or generate the final answer.

        SYSTEM_PROMPT injected as SystemMessage at position 0.
        Dynamic hint injected on first invocation of each turn (tool_call_count==0,
        no retrieved_context) to ensure the agent always calls retrieve().
        Guardrail AIMessages in history are replaced with a neutral placeholder so
        the model sees proper Q&A pairs without being contaminated by rejection text.
        """
        system_content = get_agent_system_prompt()
        if state["tool_call_count"] == 0 and not state["retrieved_context"]:
            last_q = next(
                (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
                "",
            )
            has_prior = any(isinstance(m, HumanMessage) for m in state["messages"][:-1])
            is_challenge = has_prior and _CHALLENGE_RE.search(last_q) and len(last_q.split()) <= 8
            if is_challenge:
                system_content += (
                    "\n\nIMPORTANT: The user is challenging or asking you to verify your previous answer "
                    "(e.g. 'Sei sicuro?', 'Ne sei certo?', 'Davvero?'). "
                    "You MUST call rewrite() then retrieve() to re-verify — "
                    "do NOT answer from memory or previous context alone. "
                    "This is mandatory even if retrieved context from a prior turn seems sufficient."
                )
            else:
                system_content += (
                    "\n\nIMPORTANT: This is a new user question. "
                    "Call retrieve() before answering UNLESS the tool results already in this conversation "
                    "contain the specific, complete answer to this exact question. "
                    "If there is any doubt, call retrieve(). "
                    "rewrite() is optional preparation before retrieve(), not a substitute for it. "
                    "If a previous question was rejected or unanswered, ignore it — "
                    "it has NO bearing on whether or how you should answer the current question."
                )
        system = SystemMessage(content=system_content)

        # Find start of current turn: everything before the last HumanMessage is history
        human_indices = [i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)]
        last_human_idx = human_indices[-1] if human_indices else -1
        # sliding window: drop tool pairs older than 5 turns to prevent context bloat
        cutoff_idx = human_indices[-5] if len(human_indices) > 5 else 0

        dropped = sum(
            1 for i, m in enumerate(state["messages"])
            if i < cutoff_idx and (
                isinstance(m, ToolMessage)
                or (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            )
        )
        if dropped:
            logger.info(f"agent | sliding_window | kept_turns=5 | dropped_tool_msgs={dropped}")

        clean_messages = []
        for i, m in enumerate(state["messages"]):
            # strip ToolMessages and tool-call AIMessages from turns outside the window
            if i < cutoff_idx and (
                isinstance(m, ToolMessage)
                or (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            ):
                continue
            # replace guardrail-injected AIMessages with placeholder across all turns
            if (
                isinstance(m, AIMessage)
                and not getattr(m, "tool_calls", None)
                and (
                    not (m.content if isinstance(m.content, str) else "").strip()
                    or any(
                        (m.content if isinstance(m.content, str) else "").startswith(p)
                        for p in _GUARDRAIL_PREFIXES
                    )
                )
            ):
                clean_messages.append(AIMessage(id=m.id, content=_PLACEHOLDER))
            else:
                clean_messages.append(m)
        response = self._agent_model_with_tools.invoke([system] + clean_messages)
        tool_calls = getattr(response, "tool_calls", None)

        # rewrite() must receive the user's actual latest message. The routing
        # model may otherwise expand it with generic institutional terms before
        # the dedicated rewrite model gets a chance to normalize it.
        if tool_calls and any(tc["name"] == "rewrite" for tc in tool_calls):
            user_question = (
                extract_text(state["messages"][last_human_idx].content)
                if last_human_idx >= 0
                else ""
            )
            overridden = []
            for tc in tool_calls:
                if tc["name"] == "rewrite":
                    tc = {**tc, "args": {**tc["args"], "query": user_question}}
                    logger.info(f"agent | overriding rewrite input with user question: '{user_question[:80]}'")
                overridden.append(tc)
            response = AIMessage(id=response.id, content=response.content, tool_calls=overridden)
            tool_calls = overridden

        # If agent is calling retrieve, check if rewrite was already called this turn.
        # If so, force the retrieve query to use the rewrite output — model compliance is unreliable.
        if tool_calls and any(tc["name"] == "retrieve" for tc in tool_calls):
            current_turn_start = last_human_idx if last_human_idx >= 0 else 0
            rewrite_output = next(
                (m.content for m in reversed(state["messages"][current_turn_start:])
                 if isinstance(m, ToolMessage) and m.name == "rewrite"),
                None,
            )
            if rewrite_output:
                overridden = []
                for tc in tool_calls:
                    if tc["name"] == "retrieve":
                        tc = {**tc, "args": {**tc["args"], "query": rewrite_output}}
                        logger.info(f"agent | overriding retrieve query with rewrite output: '{rewrite_output[:80]}'")
                    overridden.append(tc)
                response = AIMessage(id=response.id, content=response.content, tool_calls=overridden)
                tool_calls = overridden

        if tool_calls:
            names = [tc["name"] for tc in tool_calls]
            logger.info(f"agent | retrieve_count={state['tool_call_count']} | calling tools={names}")
        else:
            logger.info(
                f"agent | retrieve_count={state['tool_call_count']} | generating final answer "
                f"| answer_len={len(response.content) if isinstance(response.content, str) else 0}"
            )
        return {"messages": [response]}

    def _node_forced_retrieve(self, state: DiemState) -> dict:
        """Agent generated empty content without calling retrieve — force a retrieve.

        Extracts the current user question, calls the retriever directly, and injects
        a synthetic AIMessage+ToolMessage pair so the subsequent agent invocation
        receives valid context. Only triggered when tool_call_count==0 and
        retrieved_context=="" (i.e. agent skipped all tools on the very first call).
        """
        last_rewrite = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, ToolMessage) and m.name == "rewrite"
             and not m.content.startswith("[Skipped")),
            None,
        )
        question = last_rewrite or next(
            (extract_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        logger.warning(f"forced_retrieve | forcing retrieve | query='{question[:80]}' | from_rewrite={last_rewrite is not None}")
        docs = self._retriever.invoke(question)
        reranked = rerank(question, docs, top_n=CROSS_ENCODER_K) if docs else []
        self._last_docs = reranked
        context = format_context({"docs": reranked, "question": question, "history": []})["context"]
        logger.info(f"forced_retrieve | reranked={len(reranked)} | context_len={len(context)}")
        # Stub pending tool_calls from last agent message (e.g. rewrite loop cap hit)
        stubs = []
        last_ai = state["messages"][-1]
        for tc in getattr(last_ai, "tool_calls", []):
            stubs.append(ToolMessage(
                content="[Skipped — forcing retrieve]",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
        tc_id = str(uuid.uuid4())
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "retrieve", "args": {"query": question}, "id": tc_id, "type": "tool_call"}],
        )
        tool_msg = ToolMessage(content=context, tool_call_id=tc_id, name="retrieve")
        return {
            "messages": stubs + [ai_msg, tool_msg],
            "tool_call_count": state["tool_call_count"] + 1,
            "retrieved_context": context,
            "last_docs": reranked,
        }

    def _node_force_answer(self, state: DiemState) -> dict:
        """Retrieve cap hit with pending tool_calls: stub them, then force final answer.

        Synthetic ToolMessages resolve dangling tool_calls so the message history
        stays valid for the model API, then the agent model (without tools bound)
        generates a final answer from whatever context was already retrieved.

        Uses a minimal system prompt (not the full agent prompt) to avoid confusing
        the model with tool instructions when it cannot call any tools. Injects
        retrieved_context explicitly so the model does not have to reconstruct it
        from a pruned/placeholder-filled history.
        """
        messages = list(state["messages"])
        last_ai = messages[-1]
        extra: List[ToolMessage] = []
        for tc in getattr(last_ai, "tool_calls", []):
            extra.append(ToolMessage(
                content="[Tool call limit reached — result unavailable]",
                tool_call_id=tc["id"],
                name=tc["name"],
            ))
        logger.warning(
            f"force_answer | MAX_RETRIEVE_CALLS reached | stubbed {len(extra)} pending tool_call(s)"
        )

        context = state.get("retrieved_context", "")
        context_block = f"\n\n<document>\n{context}\n</document>" if context else ""

        system = SystemMessage(content=(
            "Sei un assistente virtuale del DIEM (Dipartimento di Ingegneria dell'Informazione ed Elettrica "
            "e Matematica Applicata) dell'Università di Salerno. "
            "Hai raggiunto il limite massimo di chiamate agli strumenti. "
            "Genera ORA una risposta completa e diretta alla domanda dell'utente "
            "usando ESCLUSIVAMENTE il contesto recuperato qui sotto. "
            "Non chiamare strumenti. Non rispondere con una sola parola."
            + context_block
        ))

        response = self._agent_model.invoke([system] + messages + extra)

        # Guard against degenerate single-word outputs (e.g. "yes", "sì", "ok")
        content = (response.content if isinstance(response.content, str) else "").strip()
        if len(content) < 30:
            logger.warning(f"force_answer | degenerate output ({len(content)} chars): {content!r} — using fallback")
            response = AIMessage(content=(
                "Mi dispiace, non sono riuscito a trovare informazioni sufficienti per rispondere. "
                "Prova a riformulare la domanda o a essere più specifico."
            ))

        return {"messages": extra + [response]}

    def _node_output_guard(self, state: DiemState) -> dict:
        """Offensive content check + PII redaction on the final AIMessage.

        Returns {} if content is clean (no state change).
        Returns updated messages if content was replaced or redacted.
        """
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is None:
            return {}

        content = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)

        # Empty or degenerate answer (e.g. "yes", "sì", "ok"): replace with fallback.
        if len(content.strip()) < 30:
            logger.warning(f"output_guard: degenerate answer ({len(content.strip())} chars): {content.strip()!r} — replacing with fallback")
            return {"messages": [AIMessage(
                id=last_ai.id,
                content="Mi dispiace, non sono riuscito a trovare informazioni sufficienti per rispondere a questa domanda.",
            )]}

        blocked = self._block_if_offensive(content, msg_id=last_ai.id)
        if blocked:
            return blocked

        redacted = redact_pii(content)
        if redacted != content:
            return {"messages": [AIMessage(id=last_ai.id, content=redacted)]}

        return {}
