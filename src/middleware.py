from typing import Any

from langchain.agents.middleware.pii import PIIMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage

from src.logger import get_logger

logger = get_logger(__name__)

_SCOPE_PROMPT = (
    "Is this question about a university, academic department, courses, exams, professors, "
    "research, enrollment, grades, or student life? "
    "Answer 'no' ONLY if clearly unrelated (cooking, sports, politics, entertainment). "
    "When in doubt answer 'yes'. Answer only 'yes' or 'no'."
)

# Keywords that immediately pass the scope check without any LLM call.
_SCOPE_KEYWORDS = {
    "diem", "unisa", "università", "universita", "salerno", "corso", "corsi",
    "esame", "esami", "laurea", "professore", "prof", "docente", "ricerca",
    "dipartimento", "informatica", "ingegneria", "matematica", "voto", "voti",
    "media", "cfu", "crediti", "tirocinio", "tesi", "iscrizione", "ammissione",
    "orario", "aula", "laboratorio", "erasmus", "borsa", "studente", "studenti",
    "piano", "regolamento", "offerta", "formativa", "magistrale", "triennale",
    "dottorato", "master", "faculty", "department", "enrollment", "grade",
    "average", "graduation", "thesis", "exam", "lecture", "semester",
}

_OFFENSIVE_PROMPT = (
    "Does this text contain offensive, harmful, or inappropriate content? "
    "Answer only 'yes' or 'no'."
)

_SCOPE_REJECTION = (
    "This question is outside my scope. "
    "I can only answer questions about DIEM (Department of Information and Electrical Engineering "
    "and Applied Mathematics) at the University of Salerno."
)

_OFFENSIVE_FALLBACK = (
    "Non posso fornire questa risposta. "
    "Per assistenza contatta la segreteria DIEM."
)


class ScopeGuardrail(AgentMiddleware):
    """before_agent hook: rejects out-of-scope queries before the agent loop."""

    def __init__(self, generation_model):
        self._model = generation_model

    def before_agent(self, state: AgentState, runtime: Any) -> dict | None:
        messages = state.get("messages", [])
        last_human = next(
            (m for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
            None,
        )
        if last_human is None:
            return None

        question = last_human.content if isinstance(last_human.content, str) else str(last_human.content)

        # Fast path: any academic keyword → in scope, skip LLM call entirely.
        tokens = set(question.lower().split())
        if tokens & _SCOPE_KEYWORDS:
            logger.debug("ScopeGuardrail: keyword match, passing through")
            return None

        # Slow path: ambiguous query → short LLM yes/no check.
        prompt = f"{_SCOPE_PROMPT}\n\nQuestion: {question}"
        try:
            response = self._model.invoke(prompt).content.strip().lower()
        except Exception as e:
            logger.warning(f"ScopeGuardrail LLM call failed ({e}), passing through")
            return None

        if response.startswith("no"):
            logger.info(f"ScopeGuardrail rejected: '{question[:60]}'")
            return {
                "messages": [AIMessage(content=_SCOPE_REJECTION)],
                "jump_to": "end",
            }
        return None


class OffensiveContentGuardrail(AgentMiddleware):
    """after_agent hook: replaces offensive content in the final AI message."""

    def __init__(self, generation_model):
        self._model = generation_model

    def after_agent(self, state: AgentState, runtime: Any) -> dict | None:
        messages = state.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        if last_ai is None:
            return None

        content = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
        prompt = f"{_OFFENSIVE_PROMPT}\n\nText: {content[:500]}"
        try:
            response = self._model.invoke(prompt).content.strip().lower()
        except Exception as e:
            logger.warning(f"OffensiveContentGuardrail LLM call failed ({e}), passing through")
            return None

        if response.startswith("yes"):
            logger.warning("OffensiveContentGuardrail replaced offensive output")
            return {"messages": [AIMessage(id=last_ai.id, content=_OFFENSIVE_FALLBACK)]}
        return None


def build_middleware(generation_model) -> list:
    """Return ordered list of middleware for create_agent."""
    return [
        ScopeGuardrail(generation_model),
        OffensiveContentGuardrail(generation_model),
        PIIMiddleware("email", strategy="redact", apply_to_input=False, apply_to_output=True),
        PIIMiddleware("credit_card", strategy="block", apply_to_output=True),
    ]
