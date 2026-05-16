import base64
import os
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Badwords list ──────────────────────────────────────────────────────────────

def _load_badwords() -> set:
    data_path = os.path.join(os.path.dirname(__file__), "data", "badwords.b64")
    try:
        with open(data_path, "rb") as f:
            raw = base64.b64decode(f.read()).decode("utf-8")
        words = {w.strip().lower() for w in raw.splitlines() if w.strip()}
        logger.debug("Loaded %d badwords from %s", len(words), data_path)
        return words
    except Exception as e:
        logger.warning("Could not load badwords list (%s); offensive fast path disabled", e)
        return set()


_BADWORDS: set = _load_badwords()
_BADWORDS_PATTERN: re.Pattern | None = (
    re.compile(r"\b(" + "|".join(re.escape(w) for w in _BADWORDS) + r")\b", re.IGNORECASE)
    if _BADWORDS else None
)


# ── Scope keywords: any match → in scope without LLM call ─────────────────────

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

_SCOPE_PROMPT = (
    "Is this question about a university, academic department, courses, exams, professors, "
    "research, enrollment, grades, or student life? "
    "Answer 'no' ONLY if clearly unrelated (cooking, sports, politics, entertainment). "
    "When in doubt answer 'yes'. Answer only 'yes' or 'no'."
)

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


# ── PII patterns ───────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Matches 13-16 digit sequences with optional spaces/dashes (credit card pattern)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_PII_BLOCK_MSG = "Non posso fornire questa risposta per motivi di privacy."


# ── Guardrail classes ──────────────────────────────────────────────────────────

class ScopeGuardrail:
    """Checks whether a user question is in scope for the DIEM assistant."""

    def __init__(self, generation_model):
        self._model = generation_model

    def check(self, question: str) -> bool:
        """Returns True if in scope, False if out of scope."""
        # Fast path: any academic keyword → in scope, no LLM call needed
        tokens = set(question.lower().split())
        if tokens & _SCOPE_KEYWORDS:
            logger.debug("ScopeGuardrail: keyword match, in scope")
            return True

        # Slow path: ambiguous query → short yes/no LLM check
        prompt = f"{_SCOPE_PROMPT}\n\nQuestion: {question}"
        try:
            response = self._model.invoke(prompt, max_tokens=5).content.strip().lower()
        except Exception as e:
            logger.warning(f"ScopeGuardrail LLM call failed ({e}), defaulting to in-scope")
            return True  # fail open: prefer false negatives over false positives

        in_scope = not response.startswith("no")
        if not in_scope:
            logger.info(f"ScopeGuardrail rejected: '{question[:60]}'")
        return in_scope


class OffensiveContentGuardrail:
    """Checks whether text contains offensive content."""

    def __init__(self, generation_model):
        self._model = generation_model

    def check(self, content: str) -> str | None:
        """Returns None if content is clean, or replacement string if offensive."""
        # Fast path: no badword regex match → clean, skip LLM
        if _BADWORDS_PATTERN is None or not _BADWORDS_PATTERN.search(content):
            logger.debug("OffensiveContentGuardrail: no match, clean")
            return None

        # Slow path: keyword hit → LLM confirms whether truly offensive
        prompt = f"{_OFFENSIVE_PROMPT}\n\nText: {content[:500]}"
        try:
            response = self._model.invoke(prompt).content.strip().lower()
        except Exception as e:
            logger.warning(f"OffensiveContentGuardrail LLM call failed ({e}), passing through")
            return None

        if response.startswith("yes"):
            logger.warning("OffensiveContentGuardrail: offensive content detected, replacing")
            return _OFFENSIVE_FALLBACK
        return None


# ── PII redaction ──────────────────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """Redact email addresses; block entire response if credit card number detected."""
    # Credit card: block entirely rather than redact (card numbers in any form are sensitive)
    if _CARD_RE.search(text):
        return _PII_BLOCK_MSG
    # Email: replace with placeholder, preserve rest of text
    return _EMAIL_RE.sub("[EMAIL REDACTED]", text)
