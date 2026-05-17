import base64
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

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
    "orario", "ricevimento", "riceve", "aula", "laboratorio", "erasmus", "borsa", "studente", "studenti",
    "piano", "regolamento", "offerta", "formativa", "magistrale", "triennale",
    "dottorato", "master", "faculty", "department", "enrollment", "grade",
    "average", "graduation", "thesis", "exam", "lecture", "semester",
}

_SCOPE_SYSTEM = (
    "You are a binary classifier for the DIEM chatbot — the virtual assistant of the "
    "Department of Information and Electrical Engineering and Applied Mathematics (DIEM) "
    "at the University of Salerno, Italy. "
    "DIEM covers topics such as: courses and degrees (computer science, electrical engineering, mathematics), "
    "professors and researchers, exams, academic regulations, labs, research projects, "
    "department services, Erasmus programs, and university life at UniSa. "
    "Respond with exactly one word: yes or no. No explanations, no punctuation."
)
_SCOPE_PROMPT = (
    "Is this question in scope for a university department assistant?\n\n"
    "Answer 'no' ONLY if the question is clearly about one of these off-topic categories:\n"
    "- Food, recipes, cooking\n"
    "- Sports, games, entertainment, music, movies\n"
    "- Politics or news unrelated to the university\n"
    "- Shopping, products, prices\n"
    "- Personal advice, relationships, health\n"
    "- General trivia or knowledge with no academic context\n"
    "- Questions about historical figures or celebrities with no university connection\n\n"
    "Answer 'yes' if the question:\n"
    "- Mentions any academic topic, course, exam, department, or university service\n"
    "- Asks about any person (could be a professor, researcher, or staff member)\n"
    "- Contains mainly pronouns or implicit references with no clearly off-topic content (likely a follow-up question)\n"
    "- Is ambiguous or could have any academic interpretation\n\n"
    "When in doubt, answer 'yes'."
)


_SCOPE_REJECTION = (
    "This question is outside my scope. "
    "I can only answer questions about DIEM (Department of Information and Electrical Engineering "
    "and Applied Mathematics) at the University of Salerno."
)

_OFFENSIVE_FALLBACK = (
    "Non posso elaborare questa richiesta poiché contiene contenuto inappropriato."
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
        tokens = set(question.lower().split())
        # Fast path: academic keyword match → in scope
        if tokens & _SCOPE_KEYWORDS:
            logger.debug("ScopeGuardrail: keyword match, in scope")
            return True

        # Slow path: ambiguous query → short yes/no LLM check
        try:
            response = self._model.invoke(
                [
                    SystemMessage(content=_SCOPE_SYSTEM),
                    HumanMessage(content=f"{_SCOPE_PROMPT}\n\nQuestion: {question}"),
                ],
                max_tokens=5,
            ).content.strip().lower()
        except Exception as e:
            logger.warning(f"ScopeGuardrail LLM call failed ({e}), defaulting to in-scope")
            return True  # fail open: prefer false negatives over false positives

        in_scope = "no" not in response
        if not in_scope:
            logger.info(f"ScopeGuardrail rejected: '{question[:60]}'")
        return in_scope


class OffensiveContentGuardrail:
    """Checks whether text contains offensive content via regex badword list."""

    def check(self, content: str) -> str | None:
        """Returns None if content is clean, or replacement string if offensive."""
        # Fast path: no badword regex match → clean, skip LLM
        if _BADWORDS_PATTERN is None or not _BADWORDS_PATTERN.search(content):
            logger.debug("OffensiveContentGuardrail: no match, clean")
            return None

        logger.warning("OffensiveContentGuardrail: badword match, replacing output")
        return _OFFENSIVE_FALLBACK


# ── PII redaction ──────────────────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """Redact email addresses; block entire response if credit card number detected."""
    # Credit card: block entirely rather than redact (card numbers in any form are sensitive)
    if _CARD_RE.search(text):
        return _PII_BLOCK_MSG
    # Email: replace with placeholder, preserve rest of text
    return _EMAIL_RE.sub("[EMAIL REDACTED]", text)
