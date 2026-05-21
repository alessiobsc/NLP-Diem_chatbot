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
    "piano", "regolamento", "offerta", "formativa", "magistrale", "triennale", "missione", "giunta",
    "dottorato", "master", "faculty", "department", "enrollment", "grade",
    "average", "graduation", "thesis", "exam", "lecture", "semester",
    # Academic person/role patterns — specific enough to be safe scope indicators
    "insegna", "insegnamento", "ricercatore", "pubblicazioni", "curriculum", "personale",
    # Administrative/contact data published on DIEM/UNISA site
    "fiscale", "iva", "footer", "indirizzo", "contatti", "pec",
}

_SCOPE_SYSTEM = (
    "You are a binary classifier for the DIEM chatbot — the virtual assistant of the "
    "Department of Information and Electrical Engineering and Applied Mathematics (DIEM) "
    "at the University of Salerno, Italy. "
    "DIEM covers topics such as: courses and degrees (computer science, electrical engineering, mathematics), "
    "professors and researchers, exams, academic regulations, labs, research projects, "
    "department services, Erasmus programs, and university life at UniSa. "
    "Administrative and contact data published on the DIEM or UNISA website — including "
    "P.IVA, codice fiscale, addresses, phone numbers, and PEC — are also in scope. "
    "Respond with exactly one word: yes or no. No explanations, no punctuation."
)
_SCOPE_PROMPT = (
    "Is this question in scope for a university department chatbot?\n\n"
    "Answer 'no' only if the question is clearly unrelated to any university or academic context. "
    "Examples that must get 'no':\n"
    "- 'What is the best pasta recipe?' (food/cooking)\n"
    "- 'How do I make tiramisu?' (food/cooking)\n"
    "- 'Who won the Champions League?' (sports)\n"
    "- 'What are the NBA playoffs results?' (sports)\n"
    "- 'Recommend me a Netflix series' (entertainment)\n"
    "- 'What are the best songs by Coldplay?' (music/entertainment)\n"
    "- 'What is the price of an iPhone?' (shopping/products)\n"
    "- 'Which car should I buy?' (shopping/products)\n"
    "- 'How do I fix a broken relationship?' (personal advice)\n"
    "- 'Should I break up with my partner?' (personal advice)\n"
    "- 'What are the symptoms of flu?' (medical advice)\n"
    "- 'How do I lose weight?' (health/medical)\n"
    "- 'Tell me a joke' (entertainment/trivia)\n"
    "- 'What is the capital of France?' (general trivia unrelated to university)\n"
    "- 'Who is Taylor Swift?' (celebrity with no university connection)\n"
    "- 'How do I invest in Bitcoin?' (finance/investment)\n"
    "- 'What is the weather forecast?' (general information)\n"
    "- 'Write me a poem about love' (creative writing, no academic context)\n"
    "These are examples — apply the same logic to any other topic that is clearly outside "
    "a university or academic context.\n\n"
    "Answer 'yes' for:\n"
    "- Anything touching a university, department, faculty, research, or academic life\n"
    "- Governance or administrative bodies (board, committee, giunta, consiglio...)\n"
    "- Third mission, public engagement, technology transfer, spin-off activities\n"
    "- Any person who could plausibly be a professor, researcher, or staff member\n"
    "- Ambiguous or follow-up questions — when in doubt, answer 'yes'."
)


_SCOPE_REJECTION = (
    "Mi dispiace, posso aiutarti solo con domande relative al DIEM e all'Università di Salerno — "
    "corsi di laurea, docenti, ricerca, regolamenti, laboratori e servizi del dipartimento."
)

_OFFENSIVE_FALLBACK = (
    "Non posso elaborare questa richiesta poiché contiene contenuto inappropriato."
)


# ── PII patterns ───────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Institutional domains (unisa.it and subdomains) — not redacted
_INSTITUTIONAL_EMAIL_RE = re.compile(r"@(?:[A-Za-z0-9-]+\.)*unisa\.it$", re.IGNORECASE)
# Matches credit card numbers in formatted groups (separator required between groups).
# Bare 16-digit document IDs/codes are NOT matched — they have no group separators.
_CARD_RE = re.compile(
    r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"   # Visa/MC: XXXX XXXX XXXX XXXX
    r"|\b\d{4}[ -]\d{6}[ -]\d{5}\b"             # Amex: XXXX XXXXXX XXXXX
    r"|\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{3}\b"   # 15-digit: XXXX XXXX XXXX XXX
)
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
    if _CARD_RE.search(text):
        logger.warning("redact_pii: credit card pattern detected, blocking response")
        return _PII_BLOCK_MSG
    def _maybe_redact(m: re.Match) -> str:
        email = m.group(0)
        if _INSTITUTIONAL_EMAIL_RE.search(email):
            return email
        return "[EMAIL REDACTED]"

    redacted = _EMAIL_RE.sub(_maybe_redact, text)
    if redacted != text:
        logger.info("redact_pii: email address redacted")
    return redacted
