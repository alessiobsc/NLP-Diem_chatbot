import os
import re

import requests
from dotenv import load_dotenv

from .parser import clean_text
from src.logger import get_logger
from src.prompts import CONTEXT_HEADER_PROMPT

load_dotenv()

logger = get_logger(__name__)
OLLAMA_MODEL = os.getenv("OLLAMA_ENRICHMENT_MODEL", "llama3.2:3b")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")

# TODO (Software Architect): Refactor global state variables (`_HEADER_CACHE`, `_OLLAMA_DISABLED`) into a dedicated class for better state management.
_HEADER_CACHE: dict = {}
_OLLAMA_DISABLED = False
HEADER_MAX_WORDS = 45
HEADER_INPUT_MAX_CHARS = 3200

HEADER_KEYWORDS = (
    "insegnamento", "corso", "corso di laurea", "curriculum", "docente",
    "professore", "programma", "obiettivi formativi", "modalità d'esame",
    "modalita d'esame", "esame", "ricevimento", "contatti", "email",
    "avviso", "news", "bando", "seminario", "evento", "regolamento",
    "pubblicazioni", "progetti", "ricerca", "laboratorio", "erasmus",
    "orario", "aula", "cfu", "anno accademico", "syllabus", "teaching",
    "research", "publications", "office hours",
)

NOISE_LINES = {
    "home", "condividi", "previous", "next", "precedente", "successiva",
    "tutte le news", "agenda", "docenti home", "didattica", "ricerca",
    "×", "-", "‹", "›",
}


def fallback_context_header(text: str, url: str) -> str:
    combined = f"{url}\n{text[:700]}".lower()
    if "docenti.unisa.it" in combined or "professore" in combined or "docente" in combined:
        return "Profilo docente: informazioni su ruolo, contatti, ricevimento o attivita didattiche e scientifiche di un docente collegato al DIEM."
    if "corsi.unisa.it" in combined or "corso di laurea" in combined or "insegnamento" in combined:
        return "Scheda insegnamento: informazioni su programma, obiettivi formativi, docente, CFU, esame o corso di laurea collegato al DIEM."
    if "ufficio" in combined or "segreteria" in combined or "servizio" in combined:
        return "Pagina DIEM: informazioni su uffici, servizi, sedi, orari, contatti o supporto amministrativo per studenti e utenti."
    if "avvisi" in combined or "avviso" in combined or "news" in combined:
        return "Avviso DIEM: comunicazione ufficiale su scadenze, eventi, bandi, seminari, lezioni o informazioni rivolte alla comunita DIEM."
    return "Pagina DIEM: informazioni istituzionali, didattiche, scientifiche o amministrative relative al Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica applicata."


def is_meaningful_line(line: str) -> bool:
    normalized = clean_text(line)
    lowered = normalized.lower()
    has_keyword = any(keyword in lowered for keyword in HEADER_KEYWORDS)

    if len(normalized) < 8:
        return False
    if lowered in NOISE_LINES:
        return False
    if "p.iva" in lowered or "c.f." in lowered:
        return False
    if has_keyword:
        return True
    if len(normalized) < 12 or lowered.count(" ") < 2:
        return False
    return True


def clean_passage(passage: str) -> str:
    lines = [line for line in passage.splitlines() if is_meaningful_line(line)]
    if lines:
        return clean_text("\n".join(lines))
    return clean_text(passage)


def unique_append(items: list[str], value: str, max_items: int) -> None:
    normalized = clean_text(value)
    if not normalized or normalized in items:
        return
    if len(items) < max_items:
        items.append(normalized)


def get_first_meaningful_lines(text: str, max_lines: int = 8) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if is_meaningful_line(line):
            unique_append(lines, line, max_lines)
        if len(lines) >= max_lines:
            break
    return lines


def get_keyword_passages(text: str, max_passages: int = 10) -> list[str]:
    passages: list[str] = []
    cleaned = clean_text(text)
    lowered = cleaned.lower()

    for keyword in HEADER_KEYWORDS:
        start = 0
        keyword_lower = keyword.lower()
        while len(passages) < max_passages:
            index = lowered.find(keyword_lower, start)
            if index == -1:
                break

            window_start = max(0, index - 140)
            window_end = min(len(cleaned), index + 320)
            passage = cleaned[window_start:window_end]
            passage = re.sub(r"^\S*\s*", "", passage) if window_start else passage
            passage = re.sub(r"\s*\S*$", "", passage) if window_end < len(cleaned) else passage
            passage = clean_passage(passage)

            if is_meaningful_line(passage):
                unique_append(passages, passage, max_passages)
            start = index + len(keyword_lower)

    return passages


def build_header_context(text: str, url: str) -> str:
    cleaned = clean_text(text)
    first_lines = get_first_meaningful_lines(cleaned)
    keyword_passages = get_keyword_passages(cleaned)

    sections = [f"SOURCE URL:\n{url}"]

    if first_lines:
        sections.append("FIRST MEANINGFUL LINES:\n" + "\n".join(f"- {line}" for line in first_lines))

    if keyword_passages:
        sections.append("KEY PASSAGES:\n" + "\n".join(f"- {passage}" for passage in keyword_passages))

    if cleaned:
        early_excerpt = cleaned[:900]
        mid_start = max(0, len(cleaned) // 2 - 450)
        mid_excerpt = cleaned[mid_start:mid_start + 900]
        sections.append(f"EARLY EXCERPT:\n{early_excerpt}")
        if mid_excerpt and mid_excerpt != early_excerpt:
            sections.append(f"MIDDLE EXCERPT:\n{mid_excerpt}")

    context = "\n\n".join(sections)
    return context[:HEADER_INPUT_MAX_CHARS]


def normalize_context_header(header: str, text: str, url: str) -> str:
    header = clean_text(header.strip("\"' "))
    header = re.sub(r"^[-*\d.\s]+", "", header)
    for _ in range(3):
        normalized = re.sub(
            r"^(context header|context|contesto|header|intestazione|response|risposta)\s*:\s*",
            "",
            header,
            flags=re.IGNORECASE,
        ).strip()
        if normalized == header:
            break
        header = normalized

    if not header:
        header = fallback_context_header(text, url)

    words = header.split()
    if len(words) > HEADER_MAX_WORDS:
        header = " ".join(words[:HEADER_MAX_WORDS]).rstrip(".,;:") + "."

    return header


def ensure_context_prefix(header: str) -> str:
    if not header.lower().startswith("context:"):
        return f"Context: {header}"
    return header


def generate_context_header(text: str, url: str) -> str:
    global _OLLAMA_DISABLED
    cache_key = (url, text[:500])
    if cache_key in _HEADER_CACHE:
        return _HEADER_CACHE[cache_key]

    if _OLLAMA_DISABLED:
        header = normalize_context_header(fallback_context_header(text, url), text, url)
        header = ensure_context_prefix(header)
        _HEADER_CACHE[cache_key] = header
        return header

    header_context = build_header_context(text, url)
    prompt = CONTEXT_HEADER_PROMPT.format(text=header_context, url=url)

    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 110},
            },
            timeout=20,
        )
        response.raise_for_status()
        header = response.json().get("response", "").strip().splitlines()[0]
        header = normalize_context_header(header, text, url)
    except Exception as e:
        _OLLAMA_DISABLED = True
        logger.warning(f"Ollama unavailable, using heuristic context headers: {e}")
        header = normalize_context_header(fallback_context_header(text, url), text, url)

    header = ensure_context_prefix(header)

    _HEADER_CACHE[cache_key] = header
    return header


def add_context_headers(docs: list) -> None:
    logger.info("Adding contextual headers with Ollama...")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        header = generate_context_header(doc.page_content, source)
        doc.metadata["context_header"] = header
        if i % 100 == 0 or i == len(docs):
            logger.info(f"  -> {i}/{len(docs)} contextual headers added")