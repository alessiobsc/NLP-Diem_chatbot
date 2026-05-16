import os
import re
import time
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv

from config import OPENROUTER_API_KEY
from .parser import clean_text
from src.logger import get_logger
from src.prompts import CONTEXT_HEADER_PROMPT

load_dotenv()

logger = get_logger(__name__)
OPENROUTER_ENDPOINT = os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_CONTEXT_HEADER_MODEL = os.getenv("OPENROUTER_CONTEXT_HEADER_MODEL", "mistralai/mistral-nemo")
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_CONTEXT_HEADER_TIMEOUT", "30"))
OLLAMA_MODEL = os.getenv("OLLAMA_ENRICHMENT_MODEL", "qwen2.5:3b")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")

# TODO (Software Architect): Refactor global state variables (`_HEADER_CACHE`, `_OLLAMA_DISABLED`) into a dedicated class for better state management.
_HEADER_CACHE: dict = {}
_OPENROUTER_DISABLED = False
_OPENROUTER_FAILURES = 0
_OLLAMA_DISABLED = False
_OLLAMA_FAILURES = 0
MAX_OPENROUTER_FAILURES = int(os.getenv("OPENROUTER_CONTEXT_HEADER_MAX_FAILURES", "3"))
MAX_OLLAMA_FAILURES = int(os.getenv("OLLAMA_ENRICHMENT_MAX_FAILURES", "5"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_ENRICHMENT_TIMEOUT", "10"))
HEADER_MAX_WORDS = 18
HEADER_INPUT_MAX_CHARS = 2200

ASSERTIVE_HEADER_PATTERNS = (
    "contiene informazioni",
    "fornisce",
    "gestisce",
    "prepara",
    "ha la finalità",
    "ha la finalita",
    "è richiesto",
    "e richiesto",
    "sono richiesti",
    "possono immatricolarsi",
    "consente",
    "permette",
)

GENERIC_HEADER_PATTERNS = (
    "informazioni generali",
    "pagina generale",
    "general information",
    "dettagli riguardanti",
    "informazioni sul profilo",
    "servizio |",
    "pagina istituzionale |",
    "servizio | profilo docente",
    "servizio | progetto di ricerca",
    "pagina istituzionale | profilo docente",
)

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


def title_from_url(url: str) -> str:
    path = unquote(urlparse(url).path).strip("/")
    if not path:
        return ""
    leaf = path.rsplit("/", 1)[-1]
    leaf = re.sub(r"\.(html?|pdf)$", "", leaf, flags=re.IGNORECASE)
    leaf = leaf.replace("-", " ").replace("_", " ")
    return clean_text(leaf)


def header_detail_from_text(text: str) -> str:
    lowered = text[:1800].lower()
    if any(term in lowered for term in ("requisiti di accesso", "conoscenze richieste", "tolc", "ofa")):
        return "requisiti di accesso"
    if any(term in lowered for term in ("piano degli studi", "piano di studi", "attività formative", "attivita formative")):
        return "piano degli studi"
    if any(term in lowered for term in ("obiettivi formativi", "risultati di apprendimento")):
        return "obiettivi formativi"
    if any(term in lowered for term in ("sbocchi occupazionali", "profilo professionale")):
        return "sbocchi occupazionali"
    if any(term in lowered for term in ("consultazione", "parti interessate", "organizzazioni rappresentative")):
        return "consultazione parti interessate"
    if "tirocinio" in lowered:
        return "tirocinio"
    if "prova finale" in lowered:
        return "prova finale"
    if "ricevimento" in lowered:
        return "ricevimento"
    if "pubblicazioni" in lowered:
        return "pubblicazioni"
    return ""


def metadata_text(metadata: dict | None) -> str:
    if not metadata:
        return ""
    keys = ("title", "source_page", "content_type", "date", "temporal_filter", "language")
    lines = []
    for key in keys:
        value = metadata.get(key)
        if value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def classify_context_header(text: str, url: str, metadata: dict | None = None) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = unquote(parsed.path).lower()
    meta_text = metadata_text(metadata)
    combined = f"{url}\n{meta_text}\n{text[:1200]}".lower()
    detail = header_detail_from_text(text)
    title = clean_text(str((metadata or {}).get("title", "")))

    if "__schede-sua" in path:
        base = "Scheda SUA corso di studio"
        return f"{base} - {detail}" if detail else base

    if "__regolamenti-cds" in path or "regolamento" in combined:
        base = "Regolamento corso di studio"
        return f"{base} - {detail}" if detail else base

    if "docenti.unisa.it" in host:
        person = title.split("|", 1)[0].strip()
        if "pubblicazioni" in path or "pubblicazioni" in combined:
            return f"Pubblicazioni docente - {person}" if person else "Pubblicazioni docente"
        if "curriculum" in path or "curriculum" in combined:
            return f"Curriculum docente - {person}" if person else "Curriculum docente"
        if "ricevimento" in path or "ricevimento" in combined:
            return f"Ricevimento docente - {person}" if person else "Ricevimento docente"
        if "didattica" in path or "insegnamenti" in path:
            return f"Didattica docente - {person}" if person else "Didattica docente"
        return f"Profilo docente - {person}" if person else "Profilo docente"

    if "corsi.unisa.it" in host:
        if "insegnament" in combined or re.search(r"/\d{10,}/", path):
            return "Scheda insegnamento"
        base = "Pagina corso di studio"
        return f"{base} - {detail}" if detail else base

    if "progetti-finanziati" in path or "progetti finanziati" in combined:
        return "Progetti finanziati DIEM"
    if any(term in combined for term in ("avviso", "avvisi", "news", "bando", "seminario", "evento")):
        return "Avviso DIEM"
    if "laborator" in combined:
        return "Laboratorio DIEM"
    if any(term in combined for term in ("segreteria", "ufficio", "contatti")):
        return "Servizi e contatti DIEM"
    if "didattica" in combined:
        return "Didattica DIEM"

    fallback_title = title or title_from_url(url)
    return f"Pagina DIEM - {fallback_title}" if fallback_title else "Pagina DIEM"


def fallback_context_header(text: str, url: str, metadata: dict | None = None) -> str:
    return classify_context_header(text, url, metadata)


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


def build_header_context(text: str, url: str, metadata: dict | None = None) -> str:
    cleaned = clean_text(text)
    first_lines = get_first_meaningful_lines(cleaned)
    keyword_passages = get_keyword_passages(cleaned)

    sections = [f"SOURCE URL:\n{url}"]
    meta_text = metadata_text(metadata)
    if meta_text:
        sections.append(f"GLOBAL METADATA:\n{meta_text}")

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


def normalize_context_header(header: str, text: str, url: str, metadata: dict | None = None) -> str:
    fallback = fallback_context_header(text, url, metadata)
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

    lowered = header.lower()
    evidence = f"{url}\n{metadata_text(metadata)}\n{text[:1500]}".lower()
    unsupported_research_project = (
        "progetto di ricerca" in lowered
        and not any(term in evidence for term in ("progetto di ricerca", "progetti finanziati", "ricerca/progetti", "finanziat"))
    )
    if (
        not header
        or any(pattern in lowered for pattern in ASSERTIVE_HEADER_PATTERNS)
        or any(pattern in lowered for pattern in GENERIC_HEADER_PATTERNS)
        or unsupported_research_project
        or len(header.split("|")) > 3
    ):
        header = fallback

    words = header.split()
    if len(words) > HEADER_MAX_WORDS:
        header = " ".join(words[:HEADER_MAX_WORDS]).rstrip(".,;:")

    header = header.rstrip(".,;:")

    return header


def ensure_context_prefix(header: str) -> str:
    if not header.lower().startswith("context:"):
        return f"Context: {header}"
    return header


def generate_context_header(text: str, url: str, metadata: dict | None = None) -> str:
    global _OPENROUTER_DISABLED, _OPENROUTER_FAILURES, _OLLAMA_DISABLED, _OLLAMA_FAILURES
    title = str((metadata or {}).get("title", ""))
    source_page = str((metadata or {}).get("source_page", ""))
    cache_key = (url, title, source_page, text[:500])
    if cache_key in _HEADER_CACHE:
        return _HEADER_CACHE[cache_key]

    header_context = build_header_context(text, url, metadata)
    prompt = CONTEXT_HEADER_PROMPT.format(text=header_context, url=url)
    header = ""

    if OPENROUTER_API_KEY and not _OPENROUTER_DISABLED:
        try:
            request_start = time.time()
            logger.debug(
                "OpenRouter header request start: "
                f"model={OPENROUTER_CONTEXT_HEADER_MODEL}; timeout={OPENROUTER_TIMEOUT_SECONDS}s; "
                f"source={url}; title={title[:120]}; "
                f"parent_chars={len(text or '')}; prompt_chars={len(prompt)}"
            )
            response = requests.post(
                OPENROUTER_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/local/diem-chatbot",
                    "X-Title": "DIEM Context Header Enrichment",
                },
                json={
                    "model": OPENROUTER_CONTEXT_HEADER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 60,
                },
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            raw_header = response.json()["choices"][0]["message"]["content"].strip().splitlines()[0]
            header = normalize_context_header(raw_header, text, url, metadata)
            _OPENROUTER_FAILURES = 0
            elapsed = time.time() - request_start
            logger.debug(
                "OpenRouter header request ok: "
                f"elapsed={elapsed:.2f}s; source={url}; header={header}"
            )
        except Exception as e:
            _OPENROUTER_FAILURES += 1
            if _OPENROUTER_FAILURES >= MAX_OPENROUTER_FAILURES:
                _OPENROUTER_DISABLED = True
                logger.warning(
                    "OpenRouter enrichment disabled after "
                    f"{_OPENROUTER_FAILURES} consecutive failures; falling back to local Ollama. "
                    f"source={url}; title={title[:120]}; parent_chars={len(text or '')}; "
                    f"Last error: {e}"
                )
            else:
                logger.warning(
                    "OpenRouter enrichment failed for one parent "
                    f"({_OPENROUTER_FAILURES}/{MAX_OPENROUTER_FAILURES} consecutive failures); "
                    f"falling back to local Ollama. "
                    f"source={url}; title={title[:120]}; parent_chars={len(text or '')}; "
                    f"Error: {e}"
                )
    elif not OPENROUTER_API_KEY:
        logger.debug("OpenRouter context headers skipped: OPENROUTER_API_KEY is missing")

    if not header:
        if _OLLAMA_DISABLED:
            logger.debug(
                "Header fallback used: reason=ollama_disabled; "
                f"source={url}; title={title[:120]}; parent_chars={len(text or '')}"
            )
            header = normalize_context_header(fallback_context_header(text, url, metadata), text, url, metadata)
        else:
            try:
                request_start = time.time()
                logger.debug(
                    "Ollama header request start: "
                    f"model={OLLAMA_MODEL}; timeout={OLLAMA_TIMEOUT_SECONDS}s; "
                    f"source={url}; title={title[:120]}; "
                    f"parent_chars={len(text or '')}; prompt_chars={len(prompt)}"
                )
                response = requests.post(
                    OLLAMA_ENDPOINT,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 60},
                    },
                    timeout=OLLAMA_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                raw_header = response.json().get("response", "").strip().splitlines()[0]
                header = normalize_context_header(raw_header, text, url, metadata)
                _OLLAMA_FAILURES = 0
                elapsed = time.time() - request_start
                logger.debug(
                    "Ollama header request ok: "
                    f"elapsed={elapsed:.2f}s; source={url}; header={header}"
                )
            except Exception as e:
                _OLLAMA_FAILURES += 1
                if _OLLAMA_FAILURES >= MAX_OLLAMA_FAILURES:
                    _OLLAMA_DISABLED = True
                    logger.warning(
                        "Ollama enrichment disabled after "
                        f"{_OLLAMA_FAILURES} consecutive failures; using heuristic context headers. "
                        f"source={url}; title={title[:120]}; parent_chars={len(text or '')}; "
                        f"Last error: {e}"
                    )
                else:
                    logger.warning(
                        "Ollama enrichment failed for one parent "
                        f"({_OLLAMA_FAILURES}/{MAX_OLLAMA_FAILURES} consecutive failures); "
                        f"using heuristic header for this parent. "
                        f"source={url}; title={title[:120]}; parent_chars={len(text or '')}; "
                        f"Error: {e}"
                    )
                logger.debug(
                    "Header fallback used: reason=ollama_exception; "
                    f"source={url}; title={title[:120]}; parent_chars={len(text or '')}"
                )
                header = normalize_context_header(fallback_context_header(text, url, metadata), text, url, metadata)

    header = ensure_context_prefix(header)

    _HEADER_CACHE[cache_key] = header
    return header


def add_context_headers(docs: list) -> None:
    logger.info("Adding contextual headers with Ollama...")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        header = generate_context_header(doc.page_content, source, doc.metadata)
        doc.metadata["context_header"] = header
        if i % 100 == 0 or i == len(docs):
            logger.info(f"  -> {i}/{len(docs)} contextual headers added")