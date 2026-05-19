import time
from urllib.parse import urlparse, parse_qs

import requests
from dotenv import load_dotenv

from config import (
    OPENROUTER_API_KEY, MAX_OPENROUTER_FAILURES, OPENROUTER_TIMEOUT_SECONDS,
    OPENROUTER_CONTEXT_HEADER_MODEL, OPENROUTER_ENDPOINT,
    OLLAMA_TIMEOUT_SECONDS, OLLAMA_MODEL, OLLAMA_ENDPOINT,
    MAX_OLLAMA_FAILURES, USE_LLM_CONTEXT_HEADERS,
)
from src.utils.logger import get_logger
from src.prompts import CONTEXT_HEADER_PROMPT
from .header_heuristic import (
    build_header_context,
    ensure_context_prefix,
    extract_year_tag,
    fallback_context_header,
    normalize_context_header,
)

load_dotenv()

logger = get_logger(__name__)

_HEADER_CACHE: dict = {}
_OPENROUTER_DISABLED = False
_OPENROUTER_FAILURES = 0
_OLLAMA_DISABLED = False
_OLLAMA_FAILURES = 0

# Domains where heuristic always produces optimal headers (e.g. professor name from title)
_HEURISTIC_ONLY_DOMAINS = frozenset({
    "docenti.unisa.it",
})


def _use_heuristic_for_url(url: str) -> bool:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()

    if netloc in _HEURISTIC_ONLY_DOMAINS:
        return True

    # corsi.unisa.it HTML pages: metadata title reliably contains course name → heuristic.
    # PDFs stored under /uploads/ have no course name in URL or title → LLM.
    if netloc == "corsi.unisa.it":
        return "/uploads/" not in path

    qs = parse_qs(parsed.query)
    # Listing pages (?stato= or ?tip=) contain many projects — LLM picks one at random
    if "stato" in qs or "tip" in qs:
        return True
    # progetti-finanziati without ?progetto= → listing page; with ?progetto= → single project (→ LLM)
    if "progetti-finanziati" in path and "progetto" not in qs:
        return True
    return False


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

    # Hybrid: use heuristic when URL/title already contain reliable signals
    if _use_heuristic_for_url(url):
        header = normalize_context_header(fallback_context_header(text, url, metadata), text, url, metadata)
        year_tag = extract_year_tag(url, metadata, text)
        if year_tag:
            header = f"{header} {year_tag}"
        header = ensure_context_prefix(header)
        _HEADER_CACHE[cache_key] = header
        return header

    if USE_LLM_CONTEXT_HEADERS:
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
    else:
        logger.debug(
            f"LLM context headers skipped: USE_LLM_CONTEXT_HEADERS is False; "
            f"source={url}; title={title[:120]}"
        )
        header = normalize_context_header(fallback_context_header(text, url, metadata), text, url, metadata)

    year_tag = extract_year_tag(url, metadata, text)
    if year_tag:
        header = f"{header} {year_tag}"

    header = ensure_context_prefix(header)
    _HEADER_CACHE[cache_key] = header
    return header


def add_context_headers(docs: list) -> None:
    logger.info("Adding contextual headers...")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        header = generate_context_header(doc.page_content, source, doc.metadata)
        doc.metadata["context_header"] = header
        if i % 100 == 0 or i == len(docs):
            logger.info(f"  -> {i}/{len(docs)} contextual headers added")