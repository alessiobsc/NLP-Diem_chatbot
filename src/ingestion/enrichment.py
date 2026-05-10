import os

import requests
from dotenv import load_dotenv

from .parser import clean_text
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
_HEADER_CACHE: dict = {}
_OLLAMA_DISABLED = False


def fallback_context_header(text: str, url: str) -> str:
    combined = f"{url}\n{text[:700]}".lower()
    if "docenti.unisa.it" in combined or "professore" in combined or "docente" in combined:
        return "Profile and contact info of a DIEM professor."
    if "corsi.unisa.it" in combined or "corso di laurea" in combined or "insegnamento" in combined:
        return "Syllabus and information for a DIEM course."
    if "ufficio" in combined or "segreteria" in combined or "servizio" in combined:
        return "Physical location, office hours, and contact points for DIEM."
    if "avvisi" in combined or "avviso" in combined or "news" in combined:
        return "Official notice regarding DIEM."
    return "General information page about DIEM."


def generate_context_header(text: str, url: str) -> str:
    global _OLLAMA_DISABLED
    cache_key = (url, text[:500])
    if cache_key in _HEADER_CACHE:
        return _HEADER_CACHE[cache_key]

    if _OLLAMA_DISABLED:
        header = fallback_context_header(text, url)
        _HEADER_CACHE[cache_key] = header
        return header

    prompt = f"""
ROLE: Assistant for Academic Data Indexing.
TASK: Analyze the provided text from the University of Salerno, DIEM department.
OUTPUT: A single concise sentence, max 15 words, identifying the subject and context.

CONTEXT RULES:
If it is a person: "Profile and contact info of Prof. [Name] (DIEM)."
If it is a course: "Syllabus and info for the course [Course Name] (DIEM)."
If it is a location or office page: "Physical location, office hours, and contact points for DIEM."
If it is a notice: "Official notice regarding [Subject] at DIEM."
If the subject is unclear: "General information page about DIEM."

RULES:
Do not invent names, course titles, or subjects.
Use only information explicitly present in the text or URL.
Return only the sentence.
Do not add explanations, bullets, quotes, or labels.

TEXT:
{text[:1200]}

URL:
{url}

RESPONSE:
""".strip()

    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 40},
            },
            timeout=20,
        )
        response.raise_for_status()
        header = response.json().get("response", "").strip().splitlines()[0]
        header = clean_text(header.strip("\"' "))
        words = header.split()
        if not header or len(words) > 18:
            header = fallback_context_header(text, url)
    except Exception as e:
        _OLLAMA_DISABLED = True
        logger.warning(f"Ollama unavailable, using heuristic context headers: {e}")
        header = fallback_context_header(text, url)

    if not header.lower().startswith("context:"):
        header = f"Context: {header}"

    # Enforce compactness even if the local model is verbose.
    words = header.split()
    if len(words) > 15:
        header = " ".join(words[:15]).rstrip(".,;:") + "."

    _HEADER_CACHE[cache_key] = header
    return header


def add_context_headers(docs: list) -> None:
    logger.info("Adding contextual headers with Ollama...")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        header = generate_context_header(doc.page_content, source)
        doc.metadata["context_header"] = header
        doc.page_content = f"{header}\n\n{doc.page_content}"
        if i % 100 == 0 or i == len(docs):
            logger.info(f"  -> {i}/{len(docs)} contextual headers added")
