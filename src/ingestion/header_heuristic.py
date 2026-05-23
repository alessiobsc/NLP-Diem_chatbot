import re
from urllib.parse import unquote, urlparse, parse_qs

from .parser import clean_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_YEAR = 2020
_MAX_YEAR = 2035

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

GENERIC_LOCAL_THEMES = {
    "",
    "documento",
    "pagina",
    "pagina corso di studio",
    "profilo docente",
    "docente",
    "docenti",
    "docente/personale",
    "docente/personale - profilo docente",
    "docente - profilo docente",
    "scheda corso di studio",
    "scheda sua corso di studio",
    "scheda sua",
    "corso di studio",
    "regolamento corso di studio",
    "regolamento",
}

_NON_COURSE_SEGS = frozenset({"uploads", "rescue", "public", "assets", "static", "home", "index", "pdf"})


# ---------------------------------------------------------------------------
# Year tag
# ---------------------------------------------------------------------------

def _is_academic_year_context(parsed) -> bool:
    path = parsed.path.lower()
    return (
        "__regolamenti-cds" in path
        or "__schede-sua" in path
        or "/didattica" in path
        or "/insegnament" in path
    )


def _is_publication_context(parsed) -> bool:
    path = parsed.path.lower()
    return "pubblicazioni" in path or "iris" in path


def extract_year_tag(url: str, metadata: dict | None, text: str | None = None) -> str:
    """Return [YYYY] or [AY YYYY/YYYY+1] tag if a valid year >= _MIN_YEAR is found, else ''."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    year: int | None = None

    # 1. Query param ?anno=YYYY — skip 0 (means "all years")
    if "anno" in qs:
        val = qs["anno"][0]
        if re.match(r"^\d{4}$", val):
            candidate = int(val)
            if _MIN_YEAR <= candidate <= _MAX_YEAR:
                year = candidate

    # 2. Path segment /YYYY/ (e.g. PDF files)
    if year is None:
        for py in reversed(re.findall(r"/(\d{4})/", parsed.path)):
            candidate = int(py)
            if _MIN_YEAR <= candidate <= _MAX_YEAR:
                year = candidate
                break

    # 3. Metadata date field
    if year is None and metadata:
        for key in ("date", "created", "modified", "publication_date"):
            val = str(metadata.get(key) or "")
            m = re.search(r"\b(20\d{2})\b", val)
            if m:
                candidate = int(m.group(1))
                if _MIN_YEAR <= candidate <= _MAX_YEAR:
                    year = candidate
                    break

    # 4. Year in URL filename (e.g. decreto-28.10.2025-bando.pdf)
    if year is None:
        filename = parsed.path.rstrip("/").split("/")[-1]
        m = re.search(r"\b(20\d{2})\b", filename)
        if m:
            candidate = int(m.group(1))
            if _MIN_YEAR <= candidate <= _MAX_YEAR:
                year = candidate

    # 5. Academic year pattern YYYY/YYYY+1 in text (reliable, specific)
    if year is None and text:
        for m in re.finditer(r"\b(20\d{2})/(20\d{2})\b", text):
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y2 == y1 + 1 and _MIN_YEAR <= y1 <= _MAX_YEAR:
                year = y1
                break

    if year is None:
        return ""

    if _is_academic_year_context(parsed):
        return f"[AY {year}/{year + 1}]"
    return f"[{year}]"


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

def _course_slug_from_path(path: str, marker: str) -> str:
    """Extract course slug from corsi.unisa.it path before the given URL marker."""
    if marker not in path:
        return ""
    before = path.split(marker)[0].strip("/")
    slug = before.rsplit("/", 1)[-1] if "/" in before else before
    if (not slug or slug.startswith("__") or slug in _NON_COURSE_SEGS
            or re.match(r"^\d", slug) or re.search(r"\.\w{2,4}$", slug)):
        return ""
    return re.sub(r"[_-]+", " ", slug).strip().title()


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
        course_name = _course_slug_from_path(path, "__schede-sua") if "corsi.unisa.it" in host else ""
        if course_name:
            return f"{base} - {course_name}"
        return f"{base} - {detail}" if detail else base

    # URL-only — "regolamento in combined" too broad (bandi cite regolamento in body text)
    if "__regolamenti-cds" in path:
        base = "Regolamento corso di studio"
        course_name = _course_slug_from_path(path, "__regolamenti-cds") if "corsi.unisa.it" in host else ""
        if course_name:
            return f"{base} - {course_name}"
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
            return f"Corsi insegnati - {person}" if person else "Corsi insegnati"
        return f"Profilo docente - {person}" if person else "Profilo docente"

    if "easycourse.unisa.it" in host:
        qs = parse_qs(parsed.query)
        view = qs.get("view", [""])[0].lower()
        cat = "calendario esami" if "easytest" in view else "orario lezioni"
        insegnamento = re.search(r'Insegnamento:\s*(.+)', text)
        label = insegnamento.group(1).strip().title() if insegnamento else ""
        return f"{cat} - {label}" if label else cat

    if "corsi.unisa.it" in host:
        if "insegnament" in combined or re.search(r"/\d{10,}/", path):
            return "Scheda insegnamento"
        # Prefer metadata title; discard if it's a generic doc-type label
        course_name = title.split("|", 1)[0].strip() if title else ""
        _GENERIC_LABELS = {"regolamento", "regolamento corso di studio", "offerta formativa",
                           "piano degli studi", "scheda sua", "consultazione parti interessate"}
        if course_name.lower() in _GENERIC_LABELS:
            course_name = ""
        # Fallback: first meaningful path segment is the course slug on corsi.unisa.it
        if not course_name:
            segs = [s for s in path.strip("/").split("/")
                    if s and not s.startswith("__") and s not in _NON_COURSE_SEGS
                    and not re.match(r"^\d", s) and not re.search(r"\.\w{2,4}$", s)]
            if segs:
                course_name = re.sub(r"[_-]+", " ", segs[0]).strip().title()
        base = "Regolamento corso di studio" if "regolamento" in path else "Pagina corso di studio"
        if course_name:
            return f"{base} - {course_name}"
        return f"{base} - {detail}" if detail else base

    # URL path checks before broad text checks — prevents "laborator in text" false positives
    if "/international/" in path or "/erasmus" in path:
        return "Accordi internazionali DIEM"
    if "/aree-di-ricerca" in path:
        return "Aree di ricerca DIEM"
    if "/ricerca/laboratori" in path or "/laboratori/" in path:
        return "Laboratorio DIEM"
    if "/premi-ricerca" in path:
        return "Premi ricerca DIEM"
    if "/terza-missione" in path:
        return "Terza missione DIEM"
    if "progetti-finanziati" in path:
        return "Progetti finanziati DIEM"
    if any(term in combined for term in ("avviso", "avvisi", "news", "bando", "seminario", "evento")):
        return "Avviso DIEM"
    if any(term in combined for term in ("segreteria", "ufficio", "contatti")):
        return "Servizi e contatti DIEM"
    if "didattica" in combined:
        return "Didattica DIEM"

    fallback_title = title or title_from_url(url)
    return f"Pagina DIEM - {fallback_title}" if fallback_title else "Pagina DIEM"


def fallback_context_header(text: str, url: str, metadata: dict | None = None) -> str:
    return classify_context_header(text, url, metadata)


# ---------------------------------------------------------------------------
# Semantic repair
# ---------------------------------------------------------------------------

def clean_header_for_semantic_repair(header: str) -> str:
    text = re.sub(r"^\s*context\s*:\s*", "", header or "", flags=re.IGNORECASE).strip()
    text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s*-\s*", " - ", text)
    return clean_text(text).strip(" -")


def split_header_theme(header: str) -> tuple[str, str]:
    if " - " in header:
        left, right = header.split(" - ", 1)
        return left.strip(), right.strip()
    return header.strip(), ""


def compact_header_theme(theme: str) -> str:
    theme = clean_text(theme or "").strip(" -")
    if theme.lower() in GENERIC_LOCAL_THEMES:
        return ""
    return theme


def context_header_with_topic(prefix: str, topic: str) -> str:
    topic = compact_header_theme(topic)
    if topic and topic.lower() != prefix.lower():
        return f"{prefix} - {topic}"
    return prefix


def header_contains_docente_profile(header: str) -> bool:
    lowered = clean_header_for_semantic_repair(header).lower()
    return (
        "docente/personale" in lowered
        or "profilo docente" in lowered
        or lowered.startswith("docente -")
        or lowered == "docente"
        or "curriculum docente" in lowered
        or "elenco docenti" in lowered
    )


def header_contains_scheda_insegnamento(header: str) -> bool:
    return "scheda insegnamento" in clean_header_for_semantic_repair(header).lower()


def header_contains_scheda_sua(header: str) -> bool:
    return "scheda sua" in clean_header_for_semantic_repair(header).lower()


def regolamento_header_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    left, right = split_header_theme(cleaned_header)
    if "scheda insegnamento" in lowered:
        right = compact_header_theme(right)
        if right and right.lower() != "scheda insegnamento":
            return f"insegnamento {right}"
        return "insegnamenti"
    if "scheda sua" in lowered:
        right = compact_header_theme(right)
        left = compact_header_theme(left)
        if right and "scheda" not in right.lower() and "corso di studio" not in right.lower():
            return right
        if left and "scheda" not in left.lower() and "corso di studio" not in left.lower():
            return left
        return "didattica"
    if "docente" in lowered:
        return "docenti e insegnamenti"
    return compact_header_theme(right)


def almalaurea_header_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    if "docent" in lowered:
        return "opinioni sui docenti"
    if "soddisfazione" in lowered:
        return "soddisfazione per il corso di studio"
    if "occup" in lowered:
        return "occupazione laureati"
    if "valutazione" in lowered or "opinioni" in lowered:
        return "opinioni laureati"
    if "laureat" in lowered or "statistic" in lowered:
        return "statistiche laureati"
    return ""


def repair_context_header_semantics(header: str, url: str) -> tuple[str, str] | None:
    """
    Correct only high-confidence document-type conflicts observed in the audit.
    Formatting-only cleanup stays outside this semantic repair step.
    """
    cleaned = clean_header_for_semantic_repair(header)
    source = (url or "").lower()

    if "__schede-sua" in source and header_contains_docente_profile(header):
        return "Scheda SUA corso di studio - docenti di riferimento", "schede_sua:docente_profile"

    if "__regolamenti-cds" in source:
        if header_contains_scheda_insegnamento(header):
            topic = regolamento_header_topic(cleaned)
            return (
                context_header_with_topic("Regolamento corso di studio", topic),
                "regolamenti_cds:scheda_insegnamento",
            )
        if header_contains_scheda_sua(header):
            topic = regolamento_header_topic(cleaned)
            return (
                context_header_with_topic("Regolamento corso di studio", topic),
                "regolamenti_cds:scheda_sua",
            )

    if "__almalaurea" in source:
        lowered_cleaned = cleaned.lower()
        is_wrong_type = (
            header_contains_docente_profile(header)
            or header_contains_scheda_sua(header)
            or "bando" in lowered_cleaned
            or "avviso" in lowered_cleaned
            or "progett" in lowered_cleaned
            or "ricerca" in lowered_cleaned
        )
        if is_wrong_type:
            topic = almalaurea_header_topic(cleaned)
            return (
                context_header_with_topic("Dati AlmaLaurea corso di studio", topic),
                "almalaurea:wrong_document_type",
            )

    return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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

    semantic_repair = repair_context_header_semantics(header, url)
    if semantic_repair:
        repaired_header, rule = semantic_repair
        logger.debug(
            "Context header semantic repair applied: "
            f"rule={rule}; source={url}; before={header}; after={repaired_header}"
        )
        header = repaired_header

    return header


def ensure_context_prefix(header: str) -> str:
    if not header.lower().startswith("context:"):
        return f"Context: {header}"
    return header