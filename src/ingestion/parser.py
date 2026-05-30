import re
from urllib.parse import parse_qs, urlparse, urldefrag, urljoin

import trafilatura
from bs4 import BeautifulSoup, Tag
from langchain_community.document_loaders import PDFPlumberLoader

from src.ingestion.crawler import is_pre_2020_url
from src.utils.logger import get_logger

logger = get_logger(__name__)

YEAR_CUTOFF = 2020
TEMPORAL_SCAN_CHARS = 2500
RAW_PDF_SCAN_CHARS = 2500
MIN_DOC_CHARS = 20
MAX_SYMBOL_RATIO = 0.45

METADATA_DATE_KEYS = (
    "date", "created", "creation_date", "creationdate", "moddate",
    "modified", "last_modified", "published", "publish_date", "updated",
)

# HTML <meta name/property> values that carry a publication or modification date
_DATE_META_NAMES = frozenset({
    "date", "article:published_time", "article:modified_time",
    "dc.date", "dcterms.date", "dcterms.modified",
    "pubdate", "publishdate",
})

# lang attribute prefixes considered non-Italian → drop
NON_ITALIAN_LANG_PREFIXES = ("en", "zh")

RAW_PDF_MARKERS = (
    "%pdf-",
    "/type /page",
    "/mediabox",
    "/cropbox",
    "/contents",
    "/resources",
    " endobj",
    " startxref",
)

STRUCTURED_PANEL_SELECTORS = (
    ".panel.panel-primary",
    ".panel",
    ".accordion-item",
    ".card",
)

STRUCTURED_PANEL_TITLE_SELECTORS = (
    ".panel-heading .panel-title",
    ".panel-heading",
    ".accordion-header",
    ".card-header",
)

STRUCTURED_PANEL_BODY_SELECTORS = (
    ".panel-body",
    ".accordion-body",
    ".card-body",
    ".panel-collapse",
    ".collapse",
)

STRUCTURED_NOISE_LABELS = {
    "",
    "home",
    "menu",
    "cerca",
    "filtro",
    "condividi",
    "precedente",
    "successiva",
    "pdf",
    "altri formati",
    "area utente",
    "tasse e servizi",
    "università degli studi di salerno",
}

STRUCTURED_SECTION_KEYWORDS = (
    "direttore",
    "giunta",
    "consiglio",
    "professore",
    "ricercatore",
    "personale",
    "assegnista",
    "dottorando",
    "docente",
    "aula",
    "laboratorio",
    "struttura",
)

STRUCTURED_UTILITY_SECTION_TITLES = {
    "contatti",
    "area utente",
    "area utente (esse3)",
    "disabilita e dsa",
    "disabilità e dsa",
}

STRUCTURED_UTILITY_ROWS = {
    "calendario occupazione",
    "planimetria",
    "contatti",
    "area utente",
    "area utente (esse3)",
    "disabilita e dsa",
    "disabilità e dsa",
}

CURRENT_NAV_LINES = {
    "condividi",
    "dipartimento",
    "home",
    "presentazione",
    "organi collegiali",
    "commissioni e delegati",
    "commissione paritetica docenti-studenti",
    "docenti e personale",
    "strutture",
    "dipartimento di eccellenza",
    "didattica",
    "ricerca",
    "precedente",
    "successiva",
    "-",
    "‹",
    "›",
    "×",
}


def extract_html_metadata(html: str) -> dict:
    """Extract title, language, and date from raw HTML.

    Must be called BEFORE html_extractor() — the DOM is destroyed after that.
    Returns a dict suitable for doc.metadata.update(). Keys produced:
      "title"    — text of <title> tag (if present)
      "language" — <html lang="..."> lowercased (if present)
      "date"     — content of the first matching <meta> or <time datetime> (if present)
    "date" maps to an existing key in METADATA_DATE_KEYS so temporal filter picks it up automatically.
    """
    meta: dict = {}
    try:
        # TODO (Bug Hunter): Consider using a more robust parser like lxml for BeautifulSoup to handle heavily malformed HTML better.
        soup = BeautifulSoup(html, "lxml")

        title_tag = soup.find("title")
        if title_tag:
            meta["title"] = title_tag.get_text(strip=True)

        html_tag = soup.find("html")
        if html_tag:
            lang = (html_tag.get("lang") or "").strip().lower()
            if lang:
                meta["language"] = lang

        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").lower().strip()
            if name in _DATE_META_NAMES:
                content = (tag.get("content") or "").strip()
                if content:
                    meta["date"] = content
                    break

        if "date" not in meta:
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag:
                meta["date"] = time_tag["datetime"].strip()

    except Exception:
        pass

    return meta


def clean_text(text: str) -> str:
    text = text.replace("�", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SITE_FOOTER_CONTROL_LINES = {
    "-",
    "‹",
    "›",
    "×",
    "precedente",
    "successiva",
}


def _normalize_boilerplate_line(line: str) -> str:
    return clean_text(line).lower().replace("à", "a")


def _is_site_footer_start(normalized_lines: list[str], index: int) -> bool:
    line = normalized_lines[index]
    if "universita degli studi di salerno" not in line:
        return False

    window = "\n".join(normalized_lines[index : index + 8])
    return any(
        marker in window
        for marker in (
            "via giovanni paolo ii",
            "84084 fisciano",
            "p.iva",
            "c.f.",
        )
    )


def remove_site_boilerplate(text: str) -> str:
    """Remove recurring UNISA footer/carousel text that can survive extraction."""
    lines = [clean_text(line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    normalized_lines = [_normalize_boilerplate_line(line) for line in lines]

    footer_start = None
    for index, normalized in enumerate(normalized_lines):
        if _is_site_footer_start(normalized_lines, index):
            footer_start = index
            break
        if index >= max(0, len(normalized_lines) - 8) and (
            "p.iva" in normalized or "c.f." in normalized
        ):
            footer_start = index
            break

    if footer_start is not None:
        lines = lines[:footer_start]

    while lines and _normalize_boilerplate_line(lines[-1]) in SITE_FOOTER_CONTROL_LINES:
        lines.pop()

    return clean_text("\n".join(lines))


def _bs4_extractor(html: str) -> str:
    # TODO (Code Refactorer): Repeated parsing with BeautifulSoup. If possible, parse once and pass the tree around.
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "iframe"]):
        tag.decompose()
    for selector in ["main", "article", "#content", ".content",
                     "#main", ".main-content", ".entry-content"]:
        content = soup.select_one(selector)
        if content:
            return remove_site_boilerplate(content.get_text("\n", strip=True))
    body = soup.find("body")
    source = body if body else soup
    return remove_site_boilerplate(source.get_text("\n", strip=True))


def html_extractor(html: str) -> str:
    # Primary extractor using trafilatura, which is robust to malformed HTML 
    # and often produces cleaner results than BeautifulSoup-based extraction.
    result = trafilatura.extract(
        html,
        include_tables=True,
        include_links=False,
        include_images=False,
        favor_precision=True,
    )
    if result:
        return remove_site_boilerplate(result)
    # Fallback to BeautifulSoup-based extractor if trafilatura fails (e.g. due to malformed HTML).
    return _bs4_extractor(html)


def _canonical_line(line: str) -> str:
    return clean_text(line).lower().replace("à", "a")


def _text_lines(text: str) -> list[str]:
    return [clean_text(line) for line in (text or "").splitlines() if clean_text(line)]


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        normalized = _canonical_line(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(line)
    return output


def _visible_text(tag: Tag) -> str:
    return clean_text(tag.get_text(" ", strip=True))


def _is_structured_noise_label(text: str) -> bool:
    lowered = clean_text(text).lower()
    if lowered in STRUCTURED_NOISE_LABELS:
        return True
    if len(lowered) < 3:
        return True
    if "@" in lowered:
        return True
    return False


def _is_probable_structured_heading(tag: Tag) -> bool:
    text = _visible_text(tag)
    lowered = text.lower()
    if _is_structured_noise_label(text) or len(text) > 90:
        return False

    classes = " ".join(tag.get("class", [])).lower()
    attrs = " ".join(
        str(tag.get(attr, ""))
        for attr in ("id", "role", "href", "data-toggle", "data-bs-toggle")
    ).lower()
    structural_hint = any(
        hint in f"{classes} {attrs}"
        for hint in ("accordion", "collapse", "panel", "card-header", "toggle")
    )

    if tag.name in {"h2", "h3", "h4", "h5", "h6"}:
        return any(keyword in lowered for keyword in STRUCTURED_SECTION_KEYWORDS)

    if tag.name in {"a", "button", "div", "span"}:
        if re.search(r"\baula\s+\w+", lowered, flags=re.IGNORECASE):
            return True
        return structural_hint and any(keyword in lowered for keyword in STRUCTURED_SECTION_KEYWORDS)

    return False


def _extract_table_rows(table: Tag) -> list[str]:
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [
            clean_text(cell.get_text(" ", strip=True))
            for cell in tr.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell and cell not in {"-", "×"}]
        if not cells:
            continue

        if len(cells) == 2:
            rows.append(f"{cells[0]}: {cells[1]}")
        else:
            rows.append(" | ".join(cells))

    return rows


def _extract_panel_title(panel: Tag) -> str:
    for selector in STRUCTURED_PANEL_TITLE_SELECTORS:
        title_tag = panel.select_one(selector)
        if title_tag:
            title = _visible_text(title_tag)
            if title and not _is_structured_noise_label(title):
                return title

    for heading in panel.find_all(["h2", "h3", "h4", "h5", "h6"], recursive=True):
        title = _visible_text(heading)
        if title and not _is_structured_noise_label(title):
            return title

    return ""


def _panel_body(panel: Tag) -> Tag:
    for selector in STRUCTURED_PANEL_BODY_SELECTORS:
        body = panel.select_one(selector)
        if body:
            return body
    return panel


def _extract_non_table_lines(container: Tag) -> list[str]:
    body_without_tables = BeautifulSoup(str(container), "html.parser")
    for tag in body_without_tables(["script", "style", "table", "button"]):
        tag.decompose()

    block_lines: list[str] = []
    for tag in body_without_tables.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li"], recursive=True):
        line = clean_text(tag.get_text(" ", strip=True))
        if line and not _is_structured_noise_label(line):
            block_lines.append(line)

    if block_lines:
        return block_lines

    return _text_lines(body_without_tables.get_text(" ", strip=True))


def _extract_panel_body_lines(container: Tag) -> list[str]:
    lines = _extract_non_table_lines(container)
    for table in container.find_all("table"):
        lines.extend(_extract_table_rows(table))
    return lines


def _extract_panel_sections(root: Tag) -> list[dict]:
    panels = root.select(", ".join(STRUCTURED_PANEL_SELECTORS))
    sections: list[dict] = []
    seen_panels: set[int] = set()
    seen_titles: set[str] = set()

    for panel in panels:
        panel_id = id(panel)
        if panel_id in seen_panels:
            continue
        seen_panels.add(panel_id)

        title = _extract_panel_title(panel)
        normalized_title = title.lower()
        if not title or normalized_title in seen_titles:
            continue

        body = _panel_body(panel)
        rows = _extract_panel_body_lines(body)
        # Strip rows that duplicate the title (happens when _panel_body falls back to the full panel)
        rows = [r for r in rows if r.lower() != normalized_title]
        if not rows:
            continue

        seen_titles.add(normalized_title)
        sections.append({"title": title, "rows": rows})

    return sections


def _structured_page_title(soup: BeautifulSoup) -> str:
    title = soup.find("h1")
    if title:
        return _visible_text(title)
    title = soup.find("title")
    if title:
        return _visible_text(title)
    return ""


def _structured_main_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "#content", ".content", "#main", ".main-content", "body"):
        root = soup.select_one(selector)
        if root:
            return root
    return soup


def _extract_structured_sections(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    root = _structured_main_root(soup)
    sections = _extract_panel_sections(root)
    if not sections:
        sections = []
        current: dict | None = None
        seen_tables: set[int] = set()
        seen_titles: set[str] = set()

        for tag in root.find_all(["h2", "h3", "h4", "h5", "h6", "a", "button", "div", "span", "table"]):
            if tag.name == "table":
                table_id = id(tag)
                if table_id in seen_tables:
                    continue
                seen_tables.add(table_id)

                rows = _extract_table_rows(tag)
                if rows:
                    if current is None:
                        current = {"title": "Tabella", "rows": []}
                        sections.append(current)
                    current["rows"].extend(rows)
                continue

            if not _is_probable_structured_heading(tag):
                continue

            title = _visible_text(tag)
            normalized_title = title.lower()
            if normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            current = {"title": title, "rows": []}
            sections.append(current)

    lines: list[str] = []
    title = _structured_page_title(soup)
    if title:
        lines.append(title)

    for section in sections:
        rows = section.get("rows") or []
        if not rows:
            continue
        lines.append("")
        lines.append(section.get("title", ""))
        lines.extend(rows)

    return clean_text("\n".join(lines)), sections


def _is_commissioni_detail_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.path.rstrip("/") == "/dipartimento/commissioni" and bool(parse_qs(parsed.query).get("dettaglio"))


def _is_structured_source(source: str) -> bool:
    parsed = urlparse(source)
    path = parsed.path.rstrip("/")
    netloc = parsed.netloc.lower()
    query = parse_qs(parsed.query)

    if netloc == "www.diem.unisa.it":
        if path in {
            "/dipartimento/personale",
            "/dipartimento/organi-collegiali",
            "/dipartimento/commissione-paritetica",
            "/dipartimento/commissioni",
        }:
            return True
        if path == "/dipartimento/strutture" and not query.get("id"):
            return True
        if path == "/dipartimento/commissioni" and query.get("dettaglio"):
            return True

    if netloc == "corsi.unisa.it":
        return path.endswith("/strutture-didattiche") or path.endswith("/contatti")

    if netloc == "docenti.unisa.it":
        return path.endswith("/didattica")
    return False


def _should_drop_structured_section(title: str) -> bool:
    return _canonical_line(title) in STRUCTURED_UTILITY_SECTION_TITLES


def _clean_structured_sections(sections: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    kept: list[dict] = []
    dropped_sections: list[str] = []
    dropped_rows: list[str] = []

    for section in sections:
        title = clean_text(str(section.get("title") or ""))
        if not title:
            continue
        if _should_drop_structured_section(title):
            dropped_sections.append(title)
            continue

        rows = []
        seen_rows: set[str] = set()
        for row in section.get("rows") or []:
            clean_row = clean_text(str(row))
            if not clean_row:
                continue
            normalized_row = _canonical_line(clean_row)
            if normalized_row in STRUCTURED_UTILITY_ROWS:
                dropped_rows.append(clean_row)
                continue
            if normalized_row in seen_rows:
                continue
            seen_rows.add(normalized_row)
            rows.append(clean_row)

        if rows:
            kept.append({"title": title, "rows": rows})

    return kept, dropped_sections, dropped_rows


def _extract_compiti_block(current_text: str) -> str:
    lines = _text_lines(current_text)
    start = None
    explicit_header = False
    for index, line in enumerate(lines):
        normalized = _canonical_line(line)
        if normalized == "compiti":
            start = index + 1
            explicit_header = True
            break
        if normalized.startswith("ha compiti "):
            start = index
            break

    if start is None:
        return ""

    collected: list[str] = []
    for line in lines[start:]:
        normalized = _canonical_line(line)
        if line.startswith("|"):
            break
        if normalized in CURRENT_NAV_LINES:
            break
        if "universita degli studi di salerno" in normalized:
            break
        if "p.iva" in normalized or "c.f." in normalized:
            break
        collected.append(line)

    collected = _dedupe_lines(collected)
    if not collected:
        return ""
    return clean_text("\n".join(["Compiti", *collected] if explicit_header else ["Compiti", *collected]))


def _extract_role_legend(current_text: str) -> str:
    lines = _text_lines(current_text)
    for index, line in enumerate(lines):
        if "(*) ruoli" in _canonical_line(line):
            collected = [line]
            for next_line in lines[index + 1 : index + 8]:
                normalized = _canonical_line(next_line)
                if normalized in CURRENT_NAV_LINES:
                    break
                if " = " in next_line or re.match(r"^[A-Z]{1,4}\s*=", next_line):
                    collected.append(next_line)
                    continue
                if collected:
                    break
            return clean_text("\n".join(_dedupe_lines(collected)))
    return ""


def _extract_contatti_description(current_text: str) -> str:
    useful: list[str] = []
    for line in _text_lines(current_text):
        normalized = _canonical_line(line)
        if normalized in CURRENT_NAV_LINES:
            continue
        if line.startswith("|"):
            break
        if "uffici carriere" in normalized or "uffici didattica" in normalized:
            useful.append(line)

    useful = _dedupe_lines(useful)
    if not useful:
        return ""
    return clean_text("\n".join(["Descrizione", *useful]))


def _is_low_value_current_text(current_text: str) -> bool:
    lines = _text_lines(current_text)
    if not lines:
        return True

    normalized_lines = [_canonical_line(line) for line in lines]
    if len(current_text) < 400 and any(
        "p.iva" in line or "universita degli studi di salerno" in line
        for line in normalized_lines
    ):
        return True

    informative = [
        line
        for line in normalized_lines
        if line not in CURRENT_NAV_LINES
        and "p.iva" not in line
        and "c.f." not in line
        and "universita degli studi di salerno" not in line
    ]
    return not informative


def _format_structured_document(title: str, sections: list[dict], section_title_map: dict[str, str] | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.append(title)

    section_title_map = section_title_map or {}
    for section in sections:
        section_title = section_title_map.get(section["title"], section["title"])
        lines.append("")
        lines.append(section_title)
        lines.extend(section["rows"])

    return clean_text("\n".join(lines))


def _structured_page_title_from_text(structured_text: str) -> str:
    lines = _text_lines(structured_text)
    return lines[0] if lines else ""


def _build_structured_final_text(source: str, current_text: str, structured_text: str, sections: list[dict]) -> tuple[str, str]:
    title = _structured_page_title_from_text(structured_text)
    clean_sections, dropped_sections, dropped_rows = _clean_structured_sections(sections)

    if not clean_sections:
        if _is_low_value_current_text(current_text):
            return "", "drop_low_value_page"
        return current_text, "fallback_current"

    section_title_map: dict[str, str] = {}
    for section in clean_sections:
        if _canonical_line(section["title"]) == "tabella":
            section_title_map[section["title"]] = "Componenti"

    structured_doc = _format_structured_document(title, clean_sections, section_title_map)
    extras: list[str] = []

    if _is_commissioni_detail_url(source):
        compiti = _extract_compiti_block(current_text)
        if compiti:
            extras.append(compiti)

    if "commissione-paritetica" in source:
        legend = _extract_role_legend(current_text)
        if legend:
            extras.append(legend)

    if urlparse(source).netloc.lower() == "corsi.unisa.it" and urlparse(source).path.rstrip("/").endswith("/contatti"):
        description = _extract_contatti_description(current_text)
        if description:
            extras.append(description)

    if extras:
        lines = [title] if title else []
        lines.extend(["", *extras])
        structured_body = structured_doc.replace(title, "", 1).strip() if title else structured_doc
        if structured_body:
            lines.extend(["", structured_body])
        return clean_text("\n".join(lines)), "structured_enriched_single_doc"

    return structured_doc, "structured_only_single_doc"


def html_extractor_for_source(html: str, source: str = "") -> str:
    """Extract HTML text, using structured parsing only for known critical pages."""
    if not source or not _is_structured_source(source):
        return html_extractor(html)

    current_text = html_extractor(html)
    structured_text, sections = _extract_structured_sections(html)
    final_text, strategy = _build_structured_final_text(source, current_text, structured_text, sections)

    if strategy != "fallback_current":
        logger.debug(
            "Structured HTML extraction: strategy=%s; source=%s; current_chars=%d; "
            "structured_chars=%d; final_chars=%d; sections=%d",
            strategy,
            source,
            len(current_text),
            len(structured_text),
            len(final_text),
            len(sections),
        )

    return final_text


def extract_years_from_text(text: str) -> list[int]:
    if not text:
        return []
    return [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]


def extract_years_from_metadata(metadata: dict) -> list[int]:
    years = []
    for key, value in metadata.items():
        key_name = key.lower().strip("/")
        if value is None or key_name not in METADATA_DATE_KEYS:
            continue
        years.extend(extract_years_from_text(str(value)))
    return years


def should_keep_document(doc) -> tuple[bool, str]:
    source = doc.metadata.get("source", "")
    scan_text = doc.page_content[:TEMPORAL_SCAN_CHARS]
    lowered = f"{source}\n{scan_text}".lower()
    is_pdf = ".pdf" in urlparse(source).path.lower()
    is_time_sensitive = any(
        marker in lowered
        for marker in ("avvisi", "avviso", "news", "bando", "bandi", "seminari", "eventi")
    )

    metadata_or_url_years = (
        extract_years_from_metadata(doc.metadata)
        + extract_years_from_text(source)
    )
    if metadata_or_url_years:
        newest = max(metadata_or_url_years)
        if newest < YEAR_CUTOFF:
            return False, f"old year {newest}"
        return True, f"year {newest}"

    text_years = extract_years_from_text(scan_text)
    if (is_pdf or is_time_sensitive) and text_years:
        newest = max(text_years)
        if newest < YEAR_CUTOFF:
            return False, f"old text year {newest}"
        return True, f"text year {newest}"

    return True, "no explicit date"


def filter_recent_documents(docs: list) -> list:
    kept = []
    dropped = 0
    for doc in docs:
        keep, reason = should_keep_document(doc)
        if keep:
            doc.metadata["temporal_filter"] = reason
            kept.append(doc)
        else:
            dropped += 1
            source = doc.metadata.get("source", "unknown source")
            logger.debug(f"  SKIP old document ({reason}): {source}")

    logger.info(
        f"  -> Temporal filter kept {len(kept)}/{len(docs)} documents "
        f"(cutoff year: {YEAR_CUTOFF}, dropped: {dropped})"
    )
    return kept


def is_raw_pdf_artifact(text: str) -> bool:
    """Detect PDF object streams accidentally indexed as plain text."""
    if not text:
        return False

    scan = text[:RAW_PDF_SCAN_CHARS].lower()
    if scan.lstrip().startswith("%pdf-"):
        return True

    marker_hits = sum(marker in scan for marker in RAW_PDF_MARKERS)
    object_hits = len(re.findall(r"\b\d+\s+\d+\s+obj\b", scan))
    if marker_hits >= 3 and object_hits >= 1:
        return True
    if marker_hits >= 4:
        return True
    return False


def is_low_text_quality_document(doc) -> tuple[bool, str]:
    """Return whether a document should be dropped before enrichment/indexing."""
    if is_raw_pdf_artifact(doc.page_content):
        return True, "raw PDF object stream"
    text = doc.page_content.strip()
    if len(text) < MIN_DOC_CHARS:
        return True, f"too short ({len(text)} chars)"
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    ratio = non_alnum / len(text)
    if ratio > MAX_SYMBOL_RATIO:
        return True, f"high symbol ratio ({ratio:.2f})"
    return False, ""


def filter_low_quality_documents(docs: list) -> list:
    """Drop documents whose extracted text is not useful natural language."""
    kept = []
    dropped_reasons: dict[str, int] = {}

    for doc in docs:
        drop, reason = is_low_text_quality_document(doc)
        if drop:
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            source = doc.metadata.get("source", "unknown source")
            logger.debug(f"  SKIP low-quality document ({reason}): {source}")
            continue
        kept.append(doc)

    dropped = len(docs) - len(kept)
    if dropped:
        details = ", ".join(f"{reason}: {count}" for reason, count in sorted(dropped_reasons.items()))
        logger.info(f"  -> Quality filter kept {len(kept)}/{len(docs)} documents (dropped: {dropped}; {details})")
    else:
        logger.info(f"  -> Quality filter kept {len(kept)}/{len(docs)} documents (dropped: 0)")
    return kept


def looks_like_pdf_url(href: str) -> bool:
    """Return whether an href points to a PDF by inspecting its URL path."""
    clean_href = href.split("#")[0]
    return urlparse(clean_href).path.lower().endswith(".pdf")


def resolve_pdf_url(page_url: str, href: str) -> str:
    """Resolve PDF hrefs, handling UNISA root-relative upload paths."""
    href = href.strip()
    parsed_page = urlparse(page_url)

    if href.startswith("uploads/"):
        return f"{parsed_page.scheme}://{parsed_page.netloc}/{href}"

    return urljoin(page_url, href)


def load_pdfs_from_links(raw_docs: list, seen_urls: set | None = None) -> list:
    """Load PDF documents linked from already-crawled HTML pages.

    Must be called while page_content still contains raw HTML.
    """
    seen_urls = seen_urls if seen_urls is not None else set()
    pdf_docs = []
    logger.info(f"Scanning {len(raw_docs)} documents for PDF links...")

    for doc in raw_docs:
        page_url = doc.metadata.get("source", "")
        try:
            # TODO (Bug Hunter): Consider using `lxml` here as well for better performance and error handling.
            soup = BeautifulSoup(doc.page_content, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not looks_like_pdf_url(href):
                    continue

                pdf_url, _ = urldefrag(resolve_pdf_url(page_url, href))
                logger.debug(f"  PDF candidate: page={page_url} href={href} resolved={pdf_url}")

                if pdf_url in seen_urls:
                    logger.debug(f"  SKIP duplicate PDF: {pdf_url}")
                    continue
                seen_urls.add(pdf_url)

                if is_pre_2020_url(pdf_url):
                    logger.debug(f"  SKIP pre-2020 PDF: {pdf_url}")
                    continue

                try:
                    # TODO (Software Architect): Replaced PyPDFLoader with PDFPlumberLoader for better table and layout extraction.
                    docs = PDFPlumberLoader(pdf_url).load()
                    if docs:
                        # Merge all pages into one Document so the parent splitter
                        # can create cross-page chunks instead of being capped at page boundaries.
                        merged_text = "\n\n".join(
                            d.page_content for d in docs if d.page_content.strip()
                        )
                        merged_doc = Document(
                            page_content=merged_text,
                            metadata={
                                **docs[0].metadata,
                                "source": pdf_url,
                                "source_page": page_url,
                                "total_pages": len(docs),
                            },
                        )
                        pdf_docs.append(merged_doc)
                    logger.info(f"  PDF loaded: {pdf_url} ({len(docs)} pages → 1 merged doc)")
                except Exception as e:
                    logger.warning(f"  WARNING: skipped PDF {pdf_url}: {e}")
        except Exception as e:
            logger.warning(f"  WARNING: could not inspect PDF links in {page_url}: {e}")

    return pdf_docs
