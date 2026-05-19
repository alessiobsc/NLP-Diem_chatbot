import re
from urllib.parse import parse_qs, urlparse, urldefrag, urljoin

import trafilatura
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PDFPlumberLoader

from src.ingestion.crawler import is_pre_2020_url
from src.utils.logger import get_logger

logger = get_logger(__name__)

YEAR_CUTOFF = 2020
TEMPORAL_SCAN_CHARS = 2500
RAW_PDF_SCAN_CHARS = 2500

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

STRUCTURED_UTILITY_SECTION_TITLES = {
    "contatti",
    "area utente (esse3)",
    "disabilità e dsa",
    "disabilita e dsa",
}

COMMISSIONI_DETAIL_NOISE_LINES = {
    "",
    "-",
    "condividi",
    "dipartimento",
    "commissioni e delegati",
    "presentazione",
    "organi collegiali",
    "commissione paritetica docenti-studenti",
    "docenti e personale",
    "strutture",
    "dipartimento di eccellenza",
    "didattica",
    "ricerca",
    "precedente",
    "successiva",
    "‹",
    "›",
    "×",
}

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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalized_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _is_structured_utility_title(title: str) -> bool:
    return _normalized_line(title) in STRUCTURED_UTILITY_SECTION_TITLES


def _visible_text(tag) -> str:
    return clean_text(tag.get_text(" ", strip=True))


def _main_html_root(soup: BeautifulSoup):
    for selector in ("main", "article", "#unisa-content", "#content", ".content",
                     "#main", ".main-content", ".entry-content", "body"):
        root = soup.select_one(selector)
        if root:
            return root
    return soup


def _page_title_from_soup(soup: BeautifulSoup) -> str:
    title = soup.find("h1")
    if title:
        return _visible_text(title)
    title = soup.find("title")
    if title:
        return _visible_text(title)
    return ""


def _extract_table_rows(table) -> list[str]:
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


def _extract_panel_title(panel) -> str:
    for selector in STRUCTURED_PANEL_TITLE_SELECTORS:
        title_tag = panel.select_one(selector)
        if title_tag:
            title = _visible_text(title_tag)
            if title:
                return title

    for heading in panel.find_all(["h2", "h3", "h4", "h5", "h6"], recursive=True):
        title = _visible_text(heading)
        if title:
            return title

    return ""


def _panel_body(panel):
    for selector in STRUCTURED_PANEL_BODY_SELECTORS:
        body = panel.select_one(selector)
        if body:
            return body
    return panel


def _extract_non_table_lines(container) -> list[str]:
    body_without_tables = BeautifulSoup(str(container), "html.parser")
    for tag in body_without_tables(["script", "style", "table", "button"]):
        tag.decompose()

    block_lines: list[str] = []
    for tag in body_without_tables.find_all(["h2", "h3", "h4", "h5", "h6", "p", "li"], recursive=True):
        line = clean_text(tag.get_text(" ", strip=True))
        if line:
            block_lines.append(line)

    if block_lines:
        return block_lines

    text = clean_text(body_without_tables.get_text(" ", strip=True))
    return [text] if text else []


def _extract_panel_body_lines(container) -> list[str]:
    lines = _extract_non_table_lines(container)
    for table in container.find_all("table"):
        lines.extend(_extract_table_rows(table))
    return lines


def _extract_structured_panel_sections(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    root = _main_html_root(soup)
    panel_selector = ", ".join(STRUCTURED_PANEL_SELECTORS)
    sections: list[dict] = []
    seen_titles: set[str] = set()

    for panel in root.select(panel_selector):
        title = _extract_panel_title(panel)
        normalized_title = _normalized_line(title)
        if not title or not normalized_title or normalized_title in seen_titles:
            continue
        if _is_structured_utility_title(title):
            continue

        body = _panel_body(panel)
        rows = _extract_panel_body_lines(body)
        rows = [row for row in rows if row and _normalized_line(row) != normalized_title]
        if not rows:
            continue

        seen_titles.add(normalized_title)
        sections.append({"title": title, "rows": rows})

    if not sections:
        return "", []

    lines: list[str] = []
    title = _page_title_from_soup(soup)
    if title:
        lines.append(title)

    for section in sections:
        lines.append("")
        lines.append(section["title"])
        lines.extend(section["rows"])

    return clean_text("\n".join(lines)), sections


def _extract_structured_table_sections(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    root = _main_html_root(soup)
    rows: list[str] = []
    for table in root.find_all("table"):
        rows.extend(_extract_table_rows(table))

    rows = [row for row in rows if row]
    if not rows:
        return "", []

    section = {"title": "Tabella", "rows": rows}
    lines: list[str] = []
    title = _page_title_from_soup(soup)
    if title:
        lines.append(title)
        lines.append("")
    lines.append(section["title"])
    lines.extend(rows)

    return clean_text("\n".join(lines)), [section]


def _is_targeted_structured_source(source: str) -> bool:
    parsed = urlparse(source or "")
    path = parsed.path.rstrip("/")
    query = parse_qs(parsed.query)

    if path.endswith("/strutture-didattiche"):
        return True
    if path == "/dipartimento/personale":
        return True
    if path == "/dipartimento/organi-collegiali":
        return True
    if path == "/dipartimento/commissione-paritetica":
        return True
    if path == "/dipartimento/commissioni" and "dettaglio" in query:
        return True
    return False


def _is_commissioni_detail_source(source: str) -> bool:
    parsed = urlparse(source or "")
    return parsed.path.rstrip("/") == "/dipartimento/commissioni" and "dettaglio" in parse_qs(parsed.query)


def _is_commissioni_noise_line(line: str) -> bool:
    normalized = _normalized_line(line)
    if normalized in COMMISSIONI_DETAIL_NOISE_LINES:
        return True
    if normalized.startswith("università degli studi di salerno"):
        return True
    if normalized.startswith("via giovanni paolo ii"):
        return True
    if "p.iva" in normalized or "c.f." in normalized:
        return True
    return False


def _clean_commissioni_detail_current_text(text: str) -> str:
    lines = [
        line.strip()
        for line in (text or "").splitlines()
        if line.strip() and not _is_commissioni_noise_line(line)
    ]
    return clean_text("\n".join(lines))


def _has_useful_commissioni_description(text: str) -> bool:
    lowered = (text or "").lower()
    if "compiti" in lowered or "ha compiti" in lowered:
        return True
    if "commissione" in lowered and len(text) >= 250:
        return True
    return False


def _merge_current_and_structured(current_text: str, structured_text: str) -> str:
    current_lines = {_normalized_line(line) for line in current_text.splitlines() if _normalized_line(line)}
    additions = [
        line.strip()
        for line in structured_text.splitlines()
        if line.strip() and _normalized_line(line) not in current_lines
    ]

    if not additions:
        return current_text

    return clean_text(
        f"{current_text.rstrip()}\n\nSezioni strutturate estratte:\n" + "\n".join(additions)
    )


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
            return clean_text(content.get_text("\n", strip=True))
    body = soup.find("body")
    source = body if body else soup
    return clean_text(source.get_text("\n", strip=True))


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
        return clean_text(result)
    # Fallback to BeautifulSoup-based extractor if trafilatura fails (e.g. due to malformed HTML).
    return _bs4_extractor(html)


def html_extractor_for_source(html: str, source: str = "") -> str:
    """
    Extract HTML text and selectively add structured accordion/table content.

    The structured extraction is intentionally enabled only for known DIEM/UNISA
    page shapes where trafilatura tends to lose the relation between a blue
    accordion title and its table rows (classrooms, staff lists, committees).
    """
    current_text = html_extractor(html)
    if not _is_targeted_structured_source(source):
        return current_text

    structured_text, sections = _extract_structured_panel_sections(html)
    if not structured_text or not sections:
        structured_text, sections = _extract_structured_table_sections(html)

    if not structured_text or not sections:
        logger.debug(f"Structured HTML extraction skipped (no sections): {source}")
        return current_text

    structured_rows = sum(len(section.get("rows", [])) for section in sections)
    if structured_rows < 1:
        logger.debug(f"Structured HTML extraction skipped (no rows): {source}")
        return current_text

    merge_base = current_text
    mode = "merge"
    if _is_commissioni_detail_source(source):
        cleaned_current = _clean_commissioni_detail_current_text(current_text)
        if _has_useful_commissioni_description(cleaned_current):
            merge_base = cleaned_current
            mode = "merge_clean_current"
        else:
            logger.debug(
                "Structured HTML extraction replaced noisy commissioni detail page: "
                f"source={source}; current_chars={len(current_text)}; "
                f"cleaned_current_chars={len(cleaned_current)}; structured_chars={len(structured_text)}"
            )
            return structured_text

    merged_text = _merge_current_and_structured(merge_base, structured_text)
    logger.debug(
        "Structured HTML extraction applied: "
        f"source={source}; sections={len(sections)}; rows={structured_rows}; "
        f"current_chars={len(current_text)}; structured_chars={len(structured_text)}; "
        f"merged_chars={len(merged_text)}; mode={mode}"
    )
    return merged_text


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
                    for pdf_doc in docs:
                        pdf_doc.metadata.setdefault("source", pdf_url)
                        pdf_doc.metadata.setdefault("source_page", page_url)
                    pdf_docs.extend(docs)
                    logger.info(f"  PDF loaded: {pdf_url} ({len(docs)} pages)")
                except Exception as e:
                    logger.warning(f"  WARNING: skipped PDF {pdf_url}: {e}")
        except Exception as e:
            logger.warning(f"  WARNING: could not inspect PDF links in {page_url}: {e}")

    return pdf_docs
