import re
from urllib.parse import urlparse, urldefrag, urljoin

import trafilatura
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PDFPlumberLoader

from src.ingestion.crawler import is_pre_2020_url
from src.logger import get_logger

logger = get_logger(__name__)

YEAR_CUTOFF = 2020
TEMPORAL_SCAN_CHARS = 2500

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
        soup = BeautifulSoup(html, "html.parser")

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


def _bs4_extractor(html: str) -> str:
    # TODO (Code Refactorer): Repeated parsing with BeautifulSoup. If possible, parse once and pass the tree around.
    soup = BeautifulSoup(html, "html.parser")
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
            soup = BeautifulSoup(doc.page_content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                clean_href = href.split("?")[0].split("#")[0]
                if not clean_href.lower().endswith(".pdf"):
                    continue

                pdf_url, _ = urldefrag(urljoin(page_url, href))
                if pdf_url in seen_urls:
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