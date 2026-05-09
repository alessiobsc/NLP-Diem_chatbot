import re
from urllib.parse import urlparse, urldefrag, urljoin

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader

from src.ingestion.crawler import is_pre_2020_url

YEAR_CUTOFF = 2020
TEMPORAL_SCAN_CHARS = 2500

METADATA_DATE_KEYS = (
    "date", "created", "creation_date", "creationdate", "moddate",
    "modified", "last_modified", "published", "publish_date", "updated",
)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_extractor(html: str) -> str:
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
            print(f"  SKIP old document ({reason}): {source}")

    print(
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

    for doc in raw_docs:
        page_url = doc.metadata.get("source", "")
        try:
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
                    print(f"  SKIP pre-2020 PDF: {pdf_url}")
                    continue

                try:
                    docs = PyPDFLoader(pdf_url).load()
                    for pdf_doc in docs:
                        pdf_doc.metadata.setdefault("source", pdf_url)
                        pdf_doc.metadata.setdefault("source_page", page_url)
                    pdf_docs.extend(docs)
                    print(f"  PDF loaded: {pdf_url} ({len(docs)} pages)")
                except Exception as e:
                    print(f"  WARNING: skipped PDF {pdf_url}: {e}")
        except Exception as e:
            print(f"  WARNING: could not inspect PDF links in {page_url}: {e}")

    return pdf_docs
