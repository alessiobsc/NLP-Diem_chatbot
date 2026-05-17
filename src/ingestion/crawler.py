import json
import re
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict, Set, Tuple, List, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================
HTML_LINK_REGEX = r"""<a\s+(?:[^>]*?\s+)?href=["']([^"']*)["']"""

# Paths and extensions to avoid crawling
EXCLUDE_DIRS: Tuple[str, ...] = (
    "/rescue/css/", "/rescue/js/", "/rescue/assets/",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot",
    "/idp/", "/password-recovery", "/login",
    "/print()", "/history.go(", "@",
)

# Substrings to skip when filtering already crawled documents
SKIP_DOCUMENT_SUBSTRINGS: Tuple[str, ...] = (
    "?sitemap",
    "/print()",
    "/history.go(",
    ".pdf",
    "/en/",
    "/zh/",
)

OFFERTA_FORMATIVA_PATH = "/didattica/offerta-formativa"
SITEMAP_QUERY = "?sitemap"
DIEM_PERSONALE_URL = "https://www.diem.unisa.it/dipartimento/personale"

REQUEST_TIMEOUT = 15
CONCURRENT_MAX_WORKERS = 10

# ==============================================================================
# MONKEY PATCHING & GLOBALS CONFIGURATION
# ==============================================================================

def _configure_global_session() -> None:
    """
    Configures the global behavior for 'requests' and 'urllib3'.
    - Disables insecure request warnings.
    - Sets a global default to avoid SSL verification (due to missing UNISA certs on Windows).
    - Adds a robust retry strategy for all requests using the requests.Session.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context

    _orig_session_init = Session.__init__

    def _custom_session_init(self: Session) -> None:
        _orig_session_init(self)
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.mount("http://", adapter)
        self.mount("https://", adapter)

    Session.__init__ = _custom_session_init

    _orig_request = Session.request

    def _no_ssl_verify(self: Session, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("verify", False)
        return _orig_request(self, method, url, **kwargs)

    Session.request = _no_ssl_verify

_configure_global_session()

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def is_pre_2020_url(url: str) -> bool:
    """Check if the URL belongs to an academic year prior to 2020."""
    normalized_url = url.replace("_", "/")
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20[01]\d)\b", normalized_url)]
    return any(year < 2020 for year in years)


def get_section_base(url: str) -> str:
    """
    Extract the first-segment root of a URL path.

    Examples:
        https://docenti.unisa.it/003145/home  ->  https://docenti.unisa.it/003145/
        https://corsi.unisa.it/ing-inf/home   ->  https://corsi.unisa.it/ing-inf/
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    base_path = f"/{parts[0]}/" if parts and parts[0] else "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def build_html_sitemap_url(base_url: str) -> str:
    """Construct the standard sitemap URL from a given base URL."""
    return f"{base_url.rstrip('/')}/{SITEMAP_QUERY}"

# ==============================================================================
# CRAWLING & EXTRACTION
# ==============================================================================

def crawl(start_url: str, base_url: str, max_depth: int = 2) -> Iterable:
    """
    Crawl a target URL recursively up to a given depth.
    Yields documents lazily to preserve memory.
    """
    logger.debug(f"Crawling {start_url} with max depth {max_depth}")
    try:
        loader = RecursiveUrlLoader(
            start_url,
            base_url=base_url,
            max_depth=max_depth,
            prevent_outside=True,
            timeout=REQUEST_TIMEOUT,
            check_response_status=True,
            exclude_dirs=EXCLUDE_DIRS,
            link_regex=HTML_LINK_REGEX,
        )
        return loader.lazy_load()
    except Exception as e:
        logger.warning(f"  FAILED {start_url}: {e}")
        return iter([])


def extract_html_sitemap_urls(sitemap_url: str, base_url: str) -> List[str]:
    """Extract deterministic section URLs from UNISA HTML sitemap pages."""
    logger.info(f"Fetching HTML sitemap: {sitemap_url}")
    try:
        response = requests.get(sitemap_url, timeout=REQUEST_TIMEOUT, verify=False)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"  WARNING: could not fetch HTML sitemap {sitemap_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    base_netloc = urlparse(base_url).netloc
    valid_urls: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue

        absolute_url, _ = urldefrag(urljoin(sitemap_url, href))
        
        if not _is_valid_sitemap_url(absolute_url, base_netloc):
            continue
            
        valid_urls.add(absolute_url)

    seeds = sorted(valid_urls)
    logger.info(f"  -> {len(seeds)} sitemap seeds extracted from {sitemap_url}")
    return seeds


def _is_valid_sitemap_url(url: str, expected_netloc: str) -> bool:
    """Helper to validate if a URL should be included from the sitemap."""
    parsed = urlparse(url)
    if parsed.netloc != expected_netloc:
        return False
    if SITEMAP_QUERY in url:
        return False
    if any(excluded in url for excluded in EXCLUDE_DIRS):
        return False
    if is_pre_2020_url(url):
        return False
    return True


def crawl_html_sitemap(
    base_url: str,
    max_depth: int = 1,
    crawl_base_url: Optional[str] = None,
    fallback_depth: Optional[int] = None,
) -> Iterable:
    """Crawl section seeds from a UNISA HTML sitemap using shallow recursion."""
    crawl_base_url = crawl_base_url or base_url
    fallback_depth = fallback_depth if fallback_depth is not None else max_depth
    sitemap_url = build_html_sitemap_url(base_url)
    
    seeds = extract_html_sitemap_urls(sitemap_url, base_url)
    
    if not seeds:
        logger.warning(f"  WARNING: no sitemap seeds found for {base_url}; falling back to direct crawl")
        yield from crawl(base_url, base_url=crawl_base_url, max_depth=fallback_depth)
        return

    for i, seed in enumerate(seeds, 1):
        seed_docs = crawl(seed, base_url=crawl_base_url, max_depth=max_depth)
        count = 0
        for doc in seed_docs:
            count += 1
            yield doc
        logger.debug(f"    [{i:02d}/{len(seeds)}] {seed} ({count} pages)")


def extract_corsi_urls(raw_docs: Iterable) -> List[str]:
    """Parse documents related to educational offerings and extract course URLs."""
    course_urls: Set[str] = set()
    
    # Filter documents relevant to course offerings
    source_docs = (
        doc for doc in raw_docs
        if OFFERTA_FORMATIVA_PATH in doc.metadata.get("source", "")
        and "?anno=" not in doc.metadata.get("source", "")
    )

    for doc in source_docs:
        try:
            soup = BeautifulSoup(doc.page_content, "lxml")
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if href.startswith("http") and "corsi.unisa.it" in href:
                    parsed = urlparse(href)
                    first_segment = next(iter(parsed.path.strip("/").split("/")), None)
                    if first_segment:
                        course_urls.add(f"{parsed.scheme}://{parsed.netloc}/{first_segment}")
        except Exception as e:
            logger.debug(f"Failed to parse course URL from document {doc.metadata.get('source')}: {e}")

    logger.info(f"Extracted {len(course_urls)} course URLs")
    return sorted(course_urls)


def _fetch_personale_page(url: str) -> str:
    """Fetch the main DIEM 'personale' HTML page."""
    logger.info(f"Fetching faculty list from {url}")
    response = requests.get(url, timeout=REQUEST_TIMEOUT, verify=False)
    response.raise_for_status()
    return response.text


def _parse_rubrica_links(html: str) -> Dict[str, str]:
    """
    Parse the 'personale' page HTML to extract 'rubrica' links.
    Returns a mapping of {matricola: rubrica_url}.
    """
    soup = BeautifulSoup(html, "lxml")
    rubrica_links: Dict[str, str] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href or not href.startswith("http"):
            continue

        match = re.search(r"matricola=(\d+)", href)
        if match:
            rubrica_links[match.group(1)] = href

    return rubrica_links


def _fetch_and_validate_faculty_profile(matricola: str, url: str) -> Tuple[str, bool, Exception | str]:
    """Helper for concurrently validating a faculty profile page."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        has_profile = any(
            "docenti.unisa.it" in (anchor.get("href") or "")
            for anchor in soup.find_all("a", href=True)
        )
        return matricola, has_profile, "OK"
    except Exception as exc:
        return matricola, False, exc


def _validate_faculty_urls_concurrently(rubrica_links: Dict[str, str]) -> List[str]:
    """
    Given a dict of matricola -> rubrica_url, uses a thread pool to validate
    which ones actually have a docenti.unisa.it profile.
    Returns a sorted list of validated profile URLs.
    """
    matricole = sorted(rubrica_links)
    logger.info(f"  -> {len(matricole)} matricole found on personale page, validating via rubrica...")

    validated_urls: List[str] = []

    with ThreadPoolExecutor(max_workers=CONCURRENT_MAX_WORKERS) as executor:
        future_to_matricola = {
            executor.submit(_fetch_and_validate_faculty_profile, mid, rubrica_links[mid]): mid
            for mid in matricole
        }

        count = 0
        total_matricole = len(matricole)
        
        for future in as_completed(future_to_matricola):
            count += 1
            matricola = future_to_matricola[future]
            
            try:
                res_mid, has_profile, status = future.result()
                
                if isinstance(status, Exception):
                    logger.warning(f"    [{count:02d}/{total_matricole}] {res_mid}: ERROR {status} — skipping")
                    continue
                    
                log_status = "OK" if has_profile else "SKIP (no profile)"
                logger.debug(f"    [{count:02d}/{total_matricole}] {res_mid}: {log_status}")
                
                if has_profile:
                    validated_urls.append(f"https://docenti.unisa.it/{res_mid}/home")
            except Exception as e:
                logger.warning(f"    [{count:02d}/{total_matricole}] {matricola}: ERROR {e} — skipping")

    logger.info(f"  -> {len(validated_urls)}/{total_matricole} faculty URLs validated")
    return sorted(validated_urls)


def extract_diem_faculty_urls() -> List[str]:
    """
    Scrape DIEM 'personale' page to validate docenti.unisa.it URLs.
    Uses concurrency to speed up the validation process.
    """
    try:
        html = _fetch_personale_page(DIEM_PERSONALE_URL)
        rubrica_links = _parse_rubrica_links(html)
        return _validate_faculty_urls_concurrently(rubrica_links)
    except Exception as e:
        logger.error(f"  WARNING: could not fetch or process faculty list from personale page: {e}")
        return []


# ==============================================================================
# POST-PROCESSING & I/O
# ==============================================================================

def filter_docs(docs: Iterable) -> Iterable:
    """Filter out documents whose source URL matches known exclude patterns."""
    dropped_count = 0
    for doc in docs:
        source_url = doc.metadata.get("source", "")
        
        contains_skip_substring = any(sub in source_url for sub in SKIP_DOCUMENT_SUBSTRINGS)
        is_too_old = is_pre_2020_url(source_url)
        
        if not contains_skip_substring and not is_too_old:
            yield doc
        else:
            dropped_count += 1
            
    logger.debug(f"Filtered {dropped_count} documents")


def save_crawled_pdfs_to_json(pdf_docs: Iterable, filename: str) -> None:
    """Group PDF pages by source URL and save the summary to a JSON file."""
    pdfs_summary: Dict[str, Dict] = {}
    total_pages_count = 0
    
    for doc in pdf_docs:
        total_pages_count += 1
        source_url = doc.metadata.get("source", "")
        
        if source_url not in pdfs_summary:
            pdfs_summary[source_url] = {
                "url": source_url,
                "source_page": doc.metadata.get("source_page", ""),
                "pages": 0,
            }
        pdfs_summary[source_url]["pages"] += 1

    entries = sorted(pdfs_summary.values(), key=lambda x: x["url"])
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
        
    logger.info(f"  -> Saved {len(entries)} PDF sources ({total_pages_count} pages total) to {filename}")


def save_crawled_urls_to_json(docs: Iterable, filename: str) -> None:
    """
    Extract URL and <title> from documents and save them to a JSON file.
    Falls back to parsing raw HTML with BeautifulSoup if title is missing.
    """
    entries = []
    for doc in docs:
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        
        if not title:
            try:
                soup = BeautifulSoup(doc.page_content, "lxml")
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)
            except Exception:
                pass
                
        entries.append({"url": url, "title": title})

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    logger.info(f"  -> Saved {len(entries)} URLs to {filename}")