import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict, Set, Tuple, List, Optional
from urllib.parse import parse_qs, urlencode, urldefrag, urljoin, urlparse, urlunparse

import requests
import urllib3
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import RecursiveUrlLoader
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.ingestion.crawl_state import CrawlStateManager
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
    "unisa-rescue-page",
    ".pdf",
    "/en/",
    "/zh/",
    "/valutazione-della-didattica",
    "/pubblicazioni?anno=0",  # "all years" publication aggregation — keep only year-specific URLs
)

OFFERTA_FORMATIVA_PATH = "/didattica/offerta-formativa"
SITEMAP_QUERY = "?sitemap"
DIEM_PERSONALE_URL = "https://www.diem.unisa.it/dipartimento/personale"

REQUEST_TIMEOUT = 15
CONCURRENT_MAX_WORKERS = 10

# Politeness / Rate Limiting
REQUESTS_PER_SECOND_LIMIT = 5

# ==============================================================================
# HTTP SESSION FACTORY & RATE LIMITING
# ==============================================================================

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class RateLimiter:
    """A thread-safe rate limiter implementing a basic token bucket / delay mechanism."""
    def __init__(self, calls_per_second: float):
        self.delay = 1.0 / calls_per_second
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()


# Global rate limiter instance
_global_rate_limiter = RateLimiter(REQUESTS_PER_SECOND_LIMIT)


class RateLimitedSession(Session):
    """A requests.Session subclass that respects a rate limit before each request."""
    def __init__(self, rate_limiter: RateLimiter):
        super().__init__()
        self.rate_limiter = rate_limiter

    def request(self, method, url, *args, **kwargs):
        self.rate_limiter.wait()
        return super().request(method, url, *args, **kwargs)


def create_resilient_session() -> Session:
    """
    Creates a RateLimitedSession with a retry strategy and SSL verification disabled.
    This session is designed for interacting with UNISA's infrastructure politely.
    """
    session = RateLimitedSession(_global_rate_limiter)
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.verify = False  # Disable SSL verification for this session
    return session


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

class CustomRecursiveUrlLoader(RecursiveUrlLoader):
    """
    A custom RecursiveUrlLoader that uses a resilient, pre-configured session
    and a state manager for incremental crawling.
    """

    def __init__(self, *args, session: Optional[Session] = None, state_manager: Optional[CrawlStateManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = session or requests.Session()
        self.state_manager = state_manager

    def _get_html(self, url: str) -> Optional[str]:
        if self.state_manager is None:
            # Fallback to stateless behavior
            return super()._get_html(url)

        try:
            # Get conditional headers from the state manager
            db_info = self.state_manager.get_url_info(url)
            headers = {}
            if db_info:
                if db_info["etag"]: headers["If-None-Match"] = db_info["etag"]
                if db_info["last_modified"]: headers["If-Modified-Since"] = db_info["last_modified"]

            req_kwargs = getattr(self, "requests_kwargs", {})
            response = self.session.get(
                url, timeout=self.timeout, headers={**self.headers, **headers}, **req_kwargs
            )

            if response.status_code == 304:  # Not Modified
                logger.debug(f"Skipping unmodified content: {url}")
                return None  # Prevent loader from creating a Document

            response.raise_for_status()

            # If content was modified (200 OK), update state and return content
            self.state_manager.update_url_state(url, response)
            return response.text

        except Exception as e:
            logger.warning(f"Unable to load from {url}: {e}")
            return None


def crawl(start_url: str, base_url: str, max_depth: int = 2, session: Optional[Session] = None, state_manager: Optional[CrawlStateManager] = None) -> Iterable:
    """
    Crawl a target URL recursively up to a given depth.
    Yields documents lazily to preserve memory.
    """
    logger.debug(f"Crawling {start_url} with max depth {max_depth}")
    try:
        loader = CustomRecursiveUrlLoader(
            url=start_url,
            base_url=base_url,
            max_depth=max_depth,
            prevent_outside=True,
            timeout=REQUEST_TIMEOUT,
            check_response_status=True,
            exclude_dirs=EXCLUDE_DIRS,
            link_regex=HTML_LINK_REGEX,
            session=session,
            state_manager=state_manager
        )
        yield from loader.lazy_load()
    except Exception as e:
        logger.warning(f"  FAILED {start_url}: {e}")
        yield from iter([])


def extract_html_sitemap_urls(sitemap_url: str, base_url: str, session: Session) -> List[str]:
    """Extract deterministic section URLs from UNISA HTML sitemap pages."""
    logger.info(f"Fetching HTML sitemap: {sitemap_url}")
    try:
        response = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"  WARNING: could not fetch HTML sitemap {sitemap_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    base_netloc = urlparse(base_url).netloc
    valid_urls: Set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()

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

    with CrawlStateManager() as state_manager:
        with create_resilient_session() as session:
            seeds = extract_html_sitemap_urls(sitemap_url, base_url, session)

            if not seeds:
                logger.warning(f"  WARNING: no sitemap seeds found for {base_url}; falling back to direct crawl")
                yield from crawl(base_url, base_url=crawl_base_url, max_depth=fallback_depth, session=session, state_manager=state_manager)
                return

            for i, seed in enumerate(seeds, 1):
                seed_docs = crawl(seed, base_url=crawl_base_url, max_depth=max_depth, session=session, state_manager=state_manager)
                count = 0
                for doc in seed_docs:
                    count += 1
                    yield doc
                logger.debug(f"    [{i:02d}/{len(seeds)}] {seed} ({count} pages)")


def extract_course_focus_urls(course_url: str, session: Optional[Session] = None) -> List[str]:
    """Extract detail URLs from a course /focus page without broadening recursion."""
    focus_url = f"{course_url.rstrip('/')}/focus"
    active_session = session or create_resilient_session()
    should_close = session is None

    try:
        response = active_session.get(focus_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        logger.debug(f"Unable to fetch course focus page {focus_url}: {e}")
        if should_close:
            active_session.close()
        return []

    try:
        soup = BeautifulSoup(response.text, "lxml")
        focus_parsed = urlparse(focus_url)
        focus_path = focus_parsed.path.rstrip("/")
        detail_urls: Set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:")):
                continue

            absolute_url, _ = urldefrag(urljoin(focus_url, href))
            parsed = urlparse(absolute_url)
            query = parse_qs(parsed.query)
            focus_ids = query.get("id")

            if parsed.netloc != focus_parsed.netloc:
                continue
            if parsed.path.rstrip("/") != focus_path:
                continue
            if not focus_ids:
                continue

            normalized_query = urlencode({"id": focus_ids[0]})
            detail_urls.add(
                urlunparse(("https", parsed.netloc, parsed.path.rstrip("/"), "", normalized_query, ""))
            )

        return sorted(detail_urls)
    finally:
        if should_close:
            active_session.close()


def crawl_course_focus_detail_pages(course_url: str) -> List[Document]:
    """Fetch only /focus?id=... detail pages for a course as raw HTML documents."""
    docs: List[Document] = []

    with create_resilient_session() as session:
        focus_urls = extract_course_focus_urls(course_url, session=session)
        if not focus_urls:
            logger.debug(f"No course focus detail URLs found for {course_url}")
            return docs

        logger.info(f"  -> Found {len(focus_urls)} course focus detail URLs for {course_url}")

        for focus_url in focus_urls:
            try:
                response = session.get(focus_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                docs.append(Document(page_content=response.text, metadata={"source": focus_url}))
            except Exception as e:
                logger.warning(f"Unable to fetch course focus detail page {focus_url}: {e}")

    return docs


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
                href = anchor.get("href", "").strip()
                if href.startswith("http") and "corsi.unisa.it" in href:
                    parsed = urlparse(href)
                    first_segment = next(iter(parsed.path.strip("/").split("/")), None)
                    if first_segment:
                        course_urls.add(f"{parsed.scheme}://{parsed.netloc}/{first_segment}")
        except Exception as e:
            logger.debug(f"Failed to parse course URL from document {doc.metadata.get('source')}: {e}")

    logger.info(f"Extracted {len(course_urls)} course URLs")
    return sorted(course_urls)


def _fetch_personale_page(url: str, session: Session) -> str:
    """Fetch the main DIEM 'personale' HTML page."""
    logger.info(f"Fetching faculty list from {url}")
    response = session.get(url, timeout=REQUEST_TIMEOUT)
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


def _fetch_and_validate_faculty_profile(matricola: str, url: str, session: Session, state_manager: CrawlStateManager) -> Tuple[
    str, bool, Exception | str]:
    """Helper for concurrently validating a faculty profile page, using caching."""
    try:
        db_info = state_manager.get_url_info(url)
        headers = {}
        if db_info:
            if db_info["etag"]: headers["If-None-Match"] = db_info["etag"]
            if db_info["last_modified"]: headers["If-Modified-Since"] = db_info["last_modified"]

        response = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)

        if response.status_code == 304:
            logger.debug(f"Skipping unmodified faculty profile: {url}")
            metadata = json.loads(db_info["metadata_json"]) if db_info and db_info["metadata_json"] else {}
            has_profile = metadata.get("has_profile", False)
            return matricola, has_profile, "OK (Not Modified)"

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        has_profile = any(
            "docenti.unisa.it" in (anchor.get("href") or "")
            for anchor in soup.find_all("a", href=True)
        )
        state_manager.update_url_state(url, response, metadata={"has_profile": has_profile})
        return matricola, has_profile, "OK"
    except Exception as exc:
        return matricola, False, exc


def _validate_faculty_urls_concurrently(rubrica_links: Dict[str, str], session: Session, state_manager: CrawlStateManager) -> List[str]:
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
            executor.submit(_fetch_and_validate_faculty_profile, mid, rubrica_links[mid], session, state_manager): mid
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
        with CrawlStateManager() as state_manager:
            with create_resilient_session() as session:
                html = _fetch_personale_page(DIEM_PERSONALE_URL, session)
                rubrica_links = _parse_rubrica_links(html)
                return _validate_faculty_urls_concurrently(rubrica_links, session, state_manager)
    except Exception as e:
        logger.error(f"  WARNING: could not fetch or process faculty list from personale page: {e}")
        return []

# ==============================================================================
# POST-PROCESSING & I/O
# ==============================================================================

def filter_docs(docs: Iterable) -> list:
    """Drop docs whose source URL contains substrings we want to skip, then dedup by URL."""
    docs = list(docs)
    filtered = [
        d for d in docs
        if not any(p in d.metadata.get("source", "") for p in SKIP_DOCUMENT_SUBSTRINGS)
        and not is_pre_2020_url(d.metadata.get("source", ""))
    ]

    # Dedup by source URL — same URL crawled from multiple entry points yields identical content
    seen: set = set()
    deduped = []
    for d in filtered:
        url = d.metadata.get("source", "")
        if url not in seen:
            seen.add(url)
            deduped.append(d)

    logger.debug(f"Filtered {len(docs) - len(filtered)} documents by URL pattern/date, "
                 f"deduped {len(filtered) - len(deduped)} duplicate URLs")
    return deduped


def save_crawled_pdfs_to_json(pdf_docs: Iterable, filename: str) -> None:
    """Group PDF pages by source URL and save the summary to a JSON file."""
    pdfs_summary: Dict[str, Dict] = {}
    total_pages_count = 0

    for doc in pdf_docs:
        total_pages_count += 1
        source_url = doc.metadata.get("source", "")
        # Used a fallback variable since 'url' is not defined in this scope
        url_fallback = source_url

        if source_url not in pdfs_summary:
            pdfs_summary[source_url] = {
                "url": url_fallback,
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
