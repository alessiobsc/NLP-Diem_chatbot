import json
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

from src.logger import get_logger

logger = get_logger(__name__)

HTML_LINK_REGEX = r"""<a\s+(?:[^>]*?\s+)?href=["']([^"']*)["']"""

EXCLUDE_DIRS = [
    "/rescue/css/", "/rescue/js/", "/rescue/assets/",
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot",
    "/idp/", "/password-recovery", "/login",
]

OFFERTA_FORMATIVA_PATH = "/didattica/offerta-formativa"


def is_pre_2020_url(url: str) -> bool:
    normalized = url.replace("_", "/")
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20[01]\d)\b", normalized)]
    return any(y < 2020 for y in years)


def get_section_base(url: str) -> str:
    """Return first-segment root of a URL path.

    https://docenti.unisa.it/003145/home  ->  https://docenti.unisa.it/003145/
    https://corsi.unisa.it/ing-inf/home   ->  https://corsi.unisa.it/ing-inf/
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    base_path = f"/{parts[0]}/" if parts and parts[0] else "/"
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def crawl(start_url: str, base_url: str, max_depth: int = 2) -> list:
    try:
        logger.debug(f"Crawling {start_url} with max depth {max_depth}")
        loader = RecursiveUrlLoader(
            start_url,
            base_url=base_url,
            max_depth=max_depth,
            prevent_outside=True,
            timeout=15,
            check_response_status=True,
            exclude_dirs=EXCLUDE_DIRS,
            link_regex=HTML_LINK_REGEX,
        )
        return loader.load()
    except Exception as e:
        logger.warning(f"  FAILED {start_url}: {e}")
        return []


def extract_corsi_urls(raw_docs: list) -> list[str]:
    corsi: set[str] = set()
    source_docs = [
        d for d in raw_docs
        if OFFERTA_FORMATIVA_PATH in d.metadata.get("source", "")
        and "?anno=" not in d.metadata.get("source", "")
    ]
    for doc in source_docs:
        try:
            soup = BeautifulSoup(doc.page_content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("http") and "corsi.unisa.it" in href:
                    parsed = urlparse(href)
                    first_seg = parsed.path.strip("/").split("/")[0]
                    if first_seg:
                        corsi.add(f"{parsed.scheme}://{parsed.netloc}/{first_seg}")
        except Exception:
            pass
    logger.info(f"Extracted {len(corsi)} course URLs")
    return sorted(corsi)


DIEM_PERSONALE_URL = "https://www.diem.unisa.it/dipartimento/personale"


def extract_diem_faculty_urls() -> list[str]:
    """Scrape DIEM personale page → validated docenti.unisa.it URLs.

    Personale page links to rubrica.unisa.it/persone?matricola=XXXXXX.
    For each rubrica link, fetches the page and checks for a docenti.unisa.it
    link — skips matricole whose profile page is an empty stub.
    """
    try:
        logger.info(f"Fetching faculty list from {DIEM_PERSONALE_URL}")
        resp = requests.get(DIEM_PERSONALE_URL, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rubrica_links: dict[str, str] = {}  # matricola → rubrica URL
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("http"):
                continue  # rubrica è cross-domain: link relativi non sarebbero fetchabili
            m = re.search(r"matricola=(\d+)", href)
            if m:
                rubrica_links[m.group(1)] = href

        matricole = sorted(rubrica_links)
        logger.info(f"  -> {len(matricole)} matricole found on personale page, validating via rubrica...")

        urls: list[str] = []
        for i, mid in enumerate(matricole, 1):
            try:
                r = requests.get(rubrica_links[mid], timeout=10)
                r.raise_for_status()
                rubrica_soup = BeautifulSoup(r.text, "html.parser")
                has_profile = any(
                    "docenti.unisa.it" in (a.get("href") or "")
                    for a in rubrica_soup.find_all("a", href=True)
                )
                status = "OK" if has_profile else "SKIP (no profile)"
                logger.debug(f"    [{i:02d}/{len(matricole)}] {mid}: {status}")
                if has_profile:
                    urls.append(f"https://docenti.unisa.it/{mid}/home")
            except Exception as e:
                logger.warning(f"    [{i:02d}/{len(matricole)}] {mid}: ERROR {e} — skipping")
            time.sleep(0.5)

        logger.info(f"  -> {len(urls)}/{len(matricole)} faculty URLs validated")
        return urls
    except Exception as e:
        logger.error(f"  WARNING: could not fetch faculty list from personale page: {e}")
        return []


def filter_docs(docs: list) -> list:
    """Drop docs whose source URL contains substrings we want to skip."""
    SKIP_SUBSTRINGS = (
        "?sitemap",
        # unisa-rescue-page treats JS relative hrefs (print(), history.go(-1)) as
        # literal row IDs. /row/ID/print() duplicates /row/ID/<slug>;
        # /row/print() is a server fallback for invalid IDs.
        "/print()",
        "/history.go(",
        # PDFs are handled by load_pdfs_from_links via PyPDFLoader.
        # RecursiveUrlLoader reads them as binary → garbage in page_content.
        ".pdf",
        # English/Chinese versions of Italian pages — duplicate content, IT is canonical.
        "/en/",
        "/zh/",
    )
    filtered = [
        d for d in docs
        if not any(p in d.metadata.get("source", "") for p in SKIP_SUBSTRINGS)
        and not is_pre_2020_url(d.metadata.get("source", ""))
    ]
    logger.debug(f"Filtered {len(docs) - len(filtered)} documents")
    return filtered


def save_crawled_pdfs_to_json(pdf_docs: list, filename: str) -> None:
    """Group PDF pages by source URL and save summary to JSON."""
    pdfs: dict[str, dict] = {}
    for doc in pdf_docs:
        url = doc.metadata.get("source", "")
        if url not in pdfs:
            pdfs[url] = {
                "url": url,
                "source_page": doc.metadata.get("source_page", ""),
                "pages": 0,
            }
        pdfs[url]["pages"] += 1

    entries = sorted(pdfs.values(), key=lambda x: x["url"])
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info(f"  -> Saved {len(entries)} PDF sources ({len(pdf_docs)} pages total) to {filename}")


def save_crawled_urls_to_json(docs: list, filename: str) -> None:
    """Extract URL and <title> from docs and save to JSON.

    Reads title from doc.metadata["title"] if already extracted by
    extract_html_metadata(); falls back to BeautifulSoup parse when
    page_content is still raw HTML (e.g. during --full pipeline).
    """
    entries = []
    for doc in docs:
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        if not title:
            try:
                soup = BeautifulSoup(doc.page_content, "html.parser")
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True)
            except Exception:
                pass
        entries.append({"url": url, "title": title})

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info(f"  -> Saved {len(entries)} URLs to {filename}")
