"""
Diagnostic: sample pages from each domain in Chroma and report
what date-related HTML tags and HTTP headers are actually present.
"""

import io
import sys
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app import embedding_model

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_DIMENSION

SAMPLE_PER_DOMAIN = 10
FETCH_TIMEOUT = 10
BATCH_SIZE = 2000

DATE_META_NAMES = [
    "date", "pubdate", "publishdate",
    "article:published_time", "article:modified_time",
    "og:updated_time",
    "dc.date", "dcterms.date", "dcterms.modified",
    "last-modified",
]

ITALIAN_DATE_RE = [
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",           # 12/03/2024
    r"\b\d{1,2}\s+\w+\s+\d{4}\b",            # 12 marzo 2024
    r"\bAggiornato\s+il\s+\d",               # Aggiornato il 12...
    r"\bPubblicato\s*[il:]\s*\d",            # Pubblicato il / Pubblicato: 12...
    r"\bData\s*[:\s]\s*\d{1,2}",             # Data: 12...
    r"\bUltimo\s+aggiornamento",             # Ultimo aggiornamento
]

import re
_ITAL_RE = [re.compile(p, re.IGNORECASE) for p in ITALIAN_DATE_RE]


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def fetch_urls_from_chroma() -> dict[str, list[str]]:
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
    )
    domain_urls: dict[str, set[str]] = defaultdict(set)
    offset = 0
    while True:
        batch = vectorstore.get(limit=BATCH_SIZE, offset=offset, include=["metadatas"])
        metas = batch.get("metadatas") or []
        if not metas:
            break
        for m in metas:
            url = (m or {}).get("source", "")
            if url and not url.lower().endswith(".pdf"):
                domain_urls[domain_of(url)].add(url)
        offset += len(metas)
        if len(metas) < BATCH_SIZE:
            break
    return {d: list(urls) for d, urls in domain_urls.items()}


def inspect_page(url: str) -> dict:
    result = {
        "url": url,
        "ok": False,
        "last_modified_header": False,
        "meta_tags": [],      # which meta name/property values found
        "time_datetime": False,
        "time_year": None,
        "italian_pattern": False,
        "italian_match": "",
    }
    try:
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        result["ok"] = True

        if resp.headers.get("Last-Modified"):
            result["last_modified_header"] = True

        soup = BeautifulSoup(resp.text, "html.parser")

        # meta tags
        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").lower().strip()
            if name in DATE_META_NAMES:
                content = (tag.get("content") or "").strip()
                if content:
                    result["meta_tags"].append(f"{name}={content[:40]}")

        # <time datetime>
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            dt = time_tag["datetime"]
            result["time_datetime"] = True
            m = re.search(r"(20\d{2})", dt)
            if m:
                result["time_year"] = int(m.group(1))

        # Italian date patterns in visible text
        text = soup.get_text(" ", strip=True)[:3000]
        for pattern in _ITAL_RE:
            m = pattern.search(text)
            if m:
                result["italian_pattern"] = True
                result["italian_match"] = text[max(0, m.start()-10):m.end()+30].strip()
                break

    except Exception as e:
        result["error"] = str(e)[:80]

    return result


def report_domain(domain: str, results: list[dict]) -> None:
    ok = [r for r in results if r["ok"]]
    n = len(ok)
    if n == 0:
        print(f"  All {len(results)} fetches failed.")
        return

    lm = sum(1 for r in ok if r["last_modified_header"])
    time_dt = sum(1 for r in ok if r["time_datetime"])
    ital = sum(1 for r in ok if r["italian_pattern"])

    # meta tag breakdown
    meta_counter: Counter = Counter()
    for r in ok:
        for tag in r["meta_tags"]:
            name = tag.split("=")[0]
            meta_counter[name] += 1

    print(f"  Fetched OK: {n}/{len(results)}")
    print(f"  Last-Modified header    : {lm}/{n}")
    print(f"  <time datetime>         : {time_dt}/{n}")
    if time_dt:
        years = [r["time_year"] for r in ok if r["time_year"]]
        if years:
            print(f"    years seen            : {sorted(set(years))}")
    print(f"  Italian date in text    : {ital}/{n}")
    if meta_counter:
        print(f"  <meta> date tags found  :")
        for tag, cnt in meta_counter.most_common():
            print(f"    {tag:<35s} {cnt}/{n}")
    else:
        print(f"  <meta> date tags found  : none")

    # sample italian matches
    matches = [r["italian_match"] for r in ok if r["italian_match"]]
    if matches:
        print(f"  Sample Italian matches  :")
        for m in matches[:3]:
            print(f"    '{m}'")


def main() -> None:
    print("=== Date Tag Diagnostic ===")
    print(f"Sample size: {SAMPLE_PER_DOMAIN} HTML pages per domain\n")

    if not CHROMA_DIR.exists():
        print("ERROR: chroma_diem/ not found.")
        return

    print("Loading URLs from Chroma...")
    domain_urls = fetch_urls_from_chroma()

    for domain, urls in sorted(domain_urls.items()):
        sample = random.sample(urls, min(SAMPLE_PER_DOMAIN, len(urls)))
        print(f"\n[{domain}]  ({len(urls)} unique HTML URLs, sampling {len(sample)})")
        results = []
        for url in sample:
            r = inspect_page(url)
            status = "OK" if r["ok"] else f"FAIL ({r.get('error', '?')})"
            print(f"  {status[:60]}  {url[:80]}")
            results.append(r)
            time.sleep(0.3)
        print()
        report_domain(domain, results)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()