"""
Inspect extracted text from a single URL using the project's parsing pipeline.

Usage:
    venv/Scripts/python scripts/inspect_url.py <url>
    venv/Scripts/python scripts/inspect_url.py <url> --raw      # show raw HTML instead
    venv/Scripts/python scripts/inspect_url.py <url> --meta     # show metadata too
"""

import argparse
import io
import os
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.crawler import crawl, get_section_base
from src.ingestion.parser import extract_html_metadata, html_extractor_for_source


def main():
    parser = argparse.ArgumentParser(description="Inspect parsed content from a URL")
    parser.add_argument("url", help="URL to inspect")
    parser.add_argument("--raw", action="store_true", help="Show raw HTML instead of extracted text")
    parser.add_argument("--meta", action="store_true", help="Show metadata")
    args = parser.parse_args()

    url = args.url
    base = get_section_base(url)

    print(f"Crawling: {url}", flush=True)
    docs = list(crawl(url, base_url=base, max_depth=1))

    if not docs:
        print("No documents returned.")
        sys.exit(1)

    doc = next((d for d in docs if d.metadata.get("source", "") == url), docs[0])
    raw_html = doc.page_content

    if args.meta:
        meta = {**doc.metadata, **extract_html_metadata(raw_html)}
        print("\n--- METADATA ---")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    if args.raw:
        print("\n--- RAW HTML ---")
        print(raw_html)
    else:
        text = html_extractor_for_source(raw_html, url)
        char_count = len(text) if text else 0
        print(f"\n--- EXTRACTED TEXT ({char_count} chars) ---")
        print(text if text else "[empty -- html_extractor returned nothing]")


if __name__ == "__main__":
    main()
