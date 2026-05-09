import argparse
import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import RecursiveUrlLoader

from src.ingestion.crawler import (
    HTML_LINK_REGEX,
    EXCLUDE_DIRS,
    crawl,
    extract_corsi_urls,
    extract_diem_faculty_urls,
    filter_docs,
    get_section_base,
    save_crawled_pdfs_to_json,
    save_crawled_urls_to_json,
)
from src.ingestion.parser import filter_recent_documents, html_extractor, load_pdfs_from_links
from src.ingestion.enrichment import add_context_headers
from src.ingestion.database import index_documents


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Crawl
# Returns (raw_html_docs, pdf_docs) — html_extractor NOT yet applied.
# load_pdfs_from_links is called here because it needs raw HTML to find links.
# ─────────────────────────────────────────────────────────────────────────────
def crawl_phase() -> tuple[list, list]:
    print("\n" + "=" * 60)
    print("PHASE 1 – Loading web pages")
    print("=" * 60)

    raw_html_docs: list = []
    pdf_docs: list = []
    seen_pdf_urls: set = set()

    # 1a. Crawl diem.unisa.it — keep raw HTML to extract external links
    print("\n[1/3] Crawling www.diem.unisa.it ...")
    raw_loader = RecursiveUrlLoader(
        "https://www.diem.unisa.it/",
        base_url="https://www.diem.unisa.it/",
        max_depth=3,
        prevent_outside=True,
        timeout=15,
        check_response_status=True,
        exclude_dirs=EXCLUDE_DIRS,
        link_regex=HTML_LINK_REGEX,
    )
    raw_diem = filter_docs(raw_loader.load())
    print(f"  -> {len(raw_diem)} pages found")

    corsi_urls = extract_corsi_urls(raw_diem)

    # Faculty list from /dipartimento/personale (DIEM official staff list,
    # validated via rubrica — skips empty stubs and excludes non-DIEM faculty).
    docenti_urls = sorted(extract_diem_faculty_urls())
    print(f"  -> {len(docenti_urls)} docenti URLs (from personale page), "
          f"{len(corsi_urls)} corsi links")
    print("\nDocenti URLs:")
    for url in docenti_urls:
        print(f"  {url}")

    diem_pdfs = load_pdfs_from_links(raw_diem, seen_pdf_urls)
    raw_html_docs.extend(raw_diem)
    pdf_docs.extend(diem_pdfs)

    # 1b. Crawl docenti.unisa.it
    total_docenti = len(docenti_urls)
    print(f"\n[2/3] Crawling docenti.unisa.it ({total_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls, 1):
        base = get_section_base(url)
        docs = filter_docs(crawl(url, base_url=base, max_depth=2))
        batch_pdfs = load_pdfs_from_links(docs, seen_pdf_urls)
        raw_html_docs.extend(docs)
        pdf_docs.extend(batch_pdfs)
        print(f"  [{i:02d}/{total_docenti}] {url}  ({len(docs)} sub-pages)")

    # 1c. Crawl corsi.unisa.it
    total_corsi = len(corsi_urls)
    print(f"\n[3/3] Crawling corsi.unisa.it ({total_corsi} course pages) ...")
    for i, url in enumerate(corsi_urls, 1):
        base = get_section_base(url)
        docs = filter_docs(crawl(url, base_url=base, max_depth=2))
        batch_pdfs = load_pdfs_from_links(docs, seen_pdf_urls)
        raw_html_docs.extend(docs)
        pdf_docs.extend(batch_pdfs)
        print(f"  [{i:02d}/{total_corsi}] {url}  ({len(docs)} sub-pages)")

    return raw_html_docs, pdf_docs


def run_full_pipeline(embedding_model) -> None:
    raw_html_docs, pdf_docs = crawl_phase()

    print(f"\nApplying HTML extractor to {len(raw_html_docs)} HTML documents...")
    for doc in raw_html_docs:
        doc.page_content = html_extractor(doc.page_content)

    all_docs = raw_html_docs + pdf_docs
    print(f"Total documents loaded before temporal filtering: {len(all_docs)}")

    all_docs = filter_recent_documents(all_docs)
    add_context_headers(all_docs)
    print(f"\nTotal documents after enrichment: {len(all_docs)}")

    index_documents(all_docs, embedding_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="DIEM ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--crawl-only",
        action="store_true",
        help=(
            "Run only Phase 1 (crawling). Saves URLs + titles to crawled_urls.json. "
            "Stops before temporal filtering, enrichment, and indexing."
        ),
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run the complete pipeline: crawl, filter, enrich, embed, index.",
    )
    args = parser.parse_args()

    if args.crawl_only:
        raw_html_docs, pdf_docs = crawl_phase()
        save_crawled_urls_to_json(raw_html_docs, "crawled_urls.json")
        save_crawled_pdfs_to_json(pdf_docs, "crawled_pdfs.json")
        total = len(raw_html_docs) + len(pdf_docs)
        print(f"\nCrawl-only complete. {len(raw_html_docs)} HTML docs, {len(pdf_docs)} PDF docs ({total} total).")
        print("Stopped before temporal filtering, enrichment, and indexing.")
        return

    # --full
    from brain import embedding_model
    run_full_pipeline(embedding_model)


if __name__ == "__main__":
    main()
