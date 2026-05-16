import argparse
import datetime
import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.crawler import (
    crawl,
    crawl_html_sitemap,
    extract_corsi_urls,
    extract_diem_faculty_urls,
    filter_docs,
    get_section_base,
    save_crawled_pdfs_to_json,
    save_crawled_urls_to_json,
)
from src.ingestion.parser import (
    extract_html_metadata,
    filter_low_quality_documents,
    filter_recent_documents,
    html_extractor,
    load_pdfs_from_links,
    NON_ITALIAN_LANG_PREFIXES,
)
from src.ingestion.database import index_documents
from src.logger import get_logger

logger = get_logger(__name__)


def dedupe_docs_by_source(docs: list) -> list:
    """Keep the first document for each source URL."""
    seen: set[str] = set()
    unique_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if not source or source in seen:
            continue
        seen.add(source)
        unique_docs.append(doc)
    return unique_docs

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Crawl
# Returns (raw_html_docs, pdf_docs) — html_extractor NOT yet applied.
# load_pdfs_from_links is called here because it needs raw HTML to find links.
# ─────────────────────────────────────────────────────────────────────────────
# TODO (Software Architect): Break down `crawl_phase` into smaller, single-responsibility functions.
def crawl_phase() -> tuple[list, list]:
    logger.info("=" * 60)
    logger.info("PHASE 1 – Loading web pages")
    logger.info("=" * 60)

    raw_html_docs: list = []
    pdf_docs: list = []
    seen_pdf_urls: set = set()

    # 1a. Crawl diem.unisa.it from deterministic HTML sitemap section seeds.
    # Keep raw HTML to extract external links and PDFs.
    logger.info("[1/3] Crawling www.diem.unisa.it from HTML sitemap ...")
    raw_diem = filter_docs(dedupe_docs_by_source(
        crawl_html_sitemap("https://www.diem.unisa.it/", max_depth=2, fallback_depth=4)
    ))
    logger.info(f"  -> {len(raw_diem)} pages found")

    corsi_urls = extract_corsi_urls(raw_diem)

    # Faculty list from /dipartimento/personale (DIEM official staff list,
    # validated via rubrica — skips empty stubs and excludes non-DIEM faculty).
    docenti_urls = sorted(extract_diem_faculty_urls())
    logger.info(f"  -> {len(docenti_urls)} docenti URLs (from personale page), "
          f"{len(corsi_urls)} corsi links")
    logger.debug("Docenti URLs:")
    for url in docenti_urls:
        logger.debug(f"  {url}")

    diem_pdfs = load_pdfs_from_links(raw_diem, seen_pdf_urls)
    raw_html_docs.extend(raw_diem)
    pdf_docs.extend(diem_pdfs)

    # 1b. Crawl docenti.unisa.it
    total_docenti = len(docenti_urls)
    current_year = datetime.datetime.now().year
    logger.info(f"[2/3] Crawling docenti.unisa.it ({total_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls, 1):
        base = get_section_base(url)
        docs = filter_docs(crawl(url, base_url=base, max_depth=3))

        matricola = url.rstrip("/").split("/")[-2]
        for anno in range(2020, current_year + 1):
            pub_url = f"https://docenti.unisa.it/{matricola}/ricerca/pubblicazioni?anno={anno}"
            docs.extend(filter_docs(crawl(pub_url, base_url=base, max_depth=1)))

        non_pub = [d for d in docs if "/pubblicazioni" not in d.metadata.get("source", "")]
        batch_pdfs = load_pdfs_from_links(non_pub, seen_pdf_urls)
        raw_html_docs.extend(docs)
        pdf_docs.extend(batch_pdfs)
        logger.info(f"  [{i:02d}/{total_docenti}] {url}  ({len(docs)} sub-pages)")

    # 1c. Crawl corsi.unisa.it
    total_corsi = len(corsi_urls)
    logger.info(f"[3/3] Crawling corsi.unisa.it ({total_corsi} course sitemaps) ...")
    for i, url in enumerate(corsi_urls, 1):
        # Use corsi.unisa.it domain as base (not course-specific path) so the crawler
        # can follow internal numeric-ID URLs (e.g. /0650107303300001/...) that some
        # course sites use instead of the slug-based path.
        docs = filter_docs(dedupe_docs_by_source(
            crawl_html_sitemap(
                url,
                max_depth=1,
                crawl_base_url="https://corsi.unisa.it/",
                fallback_depth=3,
            )
        ))
        batch_pdfs = load_pdfs_from_links(docs, seen_pdf_urls)
        raw_html_docs.extend(docs)
        pdf_docs.extend(batch_pdfs)
        logger.info(f"  [{i:02d}/{total_corsi}] {url}  ({len(docs)} sub-pages)")

    return raw_html_docs, pdf_docs


def apply_html_metadata_and_filter(raw_html_docs: list) -> list:
    """Extract title/lang/date from raw HTML, run html_extractor, drop non-Italian pages.

    Must be called while page_content is still raw HTML. After this call:
    - doc.metadata gains "title", "language", and/or "date" keys where present
    - doc.page_content is replaced with extracted plain text
    - docs with explicit non-Italian lang attribute are removed
    """
    kept, date_enriched, dropped = [], 0, 0
    for doc in raw_html_docs:
        html_meta = extract_html_metadata(doc.page_content)
        doc.metadata.update(html_meta)
        if "date" in html_meta:
            date_enriched += 1
        doc.page_content = html_extractor(doc.page_content)
        if doc.metadata.get("language", "").startswith(NON_ITALIAN_LANG_PREFIXES):
            logger.debug(f"  SKIP non-IT (lang={doc.metadata['language']}): {doc.metadata.get('source', '')}")
            dropped += 1
        else:
            kept.append(doc)

    logger.info(
        f"  -> Metadata extraction: {date_enriched}/{len(raw_html_docs)} HTML docs "
        f"got explicit date metadata"
    )
    logger.info(
        f"  -> Language filter (metadata-based): dropped {dropped} non-Italian pages "
        f"(kept {len(kept)}/{len(raw_html_docs)})"
    )
    return kept


def run_full_pipeline(embedding_model) -> None:
    raw_html_docs, pdf_docs = crawl_phase()

    logger.info(f"Applying HTML extractor + metadata gate to {len(raw_html_docs)} HTML documents...")
    raw_html_docs = apply_html_metadata_and_filter(raw_html_docs)

    all_docs = raw_html_docs + pdf_docs
    logger.info(f"Total documents after URL + language filters, before temporal filtering: {len(all_docs)}")

    all_docs = filter_recent_documents(all_docs)
    all_docs = filter_low_quality_documents(all_docs)

    # Save after ALL filters so JSON reflects exactly what enters the vector store
    html_final = [d for d in all_docs if ".pdf" not in d.metadata.get("source", "").lower()]
    pdf_final  = [d for d in all_docs if ".pdf" in d.metadata.get("source", "").lower()]
    save_crawled_urls_to_json(html_final, "crawled_urls.json")
    save_crawled_pdfs_to_json(pdf_final, "crawled_pdfs.json")

    logger.info(f"Total documents ready for indexing: {len(all_docs)}")
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
        # Apply metadata extraction + language filter before saving.
        # Title is stored in doc.metadata["title"] by extract_html_metadata so
        # save_crawled_urls_to_json can read it even after html_extractor runs.
        raw_html_docs = apply_html_metadata_and_filter(raw_html_docs)
        save_crawled_urls_to_json(raw_html_docs, "crawled_urls.json")
        save_crawled_pdfs_to_json(pdf_docs, "crawled_pdfs.json")
        total = len(raw_html_docs) + len(pdf_docs)
        logger.info(f"Crawl-only complete. {len(raw_html_docs)} HTML docs, {len(pdf_docs)} PDF docs ({total} total).")
        logger.info("Stopped before temporal filtering, enrichment, and indexing.")
        return

    # --full
    # TODO (Software Architect): Avoid importing `embedding_model` locally here to prevent circular dependencies or hidden side effects.
    from src.brain import embedding_model
    run_full_pipeline(embedding_model)


if __name__ == "__main__":
    main()