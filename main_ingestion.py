import argparse
import datetime
import hashlib
import os
import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
    html_extractor_for_source,
    load_pdfs_from_links,
    NON_ITALIAN_LANG_PREFIXES,
)
from src.ingestion.database import index_documents
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref",
}
MIN_DEDUP_CONTENT_CHARS = 300

DIEM_BANDI_STRUCTURE_ID = "300638"
RECENT_BANDI_MODULES = (
    ("139", "Incarichi di Insegnamento", "struttura"),
    ("504", "Collaborazioni con il Dipartimento", "struttura"),
    ("316", "Personale Tecnico Amministrativo", "struttura"),
    ("226", "Dottorati di Ricerca", None),
    ("67", "Assegni di Ricerca", "struttura"),
    ("292", "Borse di Ricerca", "struttura"),
    ("293", "Borse e Premi", "cdsStruttura"),
    ("505", "Altri bandi", "struttura"),
)


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


def build_recent_bandi_urls(reference_year: int | None = None) -> list[tuple[str, str, int]]:
    """Build explicit DIEM bandi URLs for current and previous year."""
    current_year = reference_year or datetime.datetime.now().year
    years = (current_year, current_year - 1)
    urls: list[tuple[str, str, int]] = []

    for year in years:
        for module_id, label, structure_param in RECENT_BANDI_MODULES:
            url = f"https://www.diem.unisa.it/home/bandi?modulo={module_id}"
            if structure_param:
                url += f"&{structure_param}={DIEM_BANDI_STRUCTURE_ID}"
            url += f"&anno={year}"
            urls.append((url, label, year))

    return urls


def canonicalize_source_url(url: str) -> str:
    """Normalize a source URL while preserving semantically relevant filters."""
    split = urlsplit((url or "").strip())
    scheme = split.scheme.lower() or "https"
    netloc = split.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = re.sub(r"/{2,}", "/", split.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in TRACKING_QUERY_KEYS or any(
            key_l.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES
        ):
            continue
        query_pairs.append((key, value))

    query = urlencode(sorted(query_pairs))
    return urlunsplit((scheme, netloc, path, query, ""))


def source_alias_key(url: str) -> str:
    """
    Build a cautious alias key for URLs that can point to the same source.

    The key intentionally keeps course slugs and teacher IDs separate; the
    content hash is only used as confirmation inside this alias group.
    """
    canonical = canonicalize_source_url(url)
    split = urlsplit(canonical)
    scheme = "https" if split.netloc.endswith("unisa.it") else split.scheme
    netloc = split.netloc
    path = split.path or "/"
    parts = [part for part in path.split("/") if part]

    if netloc == "www.diem.unisa.it" and path in {"/", "/home"}:
        path = "/home"

    if netloc == "corsi.unisa.it" and parts:
        course_slug = parts[0]
        if len(parts) == 1 or (len(parts) == 2 and parts[1] in {"home", "presentazione"}):
            path = f"/{course_slug}/__landing"

    return urlunsplit((scheme, netloc, path, "", ""))


def normalized_content_hash(text: str, min_chars: int = MIN_DEDUP_CONTENT_CHARS) -> str:
    """Return a stable hash for long normalized text, else an empty string."""
    normalized = re.sub(r"https?://\S+", "", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) < min_chars:
        return ""
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def dedupe_docs_by_source_alias_and_content(docs: list) -> list:
    """
    Deduplicate HTML docs only when URL alias and long-content hash both match.

    This avoids global content-hash dedup, so equal text across different courses
    or different teachers is preserved.
    """
    seen_confirmed: set[tuple[str, str]] = set()
    duplicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    unique_docs = []
    skipped_short = 0

    for doc in docs:
        source = doc.metadata.get("source", "")
        if not source:
            unique_docs.append(doc)
            continue

        content_hash = normalized_content_hash(doc.page_content)
        if not content_hash:
            skipped_short += 1
            unique_docs.append(doc)
            continue

        alias_key = source_alias_key(source)
        dedup_key = (alias_key, content_hash)
        if dedup_key in seen_confirmed:
            duplicate_groups[dedup_key].append(source)
            continue

        seen_confirmed.add(dedup_key)
        duplicate_groups[dedup_key].append(source)
        unique_docs.append(doc)

    removed = len(docs) - len(unique_docs)
    duplicate_group_count = sum(1 for sources in duplicate_groups.values() if len(sources) > 1)
    logger.info(
        "  -> Source-alias/content dedup: removed "
        f"{removed} duplicate HTML docs across {duplicate_group_count} confirmed groups "
        f"(kept {len(unique_docs)}/{len(docs)}, skipped {skipped_short} short docs under "
        f"{MIN_DEDUP_CONTENT_CHARS} chars)"
    )

    if removed:
        shown = 0
        for (alias_key, _), sources in duplicate_groups.items():
            if len(sources) <= 1:
                continue
            logger.debug(
                "  DEDUP source alias group: "
                f"alias={alias_key}; kept={sources[0]}; removed={sources[1:]}"
            )
            shown += 1
            if shown >= 20:
                break

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
    raw_diem = list(filter_docs(dedupe_docs_by_source(
        list(crawl_html_sitemap("https://www.diem.unisa.it/", max_depth=2, fallback_depth=4))
    )))
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

    recent_bandi_urls = build_recent_bandi_urls()
    logger.info(
        f"[1b/3] Crawling recent DIEM bandi explicit URLs "
        f"({len(recent_bandi_urls)} URLs: current year + previous year) ..."
    )
    bandi_docs_total = 0
    for i, (url, label, year) in enumerate(recent_bandi_urls, 1):
        docs = list(filter_docs(dedupe_docs_by_source(
            list(crawl(url, base_url="https://www.diem.unisa.it/", max_depth=1))
        )))
        batch_pdfs = load_pdfs_from_links(docs, seen_pdf_urls)
        raw_html_docs.extend(docs)
        pdf_docs.extend(batch_pdfs)
        bandi_docs_total += len(docs)
        logger.info(
            f"  [{i:02d}/{len(recent_bandi_urls)}] {year} {label} "
            f"({len(docs)} pages, {len(batch_pdfs)} PDF pages)"
        )
    logger.info(f"  -> Recent DIEM bandi crawl added {bandi_docs_total} HTML docs")

    # 1b. Crawl docenti.unisa.it
    total_docenti = len(docenti_urls)
    current_year = datetime.datetime.now().year
    logger.info(f"[2/3] Crawling docenti.unisa.it ({total_docenti} faculty pages) ...")
    for i, url in enumerate(docenti_urls, 1):
        base = get_section_base(url)
        docs = list(filter_docs(crawl(url, base_url=base, max_depth=3)))

        matricola = url.rstrip("/").split("/")[-2]
        for anno in range(2020, current_year + 1):
            pub_url = f"https://docenti.unisa.it/{matricola}/ricerca/pubblicazioni?anno={anno}"
            docs.extend(list(filter_docs(crawl(pub_url, base_url=base, max_depth=1))))

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
        docs = list(filter_docs(dedupe_docs_by_source(
            list(crawl_html_sitemap(
                url,
                max_depth=1,
                crawl_base_url="https://corsi.unisa.it/",
                fallback_depth=3,
            ))
        )))
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
    dropped_empty = 0
    for doc in raw_html_docs:
        source = doc.metadata.get("source", "")
        html_meta = extract_html_metadata(doc.page_content)
        doc.metadata.update(html_meta)
        if "date" in html_meta:
            date_enriched += 1
        doc.page_content = html_extractor_for_source(doc.page_content, source)
        if not doc.page_content.strip():
            logger.debug(f"  SKIP empty/low-value extraction: {source}")
            dropped_empty += 1
            continue
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
    if dropped_empty:
        logger.info(f"  -> Structured/HTML extractor dropped {dropped_empty} empty or low-value pages")
    return kept


def run_full_pipeline(embedding_model) -> None:
    raw_html_docs, pdf_docs = crawl_phase()

    logger.info(f"Applying HTML extractor + metadata gate to {len(raw_html_docs)} HTML documents...")
    raw_html_docs = apply_html_metadata_and_filter(raw_html_docs)
    raw_html_docs = dedupe_docs_by_source_alias_and_content(raw_html_docs)

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
        raw_html_docs = dedupe_docs_by_source_alias_and_content(raw_html_docs)
        save_crawled_urls_to_json(raw_html_docs, "crawled_urls.json")
        save_crawled_pdfs_to_json(pdf_docs, "crawled_pdfs.json")
        total = len(raw_html_docs) + len(pdf_docs)
        logger.info(f"Crawl-only complete. {len(raw_html_docs)} HTML docs, {len(pdf_docs)} PDF docs ({total} total).")
        logger.info("Stopped before temporal filtering, enrichment, and indexing.")
        return

    # --full
    from src.encoders.embedding_init import build_embedding_model

    embedding_model = build_embedding_model()
    run_full_pipeline(embedding_model)


if __name__ == "__main__":
    main()
