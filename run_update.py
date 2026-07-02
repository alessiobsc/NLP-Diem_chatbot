"""
Production-compatible refresh/update entrypoint for the DIEM Chroma store.

Two modes are available:
- default full refresh: clean rebuild with backup, safest for periodic releases;
- --incremental: re-index only changed source groups while keeping Chroma and
  parent_store consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import chromadb
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document as LCDocument

from config import CHROMA_DIR, CHROMA_DIR_NAME, COLLECTION_NAME, PARENT_STORE_DIR
from main_ingestion import (
    apply_html_metadata_and_filter,
    crawl_phase,
    dedupe_docs_by_source_alias_and_content,
)
from src.ingestion.crawler import save_crawled_pdfs_to_json, save_crawled_urls_to_json
from src.ingestion.crawl_state import CrawlStateManager
from src.ingestion.database import DocumentIndexer
from src.ingestion.parser import filter_low_quality_documents, filter_recent_documents
from src.encoders.embedding_init import build_embedding_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _next_backup_path(chroma_dir: Path) -> Path:
    base = chroma_dir.with_name(f"{chroma_dir.name}.backup-{_timestamp()}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def _move_existing_store(chroma_dir: Path, backup: bool) -> Path | None:
    if not chroma_dir.exists():
        logger.info(f"No existing Chroma store found at {chroma_dir}; building from scratch.")
        return None

    if backup:
        backup_path = _next_backup_path(chroma_dir)
        logger.info(f"Moving existing Chroma store to backup: {backup_path}")
        shutil.move(str(chroma_dir), str(backup_path))
        return backup_path

    logger.warning(f"Removing existing Chroma store without backup: {chroma_dir}")
    shutil.rmtree(chroma_dir)
    return None


def _restore_backup(chroma_dir: Path, backup_path: Path | None) -> None:
    if backup_path is None or not backup_path.exists():
        return

    failed_path = chroma_dir.with_name(f"{chroma_dir.name}.failed-{_timestamp()}")
    if chroma_dir.exists():
        logger.warning(f"Preserving failed rebuild at: {failed_path}")
        shutil.move(str(chroma_dir), str(failed_path))

    logger.warning(f"Restoring previous Chroma store from backup: {backup_path}")
    shutil.move(str(backup_path), str(chroma_dir))


def _content_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _group_docs_by_source(docs: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source:
            grouped[source].append(doc)
    return dict(grouped)


def _group_hash(docs: list) -> str:
    parts = []
    for doc in sorted(
        docs,
        key=lambda d: (d.metadata.get("title", ""), d.page_content[:200]),
    ):
        parts.append(doc.metadata.get("title", ""))
        parts.append(doc.metadata.get("context_header", ""))
        parts.append(doc.page_content)
    return _content_hash("\n\n".join(parts))


def _load_incremental_documents() -> tuple[list, set[str]]:
    """
    Crawl only documents returned by the incremental crawler and apply the same
    parsing/filtering gates used by the full pipeline.

    Returns final documents plus the raw source URLs seen in this incremental run.
    """
    raw_html_docs, pdf_docs = crawl_phase()
    raw_changed_sources = {
        doc.metadata.get("source", "")
        for doc in [*raw_html_docs, *pdf_docs]
        if doc.metadata.get("source")
    }

    logger.info(f"Applying HTML extractor + metadata gate to {len(raw_html_docs)} changed HTML documents...")
    raw_html_docs = apply_html_metadata_and_filter(raw_html_docs)
    raw_html_docs = dedupe_docs_by_source_alias_and_content(raw_html_docs)

    all_docs = raw_html_docs + pdf_docs
    logger.info(f"Changed documents before temporal/quality filtering: {len(all_docs)}")

    all_docs = filter_recent_documents(all_docs)
    all_docs = filter_low_quality_documents(all_docs)

    try:
        from src.ingestion.easycourse import fetch_easycourse_documents, fetch_easycourse_lectures

        exam_docs = fetch_easycourse_documents()
        lecture_docs = fetch_easycourse_lectures()
        all_docs = all_docs + exam_docs + lecture_docs
        logger.info(f"EasyCourse incremental refresh candidates: {len(exam_docs)} exam docs + {len(lecture_docs)} lecture docs")
    except Exception as e:
        logger.warning(f"EasyCourse fetch failed during incremental update, skipping: {e}")

    return all_docs, raw_changed_sources


def _load_full_documents() -> list:
    """Build the complete document set using the production ingestion steps."""
    raw_html_docs, pdf_docs = crawl_phase()

    logger.info(f"Applying HTML extractor + metadata gate to {len(raw_html_docs)} HTML documents...")
    raw_html_docs = apply_html_metadata_and_filter(raw_html_docs)
    raw_html_docs = dedupe_docs_by_source_alias_and_content(raw_html_docs)

    all_docs = raw_html_docs + pdf_docs
    logger.info(f"Total documents after URL + language filters, before temporal filtering: {len(all_docs)}")

    all_docs = filter_recent_documents(all_docs)
    all_docs = filter_low_quality_documents(all_docs)

    html_final = [d for d in all_docs if ".pdf" not in d.metadata.get("source", "").lower()]
    pdf_final = [d for d in all_docs if ".pdf" in d.metadata.get("source", "").lower()]
    save_crawled_urls_to_json(html_final, "crawled_urls.json")
    save_crawled_pdfs_to_json(pdf_final, "crawled_pdfs.json")

    try:
        from src.ingestion.easycourse import fetch_easycourse_documents, fetch_easycourse_lectures

        exam_docs = fetch_easycourse_documents()
        lecture_docs = fetch_easycourse_lectures()
        easycourse_docs = exam_docs + lecture_docs
        all_docs = all_docs + easycourse_docs
        with open("easycourse_docs.json", "w", encoding="utf-8") as f:
            json.dump(
                [{"url": d.metadata["source"], "content": d.page_content} for d in easycourse_docs],
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"EasyCourse: {len(exam_docs)} exam docs + {len(lecture_docs)} lecture docs")
    except Exception as e:
        logger.warning(f"EasyCourse fetch failed, skipping: {e}")

    static_facts = [
        LCDocument(
            page_content=(
                "Dati amministrativi dell'Università degli Studi di Salerno (UNISA):\n"
                "Sede: Via Giovanni Paolo II, 132 - 84084 Fisciano (SA)\n"
                "Partita IVA: 00851300657\n"
                "Codice Fiscale: 80018670655"
            ),
            metadata={
                "source": "https://www.diem.unisa.it/",
                "context_header": "contatti DIEM - dati amministrativi UNISA",
                "title": "Dati amministrativi UNISA",
            },
        )
    ]

    return all_docs + static_facts


def _write_index_state(docs_by_source: dict[str, list], indexer: DocumentIndexer) -> None:
    with CrawlStateManager() as crawl_state:
        for source, source_docs in sorted(docs_by_source.items()):
            crawl_state.update_index_state(
                source,
                _group_hash(source_docs),
                indexer.last_indexed_parent_ids_by_source.get(source, []),
            )


def _run_full_pipeline_with_index_state(embedding_model) -> None:
    crawl_state_path = Path("db/crawl_state.db")
    if crawl_state_path.exists():
        crawl_state_path.unlink()
        logger.info("Cleared crawl_state.db; full refresh will seed fresh crawl/index state")

    all_docs = _load_full_documents()
    logger.info(f"Total documents ready for indexing: {len(all_docs)}")

    indexer = DocumentIndexer(embedding_model)
    indexer.index(all_docs)
    _write_index_state(_group_docs_by_source(all_docs), indexer)
    logger.info("Seeded crawl_state.db with content hashes and parent IDs for incremental updates")


def bootstrap_state_from_chroma(dry_run: bool = False, batch_size: int = 1000) -> None:
    """
    Seed crawl_state.db from an already-built Chroma + parent_store.

    This recovers source -> parent IDs exactly from child metadata. The content
    hash is based on stored parent documents, so it is good enough to align the
    current index with future incremental deletes, but the next fetched version
    of a page may still be re-indexed once because the full pre-split source text
    is not stored verbatim.
    """
    logger.info("=" * 60)
    logger.info("Bootstrapping crawl_state.db from existing Chroma store")
    logger.info(f"Chroma directory: {CHROMA_DIR_NAME}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("=" * 60)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()

    source_to_parent_ids: dict[str, set[str]] = defaultdict(set)
    source_to_child_count: dict[str, int] = defaultdict(int)

    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for metadata in batch.get("metadatas") or []:
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source", "")
            parent_id = metadata.get("doc_id") or metadata.get("chunk_id")
            if not source:
                continue
            source_to_child_count[source] += 1
            if parent_id:
                source_to_parent_ids[source].add(parent_id)

    logger.info(
        f"Recovered {len(source_to_parent_ids)} source groups from "
        f"{total} child chunks in Chroma."
    )

    parent_store = create_kv_docstore(LocalFileStore(str(PARENT_STORE_DIR)))
    source_to_parent_docs: dict[str, list] = {}

    for source, parent_ids in source_to_parent_ids.items():
        ordered_parent_ids = sorted(parent_ids)
        parent_docs = [doc for doc in parent_store.mget(ordered_parent_ids) if doc is not None]
        if parent_docs:
            source_to_parent_docs[source] = parent_docs

    missing_sources = set(source_to_parent_ids) - set(source_to_parent_docs)
    if missing_sources:
        logger.warning(
            f"{len(missing_sources)} source groups had Chroma children but no readable parent docs."
        )

    if dry_run:
        logger.info("Dry run requested; crawl_state.db will not be changed.")
        logger.info(f"Would write index state for {len(source_to_parent_docs)} source groups.")
        return

    with CrawlStateManager() as crawl_state:
        for source, parent_docs in sorted(source_to_parent_docs.items()):
            crawl_state.update_index_state(
                source,
                _group_hash(parent_docs),
                sorted(source_to_parent_ids[source]),
            )

    logger.info(f"Seeded crawl_state.db for {len(source_to_parent_docs)} source groups.")


def run_update(backup: bool = True, dry_run: bool = False) -> None:
    """
    Rebuild the production Chroma store using the same pipeline as app.py --reindex.

    The old store is moved aside before indexing so stale child chunks and parent
    documents cannot survive across weekly refreshes. If indexing fails and a
    backup exists, the previous store is restored.
    """
    logger.info("=" * 60)
    logger.info("Starting DIEM weekly Chroma refresh")
    logger.info(f"Chroma directory: {CHROMA_DIR_NAME}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("Dry run requested; no files will be changed.")
        logger.info(f"Would rebuild Chroma store at: {CHROMA_DIR}")
        return

    backup_path: Path | None = None
    try:
        backup_path = _move_existing_store(CHROMA_DIR, backup=backup)
        embedding_model = build_embedding_model()
        _run_full_pipeline_with_index_state(embedding_model)
    except Exception:
        logger.exception("Weekly Chroma refresh failed.")
        _restore_backup(CHROMA_DIR, backup_path)
        raise

    logger.info("=" * 60)
    logger.info("Weekly Chroma refresh completed successfully")
    if backup_path:
        logger.info(f"Previous store kept at: {backup_path}")
    logger.info("=" * 60)


def run_incremental_update(dry_run: bool = False) -> None:
    """
    Incrementally update changed source groups.

    This mode avoids the unsafe prune step from the old implementation. It only
    replaces sources that were fetched and whose post-filter content hash changed.
    A full refresh should still be run periodically to remove URLs that genuinely
    disappeared from the website.
    """
    logger.info("=" * 60)
    logger.info("Starting DIEM incremental Chroma update")
    logger.info(f"Chroma directory: {CHROMA_DIR_NAME}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("Dry run requested; no files will be changed.")
        logger.info("Would crawl with conditional requests, hash fetched documents, and replace changed source groups.")
        return

    docs, raw_changed_sources = _load_incremental_documents()
    docs_by_source = _group_docs_by_source(docs)
    final_sources = set(docs_by_source)
    dropped_sources = raw_changed_sources - final_sources

    changed_sources: list[str] = []
    known_parent_ids: dict[str, list[str]] = {}
    source_hashes: dict[str, str] = {}

    with CrawlStateManager() as crawl_state:
        for source, source_docs in sorted(docs_by_source.items()):
            new_hash = _group_hash(source_docs)
            index_state = crawl_state.get_index_state(source)
            known_parent_ids[source] = index_state["parent_ids"]
            source_hashes[source] = new_hash

            if index_state["content_hash"] == new_hash:
                logger.info(f"Unchanged after hashing, skipping indexing: {source}")
                continue

            changed_sources.append(source)

        if not changed_sources and not dropped_sources:
            logger.info("No changed source groups found; Chroma store left untouched.")
            return

        embedding_model = build_embedding_model()
        indexer = DocumentIndexer(embedding_model)

        if dropped_sources:
            logger.info(f"Deleting {len(dropped_sources)} changed sources that no longer pass filters.")
            dropped_parent_ids = {
                source: crawl_state.get_index_state(source)["parent_ids"]
                for source in dropped_sources
            }
            indexer.delete_sources(dropped_sources, known_parent_ids=dropped_parent_ids)
            for source in dropped_sources:
                crawl_state.clear_index_state(source)

        if changed_sources:
            logger.info(f"Replacing {len(changed_sources)} changed source groups.")
            indexer.delete_sources(changed_sources, known_parent_ids=known_parent_ids)

            changed_docs = [
                doc
                for source in changed_sources
                for doc in docs_by_source[source]
            ]
            indexer.index(changed_docs)

            for source in changed_sources:
                parent_ids = indexer.last_indexed_parent_ids_by_source.get(source, [])
                crawl_state.update_index_state(source, source_hashes[source], parent_ids)

    logger.info("=" * 60)
    logger.info("Incremental Chroma update completed successfully")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the DIEM chatbot Chroma store using the production ingestion pipeline."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Delete the existing Chroma store instead of moving it to a timestamped backup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be refreshed without changing files.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only replace source groups whose post-filter content hash changed.",
    )
    parser.add_argument(
        "--bootstrap-state-from-chroma",
        action="store_true",
        help="Seed crawl_state.db index metadata from the existing Chroma + parent_store.",
    )
    args = parser.parse_args()

    if args.bootstrap_state_from_chroma:
        bootstrap_state_from_chroma(dry_run=args.dry_run)
    elif args.incremental:
        run_incremental_update(dry_run=args.dry_run)
    else:
        run_update(backup=not args.no_backup, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
