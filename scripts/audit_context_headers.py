"""
Audit contextual headers stored in the Chroma child chunks.

The script is read-only with respect to the vector store. It reads Chroma's
SQLite metadata table, groups chunks by contextual header and source URL, then
writes a JSON report plus a compact Markdown summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIR, COLLECTION_NAME, PARENT_STORE_DIR  # noqa: E402


DEFAULT_JSON_OUT = PROJECT_ROOT / "evaluation" / "results" / "context_header_audit.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "evaluation" / "results" / "context_header_audit_summary.md"

GENERIC_HEADER_PATTERNS = (
    "pagina diem",
    "pagina generale",
    "pagina istituzionale",
    "servizio",
    "profilo docente",
    "progetto di ricerca",
    "dettagli riguardanti",
    "informazioni sul profilo",
)

ASSERTIVE_PATTERNS = (
    "gestisce",
    "contiene informazioni",
    "prepara",
    "fornisce",
    "ha la finalità",
    "è richiesto",
    "sono richiesti",
    "possono immatricolarsi",
)

RAW_PDF_MARKERS = (
    "%pdf-",
    "/type /page",
    "/mediabox",
    "/cropbox",
    "/contents",
    "/resources",
    " endobj",
    " startxref",
)

TYPE_HINTS = {
    "docente": ("docenti.unisa.it", "personale", "professore", "prof.ssa"),
    "pubblicazioni": ("pubblicazioni", "iris"),
    "insegnamento": ("insegnamento", "syllabus", "cfu", "programma"),
    "corso": ("corsi.unisa.it", "corso di laurea", "laurea magistrale"),
    "regolamento": ("regolamento",),
    "avviso": ("avviso", "news", "bando", "seminario", "evento"),
    "progetto": ("progetti", "ricerca", "finanziati"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit contextual headers stored in Chroma child chunks."
    )
    parser.add_argument("--db", type=Path, default=CHROMA_DIR / "chroma.sqlite3")
    parser.add_argument("--parent-store", type=Path, default=PARENT_STORE_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--top-headers", type=int, default=200)
    parser.add_argument("--top-sources", type=int, default=200)
    parser.add_argument("--preview-per-header", type=int, default=3)
    parser.add_argument("--preview-per-source", type=int, default=3)
    parser.add_argument("--preview-chars", type=int, default=800)
    parser.add_argument(
        "--max-review-items",
        type=int,
        default=80,
        help="Maximum number of suspicious headers listed in needs_manual_review.",
    )
    return parser.parse_args()


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc or "unknown"
    except Exception:
        return "unknown"


def is_pdf(source: str, content_type: str = "") -> bool:
    source_path = urlparse(source).path.lower()
    return source_path.endswith(".pdf") or "pdf" in content_type.lower()


def one_line(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def is_raw_pdf_artifact(text: str) -> bool:
    if not text:
        return False
    scan = text[:2500].lower()
    if scan.lstrip().startswith("%pdf-"):
        return True
    marker_hits = sum(marker in scan for marker in RAW_PDF_MARKERS)
    object_hits = len(re.findall(r"\b\d+\s+\d+\s+obj\b", scan))
    return (marker_hits >= 3 and object_hits >= 1) or marker_hits >= 4


def metadata_value(row: sqlite3.Row) -> Any:
    for key in ("string_value", "int_value", "float_value", "bool_value"):
        value = row[key]
        if value is not None:
            return value
    return None


def fetch_chunks(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Chroma SQLite database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
                e.id AS row_id,
                e.embedding_id,
                m.key,
                m.string_value,
                m.int_value,
                m.float_value,
                m.bool_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            ORDER BY e.id
        """
        chunks_by_row: dict[int, dict[str, Any]] = {}
        for row in conn.execute(query):
            row_id = int(row["row_id"])
            chunk = chunks_by_row.setdefault(
                row_id,
                {
                    "row_id": row_id,
                    "embedding_id": row["embedding_id"],
                    "metadata": {},
                    "document": "",
                },
            )
            key = row["key"]
            value = metadata_value(row)
            if key == "chroma:document":
                chunk["document"] = value or ""
            else:
                chunk["metadata"][key] = value
        return list(chunks_by_row.values())
    finally:
        conn.close()


def fetch_parent_documents(parent_store_dir: Path) -> list[dict[str, Any]]:
    if not parent_store_dir.exists():
        return []

    parents: list[dict[str, Any]] = []
    for path in sorted(parent_store_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            kwargs = payload.get("kwargs", {})
            metadata = kwargs.get("metadata", {}) or {}
            page_content = kwargs.get("page_content", "") or ""
            parents.append(
                {
                    "id": path.name,
                    "metadata": metadata,
                    "page_content": page_content,
                    "source": str(metadata.get("source", "") or ""),
                    "title": metadata.get("title", ""),
                    "context_header": str(metadata.get("context_header", "") or "").strip(),
                }
            )
        except Exception as e:
            parents.append(
                {
                    "id": path.name,
                    "metadata": {},
                    "page_content": "",
                    "source": "",
                    "title": "",
                    "context_header": "",
                    "read_error": str(e)[:160],
                }
            )
    return parents


def header_flags(
    header: str,
    source_counter: Counter[str],
    sample_sources: list[str],
    sample_titles: list[str],
    sample_docs: list[str],
) -> list[str]:
    flags: list[str] = []
    lowered = header.lower()
    joined_context = "\n".join(sample_sources + sample_titles + sample_docs).lower()
    word_count = len(header.split())

    if not header:
        flags.append("missing_header")
    if word_count > 35 or len(header) > 260:
        flags.append("header_long")
    if any(pattern in lowered for pattern in GENERIC_HEADER_PATTERNS):
        flags.append("generic_header")
    if any(pattern in lowered for pattern in ASSERTIVE_PATTERNS):
        flags.append("header_contains_claim")
    if sum(source_counter.values()) >= 100:
        flags.append("very_repeated")
    elif sum(source_counter.values()) >= 30:
        flags.append("repeated")

    for label, hints in TYPE_HINTS.items():
        if label in lowered and not any(hint in joined_context for hint in hints):
            flags.append(f"possibly_wrong_type:{label}")

    if "progetto" in lowered and any("corsi.unisa.it" in src for src in sample_sources):
        flags.append("possibly_wrong_type:project_on_course_url")
    if "docente" in lowered and any("corsi.unisa.it" in src for src in sample_sources):
        flags.append("possibly_wrong_type:teacher_on_course_url")

    return sorted(set(flags))


def chunk_preview(chunk: dict[str, Any], preview_chars: int) -> dict[str, Any]:
    meta = chunk["metadata"]
    source = str(meta.get("source", ""))
    content_type = str(meta.get("content_type", ""))
    return {
        "embedding_id": chunk["embedding_id"],
        "source": source,
        "title": meta.get("title", ""),
        "is_pdf": is_pdf(source, content_type),
        "content_type": content_type,
        "chunk_preview": one_line(chunk["document"], preview_chars),
    }


def build_parent_store_summary(parents: list[dict[str, Any]], preview_chars: int) -> dict[str, Any]:
    parents_with_header = 0
    parents_with_header_in_content = 0
    parents_with_context_prefix = 0
    parents_with_raw_pdf = 0
    read_errors = 0
    domain_counter: Counter[str] = Counter()
    header_domain_counter: dict[str, Counter[str]] = defaultdict(Counter)
    raw_pdf_samples = []
    context_prefix_samples = []

    for parent in parents:
        if parent.get("read_error"):
            read_errors += 1

        content = parent.get("page_content", "") or ""
        header = parent.get("context_header", "") or ""
        source = parent.get("source", "") or ""
        domain = domain_of(source)
        domain_counter[domain] += 1

        if header:
            parents_with_header += 1
            header_domain_counter[domain][header] += 1
            if content.lstrip().startswith(header):
                parents_with_header_in_content += 1

        if content.lstrip().lower().startswith("context:"):
            parents_with_context_prefix += 1
            if len(context_prefix_samples) < 10:
                context_prefix_samples.append(
                    {
                        "parent_id": parent.get("id"),
                        "source": source,
                        "header": header,
                        "preview": one_line(content, preview_chars),
                    }
                )

        if is_raw_pdf_artifact(content):
            parents_with_raw_pdf += 1
            if len(raw_pdf_samples) < 10:
                raw_pdf_samples.append(
                    {
                        "parent_id": parent.get("id"),
                        "source": source,
                        "header": header,
                        "preview": one_line(content, preview_chars),
                    }
                )

    header_distribution_by_domain = [
        {
            "domain": domain,
            "parent_count": domain_counter[domain],
            "top_headers": [
                {"header": header, "count": count}
                for header, count in header_domain_counter[domain].most_common(20)
            ],
        }
        for domain, _ in domain_counter.most_common()
    ]

    return {
        "total_parent_documents": len(parents),
        "parents_with_context_header_metadata": parents_with_header,
        "parents_without_context_header": len(parents) - parents_with_header,
        "parents_with_header_in_content": parents_with_header_in_content,
        "parents_with_context_prefix_in_content": parents_with_context_prefix,
        "parents_with_raw_pdf_artifacts": parents_with_raw_pdf,
        "read_errors": read_errors,
        "domains": [
            {"domain": domain, "count": count}
            for domain, count in domain_counter.most_common()
        ],
        "header_distribution_by_domain": header_distribution_by_domain,
        "context_prefix_samples": context_prefix_samples,
        "raw_pdf_samples": raw_pdf_samples,
    }


def build_report(
    chunks: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    header_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domain_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    chunks_with_header_metadata = 0
    chunks_with_header_in_content = 0
    child_chunks_with_raw_pdf = 0
    empty_or_missing_headers = 0

    for chunk in chunks:
        meta = chunk["metadata"]
        header = str(meta.get("context_header", "") or "").strip()
        source = str(meta.get("source", "") or "")
        doc = chunk["document"] or ""
        if is_raw_pdf_artifact(doc):
            child_chunks_with_raw_pdf += 1

        if header:
            chunks_with_header_metadata += 1
            if doc.lstrip().startswith(header):
                chunks_with_header_in_content += 1
        else:
            empty_or_missing_headers += 1

        header_chunks[header].append(chunk)
        source_chunks[source].append(chunk)
        domain_counter[domain_of(source)] += 1
        source_counter[source] += 1

    header_details = []
    for header, grouped_chunks in header_chunks.items():
        per_source = Counter(str(c["metadata"].get("source", "") or "") for c in grouped_chunks)
        sample_chunks = grouped_chunks[: args.preview_per_header]
        sample_sources = [str(c["metadata"].get("source", "") or "") for c in sample_chunks]
        sample_titles = [str(c["metadata"].get("title", "") or "") for c in sample_chunks]
        sample_docs = [str(c.get("document", "") or "")[: args.preview_chars] for c in sample_chunks]
        flags = header_flags(header, per_source, sample_sources, sample_titles, sample_docs)
        header_details.append(
            {
                "header": header,
                "count": len(grouped_chunks),
                "source_count": len(per_source),
                "word_count": len(header.split()),
                "char_count": len(header),
                "flags": flags,
                "top_sources": [
                    {"source": source, "count": count}
                    for source, count in per_source.most_common(10)
                ],
                "sample_chunks": [
                    chunk_preview(c, args.preview_chars) for c in sample_chunks
                ],
            }
        )

    def header_sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
        return (len(item["flags"]), item["count"], item["char_count"])

    header_details.sort(key=header_sort_key, reverse=True)
    top_headers_by_count = sorted(
        header_details, key=lambda item: item["count"], reverse=True
    )[: args.top_headers]
    needs_manual_review = header_details[: args.max_review_items]

    source_details = []
    for source, grouped_chunks in source_chunks.items():
        meta0 = grouped_chunks[0]["metadata"] if grouped_chunks else {}
        content_type = str(meta0.get("content_type", ""))
        headers = Counter(
            str(c["metadata"].get("context_header", "") or "").strip()
            for c in grouped_chunks
        )
        source_details.append(
            {
                "source": source,
                "domain": domain_of(source),
                "title": meta0.get("title", ""),
                "is_pdf": is_pdf(source, content_type),
                "content_type": content_type,
                "chunk_count": len(grouped_chunks),
                "headers": [
                    {"header": header, "count": count}
                    for header, count in headers.most_common(10)
                ],
                "sample_chunks": [
                    chunk_preview(c, args.preview_chars)
                    for c in grouped_chunks[: args.preview_per_source]
                ],
            }
        )
    source_details.sort(key=lambda item: item["chunk_count"], reverse=True)

    flag_counter: Counter[str] = Counter()
    affected_chunk_counter: Counter[str] = Counter()
    for detail in header_details:
        flag_counter.update(detail["flags"])
        for flag in detail["flags"]:
            affected_chunk_counter[flag] += detail["count"]

    header_distribution_by_domain = []
    for domain, _ in domain_counter.most_common():
        headers_for_domain = Counter()
        for header, grouped_chunks in header_chunks.items():
            count = sum(
                1 for chunk in grouped_chunks
                if domain_of(str(chunk["metadata"].get("source", "") or "")) == domain
            )
            if count:
                headers_for_domain[header] += count
        header_distribution_by_domain.append(
            {
                "domain": domain,
                "child_chunk_count": domain_counter[domain],
                "top_headers": [
                    {"header": header, "count": count}
                    for header, count in headers_for_domain.most_common(20)
                ],
            }
        )

    return {
        "metadata": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "collection_name": COLLECTION_NAME,
            "chroma_db": str(args.db),
            "parent_store": str(args.parent_store),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "settings": {
                "top_headers": args.top_headers,
                "top_sources": args.top_sources,
                "preview_per_header": args.preview_per_header,
                "preview_per_source": args.preview_per_source,
                "preview_chars": args.preview_chars,
                "max_review_items": args.max_review_items,
            },
        },
        "summary": {
            "total_child_chunks": len(chunks),
            "chunks_with_context_header_metadata": chunks_with_header_metadata,
            "chunks_with_header_in_content": chunks_with_header_in_content,
            "chunks_without_context_header": empty_or_missing_headers,
            "child_chunks_with_raw_pdf_artifacts": child_chunks_with_raw_pdf,
            "unique_headers": len(header_chunks),
            "unique_sources": len(source_chunks),
            "domains": [
                {"domain": domain, "count": count}
                for domain, count in domain_counter.most_common()
            ],
            "flag_counts": [
                {"flag": flag, "count": count}
                for flag, count in flag_counter.most_common()
            ],
            "flag_affected_child_chunks": [
                {"flag": flag, "affected_child_chunks": count}
                for flag, count in affected_chunk_counter.most_common()
            ],
        },
        "parent_store": build_parent_store_summary(parents, args.preview_chars),
        "child_header_distribution_by_domain": header_distribution_by_domain,
        "top_repeated_headers": top_headers_by_count,
        "needs_manual_review": needs_manual_review,
        "top_sources": source_details[: args.top_sources],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    parent_store = report.get("parent_store", {})
    lines = [
        "# Context Header Audit",
        "",
        f"- Generated at: `{report['metadata']['generated_at']}`",
        f"- Collection: `{report['metadata']['collection_name']}`",
        f"- Total child chunks: **{summary['total_child_chunks']}**",
        f"- Chunks with header metadata: **{summary['chunks_with_context_header_metadata']}**",
        f"- Chunks with header prepended in content: **{summary['chunks_with_header_in_content']}**",
        f"- Child chunks with raw PDF artifacts: **{summary['child_chunks_with_raw_pdf_artifacts']}**",
        f"- Unique headers: **{summary['unique_headers']}**",
        f"- Unique sources: **{summary['unique_sources']}**",
        "",
        "## Parent Store",
        "",
        f"- Total parent documents: **{parent_store.get('total_parent_documents', 0)}**",
        f"- Parents with header metadata: **{parent_store.get('parents_with_context_header_metadata', 0)}**",
        f"- Parents without header metadata: **{parent_store.get('parents_without_context_header', 0)}**",
        f"- Parents with header in page_content: **{parent_store.get('parents_with_header_in_content', 0)}**",
        f"- Parents starting with `Context:`: **{parent_store.get('parents_with_context_prefix_in_content', 0)}**",
        f"- Parents with raw PDF artifacts: **{parent_store.get('parents_with_raw_pdf_artifacts', 0)}**",
        "",
        "## Flag Counts",
        "",
    ]

    if summary["flag_counts"]:
        for item in summary["flag_counts"]:
            lines.append(f"- `{item['flag']}`: {item['count']}")
    else:
        lines.append("- No heuristic flags found.")

    affected = summary.get("flag_affected_child_chunks", [])
    lines.extend(["", "## Flag-Affected Child Chunks", ""])
    if affected:
        for item in affected:
            lines.append(f"- `{item['flag']}`: {item['affected_child_chunks']}")
    else:
        lines.append("- No flagged child chunks.")

    lines.extend(["", "## Header Distribution By Domain", ""])
    for item in report.get("child_header_distribution_by_domain", [])[:10]:
        lines.append(f"### {item['domain']} ({item['child_chunk_count']} child chunks)")
        for header_item in item["top_headers"][:5]:
            lines.append(f"- {header_item['count']}: `{header_item['header']}`")
        lines.append("")

    lines.extend(["", "## Top Repeated Headers", ""])
    for item in report["top_repeated_headers"][:30]:
        flags = ", ".join(f"`{flag}`" for flag in item["flags"]) or "none"
        lines.append(f"### {item['count']} chunks, {item['source_count']} sources")
        lines.append(f"- Header: `{item['header']}`")
        lines.append(f"- Flags: {flags}")
        if item["top_sources"]:
            source = item["top_sources"][0]
            lines.append(f"- Top source: `{source['source']}` ({source['count']} chunks)")
        lines.append("")

    lines.extend(["## Manual Review Shortlist", ""])
    for item in report["needs_manual_review"][:30]:
        flags = ", ".join(f"`{flag}`" for flag in item["flags"]) or "none"
        lines.append(f"- **{item['count']} chunks** | {flags} | `{item['header']}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    chunks = fetch_chunks(args.db)
    parents = fetch_parent_documents(args.parent_store)
    report = build_report(chunks, parents, args)
    write_json(args.json_out, report)
    write_markdown(args.md_out, report)

    summary = report["summary"]
    print("Context header audit completed.")
    print(f"  Child chunks analyzed : {summary['total_child_chunks']}")
    print(f"  Parent docs analyzed  : {report['parent_store']['total_parent_documents']}")
    print(f"  Unique headers        : {summary['unique_headers']}")
    print(f"  Unique sources        : {summary['unique_sources']}")
    print(f"  JSON report           : {args.json_out}")
    print(f"  Markdown summary      : {args.md_out}")


if __name__ == "__main__":
    main()
