"""
Preview section-level documents for the DIEM personnel page.

This script does not modify Chroma or the ingestion pipeline. It crawls only the
target page, extracts the existing structured sections, and simulates the
proposed split into one document per personnel section plus a small index doc.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.crawler import crawl, get_section_base
from src.ingestion.parser import (
    _clean_structured_sections,
    _extract_structured_sections,
    extract_html_metadata,
    html_extractor,
    html_extractor_for_source,
)


DEFAULT_URL = "https://www.diem.unisa.it/dipartimento/personale"
DEFAULT_JSON = "evaluation/results/personale_section_docs_preview.json"
DEFAULT_MD = "evaluation/results/personale_section_docs_preview.md"

HEADER_BY_TITLE = {
    "professore ordinario": "Context: personale DIEM - professori ordinari",
    "professore associato": "Context: personale DIEM - professori associati",
    "ricercatore": "Context: personale DIEM - ricercatori",
    "personale tecnico-amministrativo": "Context: personale DIEM - personale tecnico-amministrativo",
    "assegnista": "Context: personale DIEM - assegnisti",
    "dottorando": "Context: personale DIEM - dottorandi",
    "docente a contratto": "Context: personale DIEM - docenti a contratto",
    "professore emerito": "Context: personale DIEM - professori emeriti",
}


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9àèéìòù]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "sezione"


def _canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _section_header(title: str) -> str:
    return HEADER_BY_TITLE.get(_canonical(title), f"Context: personale DIEM - {title.lower()}")


def _build_section_docs(url: str, page_title: str, sections: list[dict]) -> list[dict]:
    docs = []
    for section in sections:
        title = section["title"]
        rows = section.get("rows") or []
        header = _section_header(title)
        content = "\n".join([header, "", f"{page_title} - {title}", *rows]).strip()
        docs.append(
            {
                "source": f"{url}#{_slugify(title)}",
                "section": title,
                "header": header,
                "rows_count": len(rows),
                "chars": len(content),
                "content": content,
            }
        )
    return docs


def _build_index_doc(url: str, page_title: str, section_docs: list[dict]) -> dict:
    sections = [doc["section"] for doc in section_docs]
    content = "\n".join(
        [
            "Context: personale DIEM - sezioni disponibili",
            "",
            page_title,
            "Sezioni del personale DIEM:",
            *[f"- {section}" for section in sections],
        ]
    )
    return {
        "source": f"{url}#sezioni",
        "section": "Indice sezioni",
        "header": "Context: personale DIEM - sezioni disponibili",
        "rows_count": len(sections),
        "chars": len(content),
        "content": content,
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Preview personale DIEM per sezioni",
        "",
        f"URL: {payload['url']}",
        "",
        "## Confronto",
        "",
        f"- Testo base attuale: {payload['current']['chars']} caratteri",
        f"- Testo structured attuale: {payload['structured_current']['chars']} caratteri",
        f"- Sezioni proposte: {payload['proposed']['section_docs_count']}",
        f"- Documenti proposti totali: {payload['proposed']['total_docs_count']} inclusivo di indice",
        "",
        "## Documenti Proposti",
        "",
    ]
    for doc in payload["proposed"]["documents"]:
        lines.extend(
            [
                f"### {doc['section']}",
                "",
                f"- Source: `{doc['source']}`",
                f"- Header: `{doc['header']}`",
                f"- Righe: {doc['rows_count']}",
                f"- Caratteri: {doc['chars']}",
                "",
                "```text",
                doc["content"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview DIEM personnel section-level docs")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--json-out", default=DEFAULT_JSON)
    parser.add_argument("--md-out", default=DEFAULT_MD)
    args = parser.parse_args()

    url = args.url
    docs = list(crawl(url, base_url=get_section_base(url), max_depth=1))
    doc = next((item for item in docs if item.metadata.get("source") == url), docs[0] if docs else None)
    if doc is None:
        raise SystemExit(f"No document returned for {url}")

    raw_html = doc.page_content
    metadata = {**doc.metadata, **extract_html_metadata(raw_html)}
    page_title = metadata.get("title") or "Dipartimento | Docenti e Personale"

    current_text = html_extractor(raw_html)
    structured_current_text = html_extractor_for_source(raw_html, url)
    raw_structured_text, raw_sections = _extract_structured_sections(raw_html)
    clean_sections, dropped_sections, dropped_rows = _clean_structured_sections(raw_sections)

    section_docs = _build_section_docs(url, page_title, clean_sections)
    index_doc = _build_index_doc(url, page_title, section_docs)
    proposed_docs = [index_doc, *section_docs]

    payload = {
        "url": url,
        "host": urlparse(url).netloc,
        "metadata": metadata,
        "current": {
            "chars": len(current_text),
            "preview": current_text[:2000],
        },
        "structured_current": {
            "chars": len(structured_current_text),
            "sections_found": len(raw_sections),
            "clean_sections": len(clean_sections),
            "dropped_sections": dropped_sections,
            "dropped_rows": dropped_rows,
            "preview": structured_current_text[:2000],
        },
        "raw_structured": {
            "chars": len(raw_structured_text),
            "preview": raw_structured_text[:2000],
        },
        "proposed": {
            "section_docs_count": len(section_docs),
            "total_docs_count": len(proposed_docs),
            "documents": proposed_docs,
        },
    }

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Sections: {len(section_docs)}; proposed docs: {len(proposed_docs)}")
    for item in proposed_docs:
        print(f"- {item['section']}: rows={item['rows_count']}, chars={item['chars']}")


if __name__ == "__main__":
    main()
