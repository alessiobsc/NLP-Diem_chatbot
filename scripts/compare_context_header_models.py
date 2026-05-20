import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import CHROMA_DIR
from src.ingestion.header_heuristic import (
    ASSERTIVE_HEADER_PATTERNS,
    GENERIC_HEADER_PATTERNS,
    build_header_context,
    extract_year_tag,
    fallback_context_header,
    normalize_context_header,
)
from src.prompts import CONTEXT_HEADER_PROMPT


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_PARENT_STORE = CHROMA_DIR / "parent_store"
BACKUP_PARENT_STORE = PROJECT_ROOT / "chroma_diem.backup_header_bad" / "parent_store"
DEFAULT_JSON_OUT = PROJECT_ROOT / "evaluation" / "results" / "header_model_comparison.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "evaluation" / "results" / "header_model_comparison.md"


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def strip_context_header(page_content: str, metadata: dict[str, Any]) -> str:
    header = metadata.get("context_header", "")
    if isinstance(header, str) and header.strip():
        stripped = page_content.lstrip()
        if stripped.startswith(header.strip()):
            page_content = stripped[len(header.strip()):].lstrip()

    stripped = page_content.lstrip()
    if stripped.lower().startswith("context:"):
        return "\n".join(stripped.splitlines()[1:]).lstrip()
    return page_content


def load_parent_document(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kwargs = payload.get("kwargs", {})
        page_content = kwargs.get("page_content", "")
        metadata = kwargs.get("metadata", {})
        if not isinstance(page_content, str) or not page_content.strip():
            return None
        if not isinstance(metadata, dict):
            metadata = {}
        page_content = strip_context_header(page_content, metadata)
        if not page_content.strip():
            return None
        return {"page_content": page_content, "metadata": metadata}
    except Exception:
        return None


def choose_parent_store(path: Path | None) -> Path:
    if path:
        return path
    if DEFAULT_PARENT_STORE.exists() and any(DEFAULT_PARENT_STORE.iterdir()):
        return DEFAULT_PARENT_STORE
    return BACKUP_PARENT_STORE


def sample_parent_docs(
    parent_store: Path,
    limit: int,
    seed: int,
    min_chars: int,
    exclude_domains: list[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    exclude = set(exclude_domains or [])
    docs = []
    for path in sorted(parent_store.iterdir()):
        if not path.is_file():
            continue
        doc = load_parent_document(path)
        if not doc or len(doc["page_content"]) < min_chars:
            continue
        if exclude:
            from urllib.parse import urlparse
            domain = urlparse(doc["metadata"].get("source", "")).netloc.lower()
            if domain in exclude:
                continue
        docs.append((path, doc))

    random.Random(seed).shuffle(docs)
    return docs[:limit]


def first_response_line(value: str) -> str:
    for line in (value or "").strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def call_openrouter(api_key: str, model: str, prompt: str, timeout: float) -> tuple[str, float]:
    start = time.time()
    response = requests.post(
        OPENROUTER_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/diem-chatbot",
            "X-Title": "DIEM Context Header Comparison",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 60,
        },
        timeout=timeout,
    )
    elapsed = time.time() - start
    response.raise_for_status()
    payload = response.json()
    return first_response_line(payload["choices"][0]["message"]["content"]), elapsed


def flag_header(header: str, source: str, title: str) -> list[str]:
    flags = []
    lowered = header.lower()
    evidence = f"{source}\n{title}".lower()

    if not header:
        flags.append("missing")
    if any(pattern in lowered for pattern in GENERIC_HEADER_PATTERNS):
        flags.append("generic_header")
    if any(pattern in lowered for pattern in ASSERTIVE_HEADER_PATTERNS):
        flags.append("header_contains_claim")
    if len(header.split()) > 18:
        flags.append("too_long")
    if "|" in header:
        flags.append("pipe_separator")
    if "progetto di ricerca" in lowered and not any(
        term in evidence for term in ("progetti", "ricerca/progetti", "finanziat")
    ):
        flags.append("possibly_wrong_research_project")
    if "servizi e contatti" in lowered and any(term in evidence for term in ("bandi", "bando", "didattica")):
        flags.append("possibly_wrong_type")

    return flags


def generate_for_doc(
    doc: dict[str, Any],
    openrouter_api_key: str,
    openrouter_model: str,
    openrouter_timeout: float,
) -> dict[str, Any]:
    page_content = doc["page_content"]
    metadata = doc["metadata"]
    source = str(metadata.get("source", ""))
    title = str(metadata.get("title", ""))
    header_context = build_header_context(page_content, source, metadata)
    prompt = CONTEXT_HEADER_PROMPT.format(text=header_context, url=source)

    year_tag = extract_year_tag(source, metadata, page_content)

    _fallback_base = normalize_context_header(
        fallback_context_header(page_content, source, metadata),
        page_content,
        source,
        metadata,
    )
    fallback = f"{_fallback_base} {year_tag}".strip() if year_tag else _fallback_base

    openrouter_raw = ""
    openrouter_header = ""
    openrouter_error = ""
    openrouter_seconds = 0.0
    try:
        openrouter_raw, openrouter_seconds = call_openrouter(
            openrouter_api_key,
            openrouter_model,
            prompt,
            openrouter_timeout,
        )
        _llm_base = normalize_context_header(openrouter_raw, page_content, source, metadata)
        openrouter_header = f"{_llm_base} {year_tag}".strip() if year_tag else _llm_base
    except Exception as exc:
        openrouter_error = str(exc)

    return {
        "source": source,
        "title": title,
        "parent_chars": len(page_content),
        "fallback_header": fallback,
        "openrouter_raw_header": openrouter_raw,
        "openrouter_header": openrouter_header,
        "openrouter_seconds": round(openrouter_seconds, 3),
        "openrouter_error": openrouter_error,
        "flags": {
            "fallback": flag_header(fallback, source, title),
            "openrouter": flag_header(openrouter_header, source, title),
        },
        "content_preview": page_content[:700],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_size": len(results), "models": {}}
    for name, header_key, seconds_key, error_key in (
        ("fallback", "fallback_header", None, None),
        ("openrouter", "openrouter_header", "openrouter_seconds", "openrouter_error"),
    ):
        headers = [r.get(header_key, "") for r in results]
        flag_counter = Counter(flag for r in results for flag in r["flags"][name])
        model_summary = {
            "unique_headers": len(set(headers)),
            "top_headers": Counter(headers).most_common(10),
            "flag_counts": dict(flag_counter),
        }
        if seconds_key:
            times = [r[seconds_key] for r in results if r[seconds_key]]
            model_summary["avg_seconds"] = round(sum(times) / len(times), 3) if times else 0
        if error_key:
            errors = [r[error_key] for r in results if r[error_key]]
            model_summary["errors"] = len(errors)
            model_summary["sample_error"] = errors[0] if errors else ""
        summary["models"][name] = model_summary
    return summary


def md_escape(value: Any) -> str:
    return clean_text(str(value)).replace("|", "\\|")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    rows = [
        "# Context Header Model Comparison",
        "",
        f"- Parent store: `{report['parent_store']}`",
        f"- Sample size: `{summary['sample_size']}`",
        f"- LLM model: `{report['openrouter_model']}`",
        "",
        "## Summary",
        "",
        "| Model | Unique headers | Avg seconds | Errors | Flags |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for model_name, model_summary in summary["models"].items():
        flags = ", ".join(f"{k}: {v}" for k, v in model_summary["flag_counts"].items()) or "-"
        rows.append(
            "| "
            f"{model_name} | "
            f"{model_summary['unique_headers']} | "
            f"{model_summary.get('avg_seconds', 0)} | "
            f"{model_summary.get('errors', 0)} | "
            f"{md_escape(flags)} |"
        )

    rows.extend([
        "",
        "## Cases",
        "",
        f"| # | Source | Title | Fallback (heuristic) | {report['openrouter_model']} | Flags |",
        "| ---: | --- | --- | --- | --- | --- |",
    ])
    for index, result in enumerate(report["results"], 1):
        flags = {
            "fallback": result["flags"]["fallback"],
            "openrouter": result["flags"]["openrouter"],
        }
        rows.append(
            "| "
            f"{index} | "
            f"{md_escape(result['source'])} | "
            f"{md_escape(result['title'])} | "
            f"{md_escape(result['fallback_header'])} | "
            f"{md_escape(result['openrouter_header'] or result['openrouter_error'])} | "
            f"{md_escape(flags)} |"
        )

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compare fallback heuristic vs OpenRouter model for context header generation.")
    parser.add_argument("--parent-store", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--openrouter-model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-timeout", type=float, default=45)
    parser.add_argument("--exclude-domains", default="", help="Comma-separated domains to exclude, e.g. docenti.unisa.it")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing in .env or environment.")

    exclude_domains = [d.strip() for d in args.exclude_domains.split(",") if d.strip()]

    parent_store = choose_parent_store(args.parent_store)
    if not parent_store.exists():
        raise SystemExit(f"Parent store not found: {parent_store}")

    samples = sample_parent_docs(parent_store, args.limit, args.seed, args.min_chars, exclude_domains)
    if not samples:
        raise SystemExit(f"No readable parent docs found in: {parent_store}")

    print(f"Parent store: {parent_store}")
    if exclude_domains:
        print(f"Excluded domains: {', '.join(exclude_domains)}")
    print(f"Comparing {len(samples)} samples: fallback (heuristic) vs {args.openrouter_model}")

    results = []
    started_at = time.time()
    for index, (path, doc) in enumerate(samples, 1):
        result = generate_for_doc(
            doc,
            openrouter_api_key=openrouter_api_key,
            openrouter_model=args.openrouter_model,
            openrouter_timeout=args.openrouter_timeout,
        )
        result["parent_file"] = path.name
        results.append(result)

        print(
            f"[{index:02d}/{len(samples)}] "
            f"{result['openrouter_seconds']:.2f}s | "
            f"fallback='{result['fallback_header']}' | "
            f"llm='{result['openrouter_header'] or result['openrouter_error'] or 'ERROR'}'"
        )

    report = {
        "parent_store": str(parent_store),
        "sample_seed": args.seed,
        "openrouter_model": args.openrouter_model,
        "total_elapsed_seconds": round(time.time() - started_at, 3),
        "summary": summarize(results),
        "results": results,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(args.md_out, report)

    print(f"JSON report: {args.json_out}")
    print(f"Markdown report: {args.md_out}")


if __name__ == "__main__":
    main()
