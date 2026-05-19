"""
Simulate enrichment on a random sample of parent documents from the current Chroma store.
Calls generate_context_header (real hybrid routing) and compares old vs new headers.

Modes:
  default   : random sample of --limit docs
  --stratified : sample --per-class docs from each URL-pattern bucket (ensures coverage)
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from config import CHROMA_DIR
from src.ingestion.enrichment import (
    _use_heuristic_for_url,
    generate_context_header,
)

DEFAULT_PARENT_STORE = CHROMA_DIR / "parent_store"
DEFAULT_JSON_OUT = PROJECT_ROOT / "evaluation" / "results" / "enrichment_simulation.json"

# URL-pattern buckets for stratified sampling.
# Order matters: first matching rule wins.
_PATTERN_RULES = [
    ("docenti",          lambda h, p, q: "docenti.unisa.it" in h),
    ("corsi",            lambda h, p, q: "corsi.unisa.it" in h),
    ("schede_sua",       lambda h, p, q: "__schede-sua" in p),
    ("regolamenti_cds",  lambda h, p, q: "__regolamenti-cds" in p),
    ("progetto_singolo", lambda h, p, q: "progetti-finanziati" in p and "progetto" in q),
    ("progetti_listing", lambda h, p, q: "progetti-finanziati" in p and "progetto" not in q),
    ("uploads_pdf",      lambda h, p, q: "/uploads/" in p),
    ("terza_missione",   lambda h, p, q: "/terza-missione/" in p),
    ("international",    lambda h, p, q: "/international/" in p or "/erasmus" in p),
    ("ricerca",          lambda h, p, q: "/ricerca/" in p),
    ("dipartimento",     lambda h, p, q: "/dipartimento/" in p),
]


def url_pattern(url: str) -> str:
    parsed = urlparse(url)
    h = parsed.netloc.lower()
    p = parsed.path.lower()
    q = parse_qs(parsed.query)
    for name, rule in _PATTERN_RULES:
        if rule(h, p, q):
            return name
    return "other"


def load_parent(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kwargs = payload.get("kwargs", {})
        page_content = kwargs.get("page_content", "")
        metadata = kwargs.get("metadata", {})
        if not isinstance(page_content, str) or not page_content.strip():
            return None
        return {"page_content": page_content, "metadata": metadata, "file": path.name}
    except Exception:
        return None


def strip_existing_header(page_content: str, metadata: dict) -> str:
    """Remove prepended context header from page_content if present."""
    header = metadata.get("context_header", "")
    if header:
        stripped = page_content.lstrip()
        if stripped.startswith(header.strip()):
            page_content = stripped[len(header.strip()):].lstrip()
    stripped = page_content.lstrip()
    if stripped.lower().startswith("context:"):
        return "\n".join(stripped.splitlines()[1:]).lstrip()
    return page_content


def _load_all(store: Path, min_chars: int, exclude_domains: list[str]) -> list[dict]:
    docs = []
    for path in sorted(store.iterdir()):
        if not path.is_file():
            continue
        doc = load_parent(path)
        if not doc or len(doc["page_content"]) < min_chars:
            continue
        if exclude_domains and domain_of(doc["metadata"].get("source", "")) in exclude_domains:
            continue
        docs.append(doc)
    return docs


def sample_parents(store: Path, limit: int, seed: int, min_chars: int, exclude_domains: list[str] | None = None) -> list[dict]:
    docs = _load_all(store, min_chars, exclude_domains or [])
    random.Random(seed).shuffle(docs)
    return docs[:limit]


def stratified_sample(store: Path, per_class: int, seed: int, min_chars: int, exclude_domains: list[str] | None = None) -> list[dict]:
    """Sample per_class docs from each URL-pattern bucket."""
    docs = _load_all(store, min_chars, exclude_domains or [])
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        bucket = url_pattern(doc["metadata"].get("source", ""))
        buckets[bucket].append(doc)
    samples = []
    for name in [r[0] for r in _PATTERN_RULES] + ["other"]:
        pool = buckets.get(name, [])
        if not pool:
            continue
        chosen = rng.sample(pool, min(per_class, len(pool)))
        for doc in chosen:
            doc["_pattern"] = name
        samples.extend(chosen)
        print(f"  bucket={name:20s} pool={len(pool):4d}  sampled={len(chosen)}")
    return samples


def domain_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate enrichment on a sample of parent documents.")
    parser.add_argument("--parent-store", type=Path, default=DEFAULT_PARENT_STORE)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--exclude-domains", type=str, default="", help="Comma-separated domains to exclude")
    parser.add_argument("--stratified", action="store_true", help="Stratified sampling by URL pattern instead of random")
    parser.add_argument("--per-class", type=int, default=15, help="Docs per bucket in --stratified mode")
    args = parser.parse_args()

    exclude_domains = [d.strip() for d in args.exclude_domains.split(",") if d.strip()] if args.exclude_domains else []

    if not args.parent_store.exists():
        raise SystemExit(f"Parent store not found: {args.parent_store}")

    print(f"Parent store: {args.parent_store}")
    if exclude_domains:
        print(f"Excluding domains: {exclude_domains}")

    if args.stratified:
        print(f"Mode: stratified ({args.per_class} per bucket, seed={args.seed})")
        samples = stratified_sample(args.parent_store, args.per_class, args.seed, args.min_chars, exclude_domains)
    else:
        samples = sample_parents(args.parent_store, args.limit, args.seed, args.min_chars, exclude_domains)
    print(f"Sampled {len(samples)} parent documents")

    results = []
    llm_calls = 0
    heuristic_calls = 0
    changed = 0
    started = time.time()

    for i, doc in enumerate(samples, 1):
        metadata = doc["metadata"]
        source = metadata.get("source", "")
        old_header = metadata.get("context_header", "")
        text = strip_existing_header(doc["page_content"], metadata)

        routing = "heuristic" if _use_heuristic_for_url(source) else "llm"
        if routing == "llm":
            llm_calls += 1
        else:
            heuristic_calls += 1

        t0 = time.time()
        new_header = generate_context_header(text, source, metadata)
        elapsed = round(time.time() - t0, 2)

        is_changed = new_header.strip() != old_header.strip()
        if is_changed:
            changed += 1

        simulated_chunk = f"{new_header}\n{text[:args.preview_chars]}"

        print(
            f"[{i:03d}/{len(samples)}] {elapsed:.2f}s [{routing}] "
            f"domain={domain_of(source)}\n"
            f"  old: {old_header!r}\n"
            f"  new: {new_header!r}"
            + (" [CHANGED]" if is_changed else "")
        )

        results.append({
            "source": source,
            "domain": domain_of(source),
            "routing": routing,
            "old_header": old_header,
            "new_header": new_header,
            "changed": is_changed,
            "elapsed_s": elapsed,
            "simulated_chunk_preview": simulated_chunk,
        })

    total = round(time.time() - started, 1)
    summary = {
        "sample_size": len(samples),
        "llm_calls": llm_calls,
        "heuristic_calls": heuristic_calls,
        "changed": changed,
        "unchanged": len(samples) - changed,
        "total_seconds": total,
    }

    print(f"\n--- Summary ---")
    print(f"Total: {len(samples)} | LLM: {llm_calls} | Heuristic: {heuristic_calls}")
    print(f"Changed: {changed} / {len(samples)} | Total time: {total}s")

    report = {"summary": summary, "results": results}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON: {args.json_out}")


if __name__ == "__main__":
    main()