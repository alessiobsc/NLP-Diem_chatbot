"""
Calibrate RETRIEVER_SCORE_THRESHOLD for the current embedding model.

Queries Chroma directly (no threshold) on the golden set and prints
score distributions for in_scope vs out_of_scope queries, then
suggests a threshold value.

Usage:
    venv/Scripts/python scripts/calibrate_threshold.py
    venv/Scripts/python scripts/calibrate_threshold.py --k 20 --top 10
    venv/Scripts/python scripts/calibrate_threshold.py --output evaluation/results/threshold_qwen3.json
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from config import CHROMA_DIR_NAME, COLLECTION_NAME, EMBEDDING_DIMENSION, RETRIEVER_SCORE_THRESHOLD, OPENROUTER_EMBEDDING_MODEL
from src.encoders.embedding_init import build_embedding_model


GOLDEN_SET = ROOT / "evaluation" / "dataset" / "golden_set_it.json"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)


def compute_stats(all_scores: list[float], top_scores: list[float]) -> dict:
    def s(vals):
        if not vals:
            return {}
        return {
            "n": len(vals),
            "min": round(min(vals), 4),
            "p10": round(percentile(vals, 10), 4),
            "p25": round(percentile(vals, 25), 4),
            "median": round(percentile(vals, 50), 4),
            "p75": round(percentile(vals, 75), 4),
            "p90": round(percentile(vals, 90), 4),
            "max": round(max(vals), 4),
        }
    return {"all_chunks": s(all_scores), "top1_per_query": s(top_scores)}


def print_stats(label: str, all_scores: list[float], top_scores: list[float]) -> None:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    if not all_scores:
        print("  (no results)")
        return
    for name, scores in [("All returned chunks", all_scores), ("Top-1 per query", top_scores)]:
        print(f"\n  {name} (n={len(scores)}):")
        print(f"    min    {min(scores):.4f}")
        print(f"    p10    {percentile(scores, 10):.4f}")
        print(f"    p25    {percentile(scores, 25):.4f}")
        print(f"    median {percentile(scores, 50):.4f}")
        print(f"    p75    {percentile(scores, 75):.4f}")
        print(f"    p90    {percentile(scores, 90):.4f}")
        print(f"    max    {max(scores):.4f}")


def score_histogram(scores: list[float], bins: int = 10) -> list[dict]:
    """Print histogram and return bin data for export."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        print(f"    all scores = {lo:.4f}")
        return [{"range": f"{lo:.3f}-{hi:.3f}", "count": len(scores)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for s in scores:
        idx = min(int((s - lo) / width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_width = 30
    bin_data = []
    for i, c in enumerate(counts):
        left = lo + i * width
        right = left + width
        bar = "█" * int(c / max_count * bar_width)
        print(f"    [{left:.3f}-{right:.3f}] {bar} {c}")
        bin_data.append({"range_low": round(left, 4), "range_high": round(right, 4), "count": c})
    return bin_data


def build_markdown(
    model: str,
    k: int,
    current_threshold: float,
    suggested_threshold: float,
    n_in: int,
    n_out: int,
    stats: dict,
    per_query: dict,
    note: str,
    date: str,
) -> str:
    def stats_table(s: dict) -> str:
        rows = []
        for key, label in [
            ("all_chunks", "Tutti i chunk restituiti"),
            ("top1_per_query", "Top-1 per query"),
        ]:
            d = s.get(key, {})
            if not d:
                continue
            rows.append(f"**{label}** (n={d['n']})\n")
            rows.append("| Statistica | Valore |")
            rows.append("|---|---|")
            for stat in ["min", "p10", "p25", "median", "p75", "p90", "max"]:
                rows.append(f"| {stat} | {d[stat]:.4f} |")
            rows.append("")
        return "\n".join(rows)

    def query_table(queries_data: list[dict]) -> str:
        lines = ["| # | Query | Top-1 Score |", "|---|---|---|"]
        for i, row in enumerate(queries_data, 1):
            q = row["question"][:70] + ("…" if len(row["question"]) > 70 else "")
            lines.append(f"| {i} | {q} | {row['top1']:.4f} |")
        return "\n".join(lines)

    lines = [
        f"# Calibrazione Threshold Retriever",
        f"",
        f"**Data:** {date}  ",
        f"**Modello embedding:** `{model}`  ",
        f"**Chunk per query (k):** {k}  ",
        f"**Threshold corrente:** {current_threshold}  ",
        f"**Threshold suggerito:** **{suggested_threshold}**  ",
        f"",
        f"## Query analizzate",
        f"",
        f"- In-scope: {n_in} query",
        f"- Out-of-scope: {n_out} query (gestite da scope guardrail, non arrivano al retriever)",
        f"",
        f"## Distribuzione score — In-scope",
        f"",
        stats_table(stats["in_scope"]),
        f"### Score per query (in-scope)",
        f"",
        query_table(per_query["in_scope"]),
        f"",
        f"## Distribuzione score — Out-of-scope",
        f"",
        stats_table(stats["out_of_scope"]),
        f"### Score per query (out-of-scope)",
        f"",
        query_table(per_query["out_of_scope"]),
        f"",
        f"## Conclusione",
        f"",
        f"{note}",
        f"",
        f"Valore da impostare in `.env`:",
        f"```",
        f"RETRIEVER_SCORE_THRESHOLD={suggested_threshold}",
        f"```",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="chunks per query (default 20)")
    parser.add_argument("--top", type=int, default=None, help="limit queries per category")
    parser.add_argument("--output", type=str, default=None,
                        help="base path for output files (e.g. evaluation/results/threshold_qwen3 "
                             "→ saves .json and .md)")
    args = parser.parse_args()

    data = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    in_scope_entries = data["in_scope"]
    out_of_scope_entries = data.get("out_of_scope", [])

    if args.top:
        in_scope_entries = in_scope_entries[: args.top]
        out_of_scope_entries = out_of_scope_entries[: args.top]

    in_scope_qs = [e["question"] for e in in_scope_entries]
    out_of_scope_qs = [e["question"] for e in out_of_scope_entries]

    print(f"Loading embedding model...")
    embedding_model = build_embedding_model()

    print(f"Loading Chroma ({COLLECTION_NAME})...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR_NAME,
        collection_metadata={"hnsw:space": "cosine", "dimension": EMBEDDING_DIMENSION},
    )

    print(f"\nQuerying {len(in_scope_qs)} in_scope + {len(out_of_scope_qs)} out_of_scope (k={args.k})...")
    print(f"Current RETRIEVER_SCORE_THRESHOLD = {RETRIEVER_SCORE_THRESHOLD}\n")

    raw: dict[str, dict] = {
        "in_scope": {"all": [], "top1": [], "per_query": []},
        "out_of_scope": {"all": [], "top1": [], "per_query": []},
    }

    for category, entries in [("in_scope", in_scope_entries), ("out_of_scope", out_of_scope_entries)]:
        qs = [e["question"] for e in entries]
        for i, (entry, q) in enumerate(zip(entries, qs), 1):
            hits = vectorstore.similarity_search_with_relevance_scores(q, k=args.k)
            scores = [score for _, score in hits]
            raw[category]["all"].extend(scores)
            top1 = max(scores) if scores else 0.0
            if scores:
                raw[category]["top1"].append(top1)
            raw[category]["per_query"].append({"question": q, "top1": round(top1, 4)})
            label = q[:60] + ("…" if len(q) > 60 else "")
            print(f"  [{category[0].upper()}] {i:2d}. top={top1:.4f}  {label}")

    print_stats("IN-SCOPE queries", raw["in_scope"]["all"], raw["in_scope"]["top1"])
    print("\n  Score histogram (all chunks):")
    in_bins = score_histogram(raw["in_scope"]["all"])

    print_stats("OUT-OF-SCOPE queries", raw["out_of_scope"]["all"], raw["out_of_scope"]["top1"])
    print("\n  Score histogram (all chunks):")
    out_bins = score_histogram(raw["out_of_scope"]["all"])

    in_p10 = percentile(raw["in_scope"]["top1"], 10)
    oos_median = percentile(raw["out_of_scope"]["top1"], 50)
    in_median = percentile(raw["in_scope"]["top1"], 50)

    print(f"\n{'═'*50}")
    print("  THRESHOLD SUGGESTION")
    print(f"{'═'*50}")
    print(f"  in_scope  top-1 p10    = {in_p10:.4f}")
    print(f"  in_scope  top-1 median = {in_median:.4f}")
    print(f"  out_scope top-1 median = {oos_median:.4f}")

    if oos_median > in_p10:
        suggested = round((in_p10 + oos_median) / 2, 2)
        note = (
            f"Esiste un gap tra la distribuzione in-scope (top-1 p10={in_p10:.4f}) "
            f"e out-of-scope (top-1 median={oos_median:.4f}). "
            f"Threshold suggerito = {suggested} (punto medio del gap). "
            f"Nota: le query out-of-scope sono gestite dallo scope guardrail e non raggiungono "
            f"il retriever — il threshold è calibrato principalmente per massimizzare il recall in-scope."
        )
        print(f"\n  Suggested threshold ≈ {suggested:.2f}  (midpoint of gap)")
    else:
        suggested = round(in_p10 - 0.05, 2)
        note = (
            f"Nessun gap netto tra le distribuzioni. "
            f"Threshold suggerito = {suggested} (leggermente sotto p10 in-scope={in_p10:.4f}) "
            f"per massimizzare il recall. Il cross-encoder gestisce il filtraggio del rumore."
        )
        print(f"\n  No clear gap. Suggested: {suggested:.2f} (maximize recall)")

    print(f"\n  Set in .env:  RETRIEVER_SCORE_THRESHOLD={suggested}")
    print()

    if args.output:
        base = Path(args.output)
        base.parent.mkdir(parents=True, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        stats = {
            "in_scope": compute_stats(raw["in_scope"]["all"], raw["in_scope"]["top1"]),
            "out_of_scope": compute_stats(raw["out_of_scope"]["all"], raw["out_of_scope"]["top1"]),
        }

        json_data = {
            "date": date_str,
            "embedding_model": OPENROUTER_EMBEDDING_MODEL,
            "k": args.k,
            "current_threshold": RETRIEVER_SCORE_THRESHOLD,
            "suggested_threshold": suggested,
            "n_in_scope": len(in_scope_qs),
            "n_out_of_scope": len(out_of_scope_qs),
            "stats": stats,
            "per_query": {
                "in_scope": raw["in_scope"]["per_query"],
                "out_of_scope": raw["out_of_scope"]["per_query"],
            },
            "histograms": {
                "in_scope": in_bins,
                "out_of_scope": out_bins,
            },
            "note": note,
        }

        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON saved → {json_path}")

        md = build_markdown(
            model=OPENROUTER_EMBEDDING_MODEL,
            k=args.k,
            current_threshold=RETRIEVER_SCORE_THRESHOLD,
            suggested_threshold=suggested,
            n_in=len(in_scope_qs),
            n_out=len(out_of_scope_qs),
            stats=stats,
            per_query=raw,
            note=note,
            date=date_str,
        )
        md_path = base.with_suffix(".md")
        md_path.write_text(md, encoding="utf-8")
        print(f"  Markdown saved → {md_path}")
        print()


if __name__ == "__main__":
    main()
