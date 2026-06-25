"""
Re-run selected RAGAS metrics on an existing per_question.json without
re-invoking the chatbot. Useful to recover from API-error-induced NaN/0
in a previous evaluation run.

Usage (from the evaluation/ directory):
    python rerun_ragas.py --run-dir results/20260625_192500_it --metrics coherence faithfulness
    python rerun_ragas.py --run-dir results/20260625_192500_it  # all metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make evaluation/ and project root importable
_eval_dir = Path(__file__).parent
_project_root = _eval_dir.parent
sys.path.insert(0, str(_eval_dir))
sys.path.insert(0, str(_project_root))

from runner import embedding_model, setup_logging  # noqa: E402
from ragas_runner import run_ragas, RAGAS_METRIC_KEYS  # noqa: E402


def build_rag_rows(per_question: list[dict]) -> list[dict]:
    """Convert per_question.json entries to the format expected by run_ragas."""
    rows = []
    for item in per_question:
        if item.get("error"):
            continue
        answer = item.get("answer", "") or ""
        if not answer.strip():
            continue
        rows.append({
            "user_input": item["question"],
            "retrieved_contexts": item.get("contexts") or [],
            "response": answer,
            "reference": item.get("reference") or "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run RAGAS on existing per_question.json")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the evaluation run directory (contains per_question.json)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(RAGAS_METRIC_KEYS),
        choices=list(RAGAS_METRIC_KEYS),
        help="RAGAS metrics to compute (default: all)",
    )
    parser.add_argument(
        "--out-csv",
        default="ragas_metrics_rerun.csv",
        help="Output CSV filename inside run-dir (default: ragas_metrics_rerun.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="RAGAS max_workers (default: 1)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).parent / run_dir
    run_dir = run_dir.resolve()

    pq_path = run_dir / "per_question.json"
    if not pq_path.exists():
        print(f"ERROR: {pq_path} not found", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(run_dir)
    logger.info(f"[rerun_ragas] run_dir={run_dir}")
    logger.info(f"[rerun_ragas] metrics={args.metrics}")

    with pq_path.open(encoding="utf-8") as f:
        per_question = json.load(f)
    logger.info(f"[rerun_ragas] loaded {len(per_question)} entries from per_question.json")

    rag_rows = build_rag_rows(per_question)
    logger.info(f"[rerun_ragas] {len(rag_rows)} rows eligible for RAGAS")

    # Temporarily rename the output CSV so run_ragas writes to args.out_csv
    original_csv_name = "ragas_metrics.csv"
    out_path = run_dir / args.out_csv
    # run_ragas always writes to run_dir / "ragas_metrics.csv"; rename after
    result = run_ragas(
        rag_rows=rag_rows,
        logger=logger,
        run_dir=run_dir,
        selected_metrics=args.metrics,
        ragas_workers=args.workers,
    )

    # Rename the output file if a custom name was requested
    default_out = run_dir / original_csv_name
    if args.out_csv != original_csv_name and default_out.exists():
        default_out.rename(out_path)
        logger.info(f"[rerun_ragas] output saved to {out_path}")

    if result:
        logger.info("[rerun_ragas] aggregated results:")
        for k, v in result.get("aggregated", {}).items():
            cov = result.get("coverage", {}).get(k, {})
            logger.info(f"  {k}: {v:.3f} (valid={cov.get('valid','?')}/{cov.get('total','?')})")
    else:
        logger.error("[rerun_ragas] RAGAS returned no results")
        sys.exit(1)


if __name__ == "__main__":
    main()
