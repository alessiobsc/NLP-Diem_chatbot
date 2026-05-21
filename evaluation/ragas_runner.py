from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from typing import Any

from llm_factory import _build_judge_llm
from runner import embedding_model


# CLI-friendly metric keys exposed via --ragas-metrics. The ragas column
# names produced in the result DataFrame are slightly different (e.g.
# "answer_relevancy" vs the CLI key "response_relevancy" used by users).
# The mapping is built inside run_ragas() via metric_registry.
RAGAS_METRIC_KEYS = (
    "coherence",
    "response_relevancy",
    "faithfulness",
    "answer_correctness",
    "context_precision",
    "context_recall",
)

# Per-column truncation budgets for the human-legible CSV. Long strings get
# cropped, lists get joined with a separator. Tuned so a typical row fits
# comfortably in a spreadsheet cell while remaining diagnostic.
_CSV_TEXT_LIMITS = {
    "user_input": 400,
    "response": 500,
    "reference": 400,
    "retrieved_contexts": 600,
}


def _flatten_for_csv(value: Any, max_len: int) -> str:
    """Collapse a cell value into a single-line, length-capped string.

    Lists of strings (e.g. ``retrieved_contexts``) become ``"[1] ... | [2]
    ... | [3] ..."``. Newlines and tabs are replaced with spaces. Strings
    longer than ``max_len`` are truncated with an ellipsis. Non-string,
    non-list values pass through unchanged so numeric metric columns are
    preserved.
    """
    if isinstance(value, list):
        parts = []
        for i, item in enumerate(value, 1):
            s = str(item).replace("\n", " ").replace("\r", " ").replace("\t", " ")
            parts.append(f"[{i}] {s}")
        flat = " | ".join(parts)
    elif isinstance(value, str):
        flat = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    else:
        return value  # leave numbers / NaN / bools as-is
    flat = re.sub(r"\s{2,}", " ", flat).strip()
    if len(flat) > max_len:
        flat = flat[: max_len - 1] + "…"
    return flat


def _prettify_ragas_dataframe(df: Any) -> Any:
    """Return a copy of ``df`` with text columns flattened to one line and
    truncated. Numeric metric columns are left untouched so they remain
    sortable / aggregatable in spreadsheets."""
    pretty = df.copy()
    for col, limit in _CSV_TEXT_LIMITS.items():
        if col in pretty.columns:
            pretty[col] = pretty[col].apply(lambda v, _lim=limit: _flatten_for_csv(v, _lim))
    return pretty


def run_ragas(
    rag_rows: list[dict[str, Any]],
    logger: logging.Logger,
    run_dir: Path,
    selected_metrics: list[str] | None = None,
    ragas_timeout: int = 1200,
    ragas_workers: int = 1,
    force_local_judge: bool = False,
) -> dict[str, Any] | None:
    """Run the selected Ragas metrics on ``rag_rows`` and write
    ``ragas_metrics.csv`` to ``run_dir``. Returns a dict shaped as
    ``{"aggregated": {<col>: float}, "coverage": {<col>: {valid, total}},
    "n_rows": int}`` or None if the entire evaluation crashed.

    Metrics are split in two groups by reference requirement:
      - Reference-free metrics (coherence, response_relevancy,
        faithfulness) run on ALL rows.
      - Reference-requiring metrics (answer_correctness,
        context_precision, context_recall) run only on rows whose
        ``reference`` field is non-empty. Running them on rows with an
        empty reference produces silent NaN, so we filter explicitly.

    The two phases are evaluated separately and their result frames are
    merged on (user_input, response). The CSV is the per-question audit
    trail; the returned dict feeds the summary.
    """
    if not rag_rows:
        logger.warning("[ragas] no rows to evaluate, skipping")
        return None

    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AspectCritic,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as e:
        logger.error(f"[ragas] import failed, skipping ragas eval: {e}")
        return None

    # force_json=True: Ragas requires structured JSON output; provider-specific
    # parameters are set inside _build_judge_llm.
    judge_llm = _build_judge_llm(force_json=True, force_local=force_local_judge)
    logger.info(f"[ragas] judge model: {getattr(judge_llm, 'model', getattr(judge_llm, 'model_name', '?'))}")
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_emb = LangchainEmbeddingsWrapper(embedding_model)

    coherence = AspectCritic(
        name="coherence",
        definition=(
            "Return 1 if the response is logically coherent, internally consistent, "
            "and free of contradictions or non-sequiturs. Return 0 otherwise."
        ),
        llm=ragas_llm,
    )

    # Map of metric key (CLI-friendly) -> (instance, requires_reference)
    metric_registry: dict[str, tuple[Any, bool]] = {
        "coherence": (coherence, False),
        "response_relevancy": (ResponseRelevancy(), False),
        "faithfulness": (Faithfulness(), False),
        "answer_correctness": (AnswerCorrectness(), True),
        "context_precision": (LLMContextPrecisionWithReference(), True),
        "context_recall": (LLMContextRecall(), True),
    }

    # Subset selection: validate keys, default to all
    if selected_metrics:
        unknown = [k for k in selected_metrics if k not in metric_registry]
        if unknown:
            logger.error(f"[ragas] unknown metric keys: {unknown}; valid: {list(metric_registry)}")
            return None
        chosen_keys = [k for k in metric_registry if k in selected_metrics]
    else:
        chosen_keys = list(metric_registry)
    logger.info(f"[ragas] selected metrics: {chosen_keys}")

    # Metrics partitioned by reference requirement.
    # AnswerCorrectness / LLMContextPrecisionWithReference / LLMContextRecall
    # need a non-empty reference; running them on rows with empty reference
    # produces NaN, so we filter them out per-row.
    metrics_no_ref = [metric_registry[k][0] for k in chosen_keys if not metric_registry[k][1]]
    metrics_with_ref = [metric_registry[k][0] for k in chosen_keys if metric_registry[k][1]]

    rows_with_ref = [r for r in rag_rows if (r.get("reference") or "").strip()]
    rows_no_ref = [r for r in rag_rows if not (r.get("reference") or "").strip()]
    logger.info(
        f"[ragas] split: {len(rows_with_ref)} rows with reference, "
        f"{len(rows_no_ref)} rows without reference"
    )

    # Ollama on CPU is serial: only one LLM call runs at a time regardless of
    # max_workers. With max_workers>1, Ragas submits N concurrent jobs that
    # compete for the same CPU, making each call slower and more likely to
    # timeout. max_workers=1 serialises the jobs and gives each call the full
    # CPU budget. Raise via --ragas-workers only if running on GPU or with a
    # remote inference endpoint that genuinely handles parallel requests.
    # max_retries=1: a single timeout already costs `ragas_timeout` seconds;
    # retrying 3x triples that cost with no benefit on a loaded CPU.
    run_config = RunConfig(
        timeout=ragas_timeout,
        max_workers=ragas_workers,
        max_retries=1,
        max_wait=60,
    )
    logger.info(
        f"[ragas] RunConfig: timeout={ragas_timeout}s, workers={ragas_workers}, "
        f"max_retries=1"
    )

    def _evaluate_safe(rows: list[dict[str, Any]], metrics: list[Any], label: str):
        """Wrapper around ragas.evaluate that logs failures with traceback
        instead of swallowing them. Returns None on exception so the
        caller can decide whether to continue with partial results."""
        if not rows or not metrics:
            return None
        ds = EvaluationDataset.from_list(rows)
        logger.info(f"[ragas:{label}] evaluating {len(rows)} rows x {len(metrics)} metrics")
        try:
            return evaluate(
                dataset=ds,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=True,
            )
        except Exception as e:
            logger.error(f"[ragas:{label}] evaluate() crashed: {e}")
            logger.error(traceback.format_exc())
            return None

    # Always run no-reference metrics over ALL rows; reference-requiring metrics
    # only over rows that have a reference.
    full_result = _evaluate_safe(rag_rows, metrics_no_ref, "no_ref")
    ref_result = _evaluate_safe(rows_with_ref, metrics_with_ref, "with_ref")

    frames = []
    if full_result is not None:
        try:
            frames.append(full_result.to_pandas())
        except Exception as e:
            logger.error(f"[ragas] to_pandas (no_ref) failed: {e}")
    if ref_result is not None:
        try:
            frames.append(ref_result.to_pandas())
        except Exception as e:
            logger.error(f"[ragas] to_pandas (with_ref) failed: {e}")

    if not frames:
        logger.error("[ragas] no result frames produced")
        return None

    # Merge on the row identity columns (user_input/response). Both frames share
    # those columns; metrics columns are disjoint between the two runs.
    join_cols = ["user_input", "response"]
    df = frames[0]
    for extra in frames[1:]:
        keep = [c for c in extra.columns if c in join_cols or c not in df.columns]
        df = df.merge(extra[keep], on=join_cols, how="left")

    # Write the CSV in a human-legible form: every text field has its newlines
    # replaced and is truncated to a manageable length. Without this the file
    # is technically valid CSV but unusable in plain editors because each cell
    # spans many physical lines. The full raw data remains available in
    # per_question.json for whoever needs the originals.
    df_pretty = _prettify_ragas_dataframe(df)
    df_pretty.to_csv(run_dir / "ragas_metrics.csv", index=False)

    # Aggregate (mean, skipping NaN) over numeric metric columns and report
    # per-metric coverage so NaN-only metrics are visible.
    non_metric_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    agg: dict[str, float] = {}
    coverage: dict[str, dict[str, int]] = {}
    for col in df.columns:
        if col in non_metric_cols or df[col].dtype.kind not in "fiu":
            continue
        n_total = len(df)
        n_valid = int(df[col].notna().sum())
        agg[col] = float(df[col].mean(skipna=True)) if n_valid else float("nan")
        coverage[col] = {"valid": n_valid, "total": n_total}
    logger.info(f"[ragas] aggregated metrics: {agg}")
    logger.info(f"[ragas] coverage: {coverage}")
    return {"aggregated": agg, "coverage": coverage, "n_rows": len(df)}
