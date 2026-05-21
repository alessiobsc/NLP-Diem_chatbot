from __future__ import annotations

from pathlib import Path
from typing import Any


def write_summary(
    run_dir: Path,
    config: dict[str, Any],
    ragas_agg: dict[str, Any] | None,
    scope_res: dict[str, Any] | None,
    robust_res: dict[str, Any] | None,
) -> None:
    """Render the human-readable ``summary.md`` report by combining the
    Ragas aggregates with the custom scope and robustness results.

    The report deliberately distinguishes three states for each Ragas
    metric:
      - ``n/a (not run)``       : metric was excluded via --ragas-metrics
      - ``NaN (judge failed)``  : metric was run but every row failed
                                  parsing (LLM judge issue)
      - ``0.823 (5/6 valid)``   : valid score with coverage
    so that "skipped on purpose" never looks the same as "broken".
    """
    lines: list[str] = []
    lines.append("# DIEM Chatbot Evaluation Summary")
    lines.append("")
    lines.append(f"- Timestamp: {config['timestamp']}")
    lines.append(f"- Language: **{config['lang']}** (`{config['dataset']}`)")
    lines.append(f"- Chat model: `{config['chat_model']}`")
    lines.append(f"- Judge model: `{config['judge_model']}`")
    lines.append(f"- Embedding model: `{config['embedding_model']}`")
    lines.append(f"- Categories: {', '.join(config['categories'])}")
    lines.append(f"- Limit per category: {config['limit'] or 'none'}")
    lines.append("")

    lines.append("## Results by Traccia Criterion")
    lines.append("")

    if ragas_agg and ragas_agg.get("aggregated"):
        agg = ragas_agg["aggregated"]
        cov = ragas_agg.get("coverage", {})
        n = ragas_agg.get("n_rows", "?")

        def fmt(metric_key: str) -> str:
            """Format a metric with valid/total coverage. Distinguishes NaN
            (judge failure) from missing key (metric not run)."""
            if metric_key not in agg:
                return "n/a (not run)"
            v = agg[metric_key]
            c = cov.get(metric_key, {})
            valid = c.get("valid", 0)
            total = c.get("total", 0)
            if not isinstance(v, (int, float)) or v != v:  # NaN check
                return f"NaN (0/{total} valid - judge failed)"
            return f"{v:.3f} ({valid}/{total} valid)"

        lines.append(f"### Relevance ({n} questions)")
        lines.append(f"- ResponseRelevancy: **{fmt('answer_relevancy')}**")
        lines.append("")
        lines.append(f"### Correctness ({n} questions)")
        lines.append(f"- Faithfulness: **{fmt('faithfulness')}**")
        lines.append(f"- AnswerCorrectness: **{fmt('answer_correctness')}**")
        lines.append(f"- LLMContextPrecision: **{fmt('llm_context_precision_with_reference')}**")
        lines.append(f"- LLMContextRecall: **{fmt('context_recall')}**")
        lines.append("")
        lines.append(f"### Coherence ({n} questions)")
        lines.append(f"- AspectCritic / coherence: **{fmt('coherence')}**")
        lines.append("")
    else:
        lines.append("### Relevance / Correctness / Coherence")
        lines.append("- Ragas evaluation skipped or failed. See run.log.")
        lines.append("")

    if scope_res:
        lines.append(f"### Scope Awareness ({scope_res['total']} questions)")
        lines.append(
            f"- Pass rate (LLM judge): **{scope_res['pass_rate']:.1%}** "
            f"({scope_res['passed']}/{scope_res['total']})"
        )
        lines.append("")
    else:
        lines.append("### Scope Awareness")
        lines.append("- Skipped.")
        lines.append("")

    if robust_res:
        lines.append(f"### Robustness ({robust_res['total']} scenarios)")
        lines.append(
            f"- Pass rate: **{robust_res['pass_rate']:.1%}** "
            f"({robust_res['passed']}/{robust_res['total']})"
        )
        lines.append("")
    else:
        lines.append("### Robustness")
        lines.append("- Skipped.")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("- `run.log` - run-level log")
    lines.append("- `per_question.json` - raw chatbot outputs (questions, answers, retrieved contexts)")
    lines.append("- `ragas_metrics.csv` - per-question Ragas scores")
    lines.append("- `ragas_metrics.json` - aggregated Ragas scores")
    lines.append("- `scope_awareness.json` - per-question scope-rejection results")
    lines.append("- `robustness.json` - per-scenario robustness results")
    lines.append("")

    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
