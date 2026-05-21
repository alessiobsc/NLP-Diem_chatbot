"""
DIEM Chatbot Evaluation Runner.

Runs the chatbot defined in `brain.py` against the golden sets under
`evaluation/dataset/` and produces a full evaluation report mapped to the
five traccia criteria. The runner mixes Ragas (for content quality) with
custom multi-turn checks (for robustness and scope awareness).

----------------------------------------------------------------------------
Mapping: traccia criterion -> concrete metric
----------------------------------------------------------------------------
    Relevance        -> Ragas ResponseRelevancy          (in_scope + multi_turn)
    Correctness      -> Ragas Faithfulness + AnswerCorrectness
                        (LLMContextPrecision/Recall as retrieval health)
                        (in_scope + multi_turn)
    Coherence        -> Ragas AspectCritic (LLM-as-judge, binary 0/1)
                        (in_scope + multi_turn)
    Robustness       -> custom multi-turn LLM judge only (no Ragas).
                        Answer must not flip under "Are you sure?",
                        false premises, jailbreaks, role overrides.
    Scope Awareness  -> LLM judge only (no Ragas, no soft_pass).
                        Pass = judge says the bot refused out-of-scope.
                        marker_kind logged for diagnostics but does not
                        affect the pass rate.

----------------------------------------------------------------------------
Requirements
----------------------------------------------------------------------------
    - A populated Chroma index at config.CHROMA_DIR
      (`python main_ingestion.py --full`).
    - An Ollama daemon running locally with at least:
        * config.OLLAMA_CHAT_MODEL (the chatbot under test)
        * the judge model (default: llama3.1:8b-instruct-q4_K_M; pull with
          `ollama pull llama3.1:8b-instruct-q4_K_M`)
      A judge model DIFFERENT from the chat model is preferred to avoid
      self-confirmation bias.

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------
    python evaluation/tester.py --lang it
    python evaluation/tester.py --lang en
    python evaluation/tester.py --lang it --limit 3
    python evaluation/tester.py --lang en --skip-ragas
    python evaluation/tester.py --lang it --categories in_scope out_of_scope
    python evaluation/tester.py --lang it --judge-model llama3.2:3b
    # Smoke test ultra-veloce: solo coherence (1 LLM call/riga)
    python evaluation/tester.py --lang it --limit 1 --ragas-metrics coherence
    # Smoke leggero: relevancy + coherence (no long decomposition)
    python evaluation/tester.py --lang it --limit 1 --ragas-metrics coherence response_relevancy
    # First run populates the chatbot cache; subsequent --cache use runs skip
    # the chatbot entirely (handy when iterating on judge model / metric set).
    python evaluation/tester.py --lang it --cache use
    python evaluation/tester.py --lang it --cache use --ragas-metrics faithfulness

----------------------------------------------------------------------------
Outputs (per run, under evaluation/results/<timestamp>_<lang>/)
----------------------------------------------------------------------------
    - run.log                run-level log (also echoed to stdout)
    - per_question.json      raw chatbot outputs: question, answer,
                             retrieved contexts, sources, errors
    - ragas_metrics.csv      per-question Ragas scores (one row per Q,
                             one column per metric). Use this for
                             distributions, worst-case analysis, and
                             grouped statistics by category.
    - ragas_metrics.json     aggregated Ragas scores + per-metric coverage
                             (valid/total) so NaN metrics are visible.
    - scope_awareness.json   per-question scope-rejection results, with
                             both strict and soft pass flags and the
                             judge's reasoning.
    - robustness.json        per-scenario robustness results, including
                             when the marker-based double-check overrode
                             a noisy LLM-judge verdict.
    - summary.md             human-readable summary mapped to the five
                             traccia criteria. Distinguishes
                             `n/a (not run)` from `NaN (judge failed)`
                             from valid scores.

----------------------------------------------------------------------------
Performance notes
----------------------------------------------------------------------------
    Use --limit N and --ragas-metrics to scale
    down during development. Ragas runs in two phases (no-reference
    metrics on all rows; reference-requiring metrics only on rows with
    a non-empty `reference`) to avoid wasted LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

# Project root must be on sys.path before any project-level import.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Silence ragas 0.4.x migration warnings: the old import paths and Langchain
# wrappers still work and remain the only viable path for local Ollama models
# (the new llm_factory targets OpenAI by default).
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"ragas(\..*)?",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"LangchainLLMWrapper is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"LangchainEmbeddingsWrapper is deprecated.*",
)

from config import LLM_TEMPERATURE  # noqa: E402
from cache import TurnCache  # noqa: E402
from llm_factory import _active_chat_model, _build_judge_llm  # noqa: E402
from runner import (  # noqa: E402
    embedding_model,
    setup_logging,
    load_brain,
    load_golden_set,
    collect_rag_rows,
    collect_multi_turn_rag_rows,
)
from scope import run_scope_awareness  # noqa: E402
from robustness import run_robustness  # noqa: E402
from ragas_runner import RAGAS_METRIC_KEYS, run_ragas  # noqa: E402
from report import write_summary  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
RESULTS_ROOT = HERE / "results"
CACHE_ROOT = HERE / "cache"  # filesystem-backed TurnCache (see cache.py)
SUPPORTED_LANGS = ("en", "it")


def main() -> None:
    """Entry point.

    Pipeline:
      1. Parse CLI flags and locate the requested golden set.
      2. Create a timestamped output folder under ``evaluation/results/``.
      3. Load the chatbot (fails fast if Chroma is missing) and the
         judge LLM (default: llama3.1:8b-instruct-q4_K_M, picked to
         avoid self-confirmation bias with the chat model).
      4. For each enabled category, invoke the chatbot on every item
         (subject to ``--limit``), persist the raw results, and stage
         Ragas rows where applicable.
      5. Run Ragas (unless ``--skip-ragas``), then write
         ``summary.md`` mapped to the five traccia criteria.

    The runner is best invoked twice: once as a smoke test
    (``--limit 1 --ragas-metrics coherence``) to validate the pipeline
    and once for real evaluation. Full runs can take many hours on
    CPU-bound Ollama setups; see the module-level Performance notes.
    """
    parser = argparse.ArgumentParser(description="DIEM Chatbot evaluation runner")
    parser.add_argument("--lang", choices=SUPPORTED_LANGS, default="it",
                        help="Which golden set to evaluate: 'it' (default) or 'en'. "
                             "Each language has its own JSON file under dataset/.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Take only the first N items per category (smoke test).")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["in_scope", "multi_turn", "out_of_scope", "robustness"],
        choices=["in_scope", "multi_turn", "out_of_scope", "robustness"],
    )
    parser.add_argument(
        "--judge-provider",
        choices=["auto", "local"],
        default="auto",
        help="Judge LLM provider: 'auto' follows LLM_PROVIDER (default), "
             "'local' forces ChatOllama regardless of LLM_PROVIDER "
             "(useful for offline development).",
    )
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip Ragas (only run scope + robustness custom checks).")
    parser.add_argument(
        "--ragas-metrics",
        nargs="+",
        default=None,
        choices=list(RAGAS_METRIC_KEYS),
        help="Subset of Ragas metrics to compute (default: all). Use a single "
             "fast metric like 'coherence' for a quick smoke test.",
    )
    parser.add_argument(
        "--cache",
        choices=["off", "use", "refresh"],
        default="off",
        help="Chatbot response cache mode (default: off). "
             "'use' reads existing entries and writes new ones, letting "
             "you re-run the tester (e.g. for a different judge or metric "
             "subset) without paying the chatbot cost again. "
             "'refresh' ignores existing entries on read but overwrites "
             "them on write, useful after changing the chat model or "
             "system prompt. Cache files live under evaluation/cache/.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=1200,
        help="Per-job timeout in seconds for Ragas LLM calls (default: 1200). "
             "Ollama on CPU can take 300-600s per call; 1200s gives 2x headroom. "
             "Raise to 1800+ on very slow hardware or with large judge models.",
    )
    parser.add_argument(
        "--ragas-workers",
        type=int,
        default=1,
        help="Parallel workers for Ragas evaluation (default: 1). "
             "Keep at 1 for local Ollama on CPU: Ollama is serial so extra "
             "workers only compete for the same CPU and slow each call down. "
             "Raise to 2-4 only when using a GPU or a remote inference endpoint.",
    )
    args = parser.parse_args()

    dataset_path = DATASET_DIR / f"golden_set_{args.lang}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden set not found: {dataset_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / f"{timestamp}_{args.lang}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Args: {vars(args)}")

    golden = load_golden_set(dataset_path)
    logger.info(f"Golden set: {dataset_path.name} (lang={args.lang})")
    logger.info(
        "Counts: "
        + ", ".join(
            f"{k}={len(v)}" for k, v in golden.items() if k != "metadata" and isinstance(v, list)
        )
    )

    brain = load_brain(logger)
    force_local = args.judge_provider == "local"
    judge = _build_judge_llm(force_local=force_local)
    judge_model_name = getattr(judge, "model", getattr(judge, "model_name", "?"))
    logger.info(f"Judge model: {judge_model_name}")

    # Cache is keyed by (chat model, temperature, session history, question).
    # See evaluation/cache.py for invalidation semantics.
    cache = TurnCache(
        cache_dir=CACHE_ROOT,
        chat_model=_active_chat_model(),
        temperature=LLM_TEMPERATURE,
        mode=args.cache,
    )
    logger.info(
        f"Cache mode: {args.cache}"
        + (f" (dir: {CACHE_ROOT})" if args.cache != "off" else "")
    )

    rag_rows: list[dict[str, Any]] = []
    raw_log: list[dict[str, Any]] = []
    scope_res: dict[str, Any] | None = None
    robust_res: dict[str, Any] | None = None

    if "in_scope" in args.categories:
        rows, raw = collect_rag_rows(
            brain, golden["in_scope"], "in_scope", logger, args.limit, cache=cache,
        )
        rag_rows += rows
        raw_log += raw

    if "multi_turn" in args.categories:
        rows, raw = collect_multi_turn_rag_rows(
            brain, golden["multi_turn"], "multi_turn", logger, args.limit, cache=cache,
        )
        rag_rows += rows
        raw_log += raw

    if "out_of_scope" in args.categories:
        scope_res = run_scope_awareness(
            brain, judge, golden["out_of_scope"], logger, args.limit, cache=cache,
        )
        with open(run_dir / "scope_awareness.json", "w", encoding="utf-8") as f:
            json.dump(scope_res, f, ensure_ascii=False, indent=2)

    if "robustness" in args.categories:
        robust_res = run_robustness(
            brain, judge, golden["robustness"], logger, args.limit, cache=cache,
        )
        with open(run_dir / "robustness.json", "w", encoding="utf-8") as f:
            json.dump(robust_res, f, ensure_ascii=False, indent=2)

    with open(run_dir / "per_question.json", "w", encoding="utf-8") as f:
        json.dump(raw_log, f, ensure_ascii=False, indent=2)

    ragas_agg: dict[str, Any] | None = None
    if not args.skip_ragas and rag_rows:
        ragas_agg = run_ragas(
            rag_rows, logger, run_dir,
            selected_metrics=args.ragas_metrics,
            ragas_timeout=args.ragas_timeout,
            ragas_workers=args.ragas_workers,
            force_local_judge=force_local,
        )
        if ragas_agg:
            with open(run_dir / "ragas_metrics.json", "w", encoding="utf-8") as f:
                json.dump(ragas_agg, f, ensure_ascii=False, indent=2)

    write_summary(
        run_dir,
        {
            "timestamp": timestamp,
            "lang": args.lang,
            "dataset": dataset_path.name,
            "chat_model": _active_chat_model(),
            "judge_model": judge_model_name,
            "embedding_model": getattr(embedding_model, "model_name", "?"),
            "categories": args.categories,
            "limit": args.limit,
        },
        ragas_agg,
        scope_res,
        robust_res,
    )

    if cache.enabled():
        logger.info(f"Cache stats: {cache.stats()}")
    logger.info(f"Done. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
