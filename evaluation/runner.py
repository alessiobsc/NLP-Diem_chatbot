from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from src.agent.brain import DiemBrain
from src.rag_hybrid import QdrantRAG
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    OPENROUTER_API_KEY
)
from evaluation.cache import TurnCache, serialise_history

def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure a dedicated logger that writes both to ``run_dir/run.log``
    and to stdout, with HH:MM:SS timestamps.

    A dedicated namespaced logger (``diem.eval``) avoids polluting the root
    logger and isolates this run's handlers across repeated invocations
    (handlers are cleared at the start of each call).
    """
    log_path = run_dir / "run.log"
    logger = logging.getLogger("diem.eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def load_brain(logger: logging.Logger) -> DiemBrain:
    """Open the persisted Qdrant index and instantiate the production
    chatbot exactly as the live app would.
    """
    try:
        hybrid_rag = QdrantRAG(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            openrouter_api_key=OPENROUTER_API_KEY
        )
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        # Could potentially query the collection count here
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
            f"Ensure Qdrant is running and populated. Error: {e}"
        )
    return DiemBrain(hybrid_rag)


def load_golden_set(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON golden set into a dict. Schema is validated
    indirectly by the consumers (collect_rag_rows, run_scope_awareness,
    run_robustness)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TurnResult:
    """Outcome of a single chatbot turn.

    Attributes
    ----------
    question : the user's question as fed to the chatbot
    answer   : the assistant's textual response (empty on error)
    contexts : the page_content of every document retrieved by the RAG
               pipeline. Used as `retrieved_contexts` for Ragas metrics.
    sources  : deduplicated list of source URLs present in the retrieved
               documents' metadata. Used for traceability in raw logs.
    error    : "{ExceptionClass}: {message}" if the invocation raised,
               otherwise None. A populated `error` always coexists with an
               empty `answer`.
    """
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: str | None = None


def run_turn(
    brain: DiemBrain,
    question: str,
    session_id: str,
    cache: TurnCache | None = None,
) -> TurnResult:
    """Invoke the conversational RAG once and capture its full output.

    When ``cache`` is provided and enabled, the call may be served from
    disk instead of hitting the chatbot. On a cache hit, the chatbot's
    in-memory history for ``session_id`` is back-filled with a synthetic
    (user, assistant) pair so that subsequent turns observe the same
    conversation state they would have seen on a real call.

    Errors are NOT raised: they are caught and surfaced in
    ``TurnResult.error`` so that one bad question does not abort the
    whole run. Downstream consumers can treat empty answers as failures
    without special-casing exceptions.
    """
    history_messages = brain.get_history(session_id)
    history_pairs = serialise_history(history_messages)

    # 1) Cache lookup. A hit short-circuits the LLM call entirely.
    if cache is not None and cache.enabled():
        cached = cache.get(session_id, history_pairs, question)
        if cached is not None:
            # Note: MemorySaver history re-hydration is skipped for cache hits.
            # Single-turn cache works correctly; multi-turn cache hits may lack
            # prior context — acceptable for development iteration.
            return TurnResult(
                question=cached.get("question", question),
                answer=cached.get("answer", "") or "",
                contexts=list(cached.get("contexts") or []),
                sources=list(cached.get("sources") or []),
                error=cached.get("error"),
            )

    # 2) Live invocation.
    try:
        result = brain.chat_eval(question, session_id)
        answer = result.get("answer", "") or ""
        docs: list[Document] = result.get("sources", []) or []
        contexts = [d.page_content for d in docs]
        sources = list({d.metadata.get("source", "") for d in docs if d.metadata.get("source")})
        if result.get("error"):
            turn = TurnResult(question=question, answer="", error=result["error"])
        else:
            turn = TurnResult(question=question, answer=answer, contexts=contexts, sources=sources)
    except Exception as e:
        turn = TurnResult(question=question, answer="", error=f"{type(e).__name__}: {e}")

    # 3) Persist the result. Errors are also cached: re-running them
    # would just hit the same failure mode given identical inputs, so
    # short-circuiting saves time without hiding diagnostic value (the
    # error string is preserved in the cached entry).
    if cache is not None and cache.enabled():
        cache.put(session_id, history_pairs, question, asdict(turn))

    return turn


def collect_rag_rows(
    brain: DiemBrain,
    items: list[dict[str, Any]],
    category: str,
    logger: logging.Logger,
    limit: int | None = None,
    cache: TurnCache | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the chatbot once per single-turn item and produce two parallel
    lists:

    - ``ragas_rows``: minimal dicts in the schema Ragas expects
      (``user_input``/``retrieved_contexts``/``response``/``reference``).
    - ``raw_log``: full per-question record (id, tags, language, full
      TurnResult, reference) for traceability and post-hoc inspection.

    Each item gets its own session_id so the chatbot's history store does
    not bleed context between unrelated questions. ``retrieved_contexts``
    falls back to ``[""]`` when retrieval returned nothing because Ragas
    rejects empty lists for that field.
    """
    items = items[:limit] if limit else items
    ragas_rows: list[dict[str, Any]] = []
    raw_log: list[dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        qid = item["id"]
        session_id = f"eval-{category}-{qid}"
        logger.info(f"[{category}] {i}/{len(items)} {qid}: {item['question'][:80]}")
        turn = run_turn(brain, item["question"], session_id, cache=cache)

        ragas_rows.append({
            "user_input": item["question"],
            "retrieved_contexts": turn.contexts or [""],
            "response": turn.answer,
            "reference": item.get("reference", ""),
        })
        raw_log.append({
            "id": qid,
            "category": category,
            "language": item.get("language", "en"),
            "tags": item.get("tags", []),
            **asdict(turn),
            "reference": item.get("reference", ""),
        })

    return ragas_rows, raw_log


def collect_multi_turn_rag_rows(
    brain: DiemBrain,
    items: list[dict[str, Any]],
    category: str,
    logger: logging.Logger,
    limit: int | None = None,
    cache: TurnCache | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Like ``collect_rag_rows`` but for items with multiple turns.

    Every turn is fed into the chatbot in order (sharing the same
    session_id so the bot's history is preserved); only the final turn
    contributes a row to ``ragas_rows``. The ``raw_log`` records ALL
    turns so reviewers can diagnose breakdowns mid-conversation.
    """
    items = items[:limit] if limit else items
    ragas_rows: list[dict[str, Any]] = []
    raw_log: list[dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        qid = item["id"]
        session_id = f"eval-{category}-{qid}"
        logger.info(f"[{category}] {i}/{len(items)} {qid} ({len(item['turns'])} turns)")

        turns: list[TurnResult] = []
        for t in item["turns"]:
            tr = run_turn(brain, t["question"], session_id, cache=cache)
            turns.append(tr)

        final = turns[-1]
        # Build a self-contained user_input that includes prior conversational
        # context. Without this, ragas evaluates the final question in isolation
        # ("Quali sono i suoi orari di ricevimento?") with no antecedent for the
        # pronoun, which makes ResponseRelevancy / AnswerCorrectness meaningless.
        if len(turns) > 1:
            history_lines = []
            for prev_q, prev_t in zip(item["turns"][:-1], turns[:-1]):
                history_lines.append(f"USER: {prev_q['question']}")
                # Truncate long previous answers to keep the prompt manageable.
                prev_ans = (prev_t.answer or "").strip().replace("\n", " ")
                if len(prev_ans) > 400:
                    prev_ans = prev_ans[:400] + "..."
                history_lines.append(f"ASSISTANT: {prev_ans}")
            history_lines.append(f"USER: {item['turns'][-1]['question']}")
            combined_user_input = "\n".join(history_lines)
        else:
            combined_user_input = item["turns"][-1]["question"]

        ragas_rows.append({
            "user_input": combined_user_input,
            "retrieved_contexts": final.contexts or [""],
            "response": final.answer,
            "reference": item.get("reference", ""),
        })
        raw_log.append({
            "id": qid,
            "category": category,
            "tags": [item.get("tag", "")],
            "turns": [asdict(t) for t in turns],
            "reference": item.get("reference", ""),
        })

    return ragas_rows, raw_log
