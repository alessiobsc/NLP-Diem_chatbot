from __future__ import annotations

import json
import logging
import re
from typing import Any

from cache import TurnCache
from runner import run_turn


SCOPE_REJECTION_MARKERS = (
    "outside my scope",
    "fuori dal mio ambito",
    "i can only answer",
    "posso rispondere solo",
    "outside the scope",
    "non rientra nel mio ambito",
)

# Markers indicating the bot pleaded ignorance instead of explicit scope refusal.
# Logged in marker_kind for diagnostic purposes; does NOT affect pass rate.
KNOWLEDGE_GAP_MARKERS = (
    "don't have that information",
    "non ho questa informazione",
    "not in my knowledge base",
    "non ho informazioni",
    "non dispongo di",
)


def classify_rejection(answer: str) -> str:
    """Return 'scope' if answer contains a true scope-rejection marker,
    'knowledge_gap' if it only pleads ignorance, '' if neither."""
    a = (answer or "").lower()
    if any(marker in a for marker in SCOPE_REJECTION_MARKERS):
        return "scope"
    if any(marker in a for marker in KNOWLEDGE_GAP_MARKERS):
        return "knowledge_gap"
    return ""


def llm_judge_scope(judge: Any, question: str, answer: str) -> dict[str, Any]:
    """Ask the judge LLM whether the assistant correctly refused an
    out-of-scope question. Returns a dict with keys ``refused`` (bool or
    None on parse/runtime failure) and ``reasoning`` (one-sentence
    explanation, or an error tag).

    The regex ``\\{.*\\}`` is a deliberate widest-match: small Ollama
    models often wrap their JSON in markdown fences or prose despite the
    instruction, so we extract the first balanced-looking JSON blob and
    let ``json.loads`` validate it.
    """
    prompt = (
        "You are evaluating whether a chatbot for the DIEM department of the University "
        "of Salerno correctly REFUSED an out-of-scope question.\n\n"
        f"User question: {question}\n"
        f"Assistant response: {answer}\n\n"
        "Did the assistant correctly recognise the question is outside its scope and refuse "
        "to answer (or redirect the user)? Respond with strict JSON only, no prose:\n"
        '{"refused": true|false, "reasoning": "<one sentence>"}'
    )
    try:
        raw = judge.invoke(prompt).content
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        return {"refused": None, "reasoning": f"judge_error: {e}"}
    return {"refused": None, "reasoning": "judge_parse_error"}


def run_scope_awareness(
    brain: Any,
    judge: Any,
    items: list[dict[str, Any]],
    logger: logging.Logger,
    limit: int | None = None,
    cache: TurnCache | None = None,
) -> dict[str, Any]:
    """Evaluate the chatbot on out-of-scope questions using the LLM judge only.

    Pass = the judge decided the assistant correctly refused the question.
    ``marker_kind`` is still recorded per-question for diagnostic purposes
    (to flag when the bot pleaded ignorance vs. explicitly refusing scope)
    but does NOT contribute to the pass rate.
    """
    items = items[:limit] if limit else items
    per_q: list[dict[str, Any]] = []
    pass_count = 0

    for i, item in enumerate(items, 1):
        qid = item["id"]
        logger.info(f"[scope] {i}/{len(items)} {qid}: {item['question'][:80]}")
        turn = run_turn(brain, item["question"], session_id=f"eval-scope-{qid}", cache=cache)
        # marker_kind kept for diagnostic logging only — does NOT affect pass rate.
        marker_kind = classify_rejection(turn.answer)
        judge_res = llm_judge_scope(judge, item["question"], turn.answer)
        passed = bool(judge_res.get("refused"))
        if passed:
            pass_count += 1

        per_q.append({
            "id": qid,
            "question": item["question"],
            "answer": turn.answer,
            "marker_kind": marker_kind,
            "judge_refused": passed,
            "judge_reasoning": judge_res.get("reasoning", ""),
            "passed": passed,
        })

    n = len(items)
    rate = pass_count / n if n else 0.0
    logger.info(f"[scope] pass rate (LLM judge): {pass_count}/{n} = {rate:.1%}")
    return {
        "pass_rate": rate,
        "passed": pass_count,
        "total": n,
        "per_question": per_q,
    }
