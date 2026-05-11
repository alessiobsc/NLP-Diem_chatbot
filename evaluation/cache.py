"""
Persistent cache for chatbot turn results.

The evaluation runner spends most of its time invoking the chatbot
(``brain.conversational_rag.invoke``) on the golden set. Each invocation
costs minutes on local Ollama setups, but the chatbot's response is a
deterministic function of (chat model, system prompt, conversation
history, current question) once temperature is fixed.

This module memoises ``TurnResult`` objects keyed by that tuple so a
second run of the tester (e.g. iterating on the judge model, the metric
selection, or the summary format) can replay the chatbot offline.

Design choices
--------------
* **Disk-backed**: cache lives in ``evaluation/cache/`` as one JSON file
  per entry. Plain JSON keeps it inspectable and diffable.
* **History-aware key**: multi-turn behaviour depends on the conversation
  history at the moment of the call, so the cache key incorporates a
  serialised view of the prior turns of the same ``session_id``.
* **Configuration-aware key**: a model upgrade or temperature change
  must invalidate previously cached responses. The key includes
  ``chat_model`` and ``temperature``.
* **Three modes**:
    - ``off``     : ignore the cache entirely (default; preserves
                    backwards compatibility).
    - ``use``     : read existing entries; write new ones.
    - ``refresh`` : ignore existing entries on read but overwrite them
                    on write, so a single ``refresh`` run rebuilds the
                    cache for the current configuration.

Invalidate manually (delete ``evaluation/cache/``) when the underlying
Chroma index or the ``DiemBrain`` system prompt change, because those
inputs are NOT tracked in the cache key.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal


CacheMode = Literal["off", "use", "refresh"]


class TurnCache:
    """Filesystem-backed cache for chatbot turn results.

    A miss on ``get`` returns None. A hit returns the dict that was
    previously stored via ``put`` (the caller is responsible for
    rebuilding any in-memory structures, e.g. injecting the cached
    answer back into the chatbot's history store so the next turn sees
    a consistent history).
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        cache_dir: Path,
        chat_model: str,
        temperature: float,
        mode: CacheMode = "off",
    ) -> None:
        self.cache_dir = cache_dir
        self.chat_model = chat_model
        self.temperature = float(temperature)
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if mode != "off":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def enabled(self) -> bool:
        return self.mode != "off"

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------
    def _key(self, session_id: str, history: list[tuple[str, str]], question: str) -> str:
        """Build a stable hash from the inputs that determine the
        chatbot's output. ``history`` is a list of (user, assistant)
        message pairs accumulated for ``session_id`` so far.
        """
        payload = {
            "schema": self.SCHEMA_VERSION,
            "chat_model": self.chat_model,
            "temperature": self.temperature,
            "session_id": session_id,
            "history": history,
            "question": question,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _path(self, key: str) -> Path:
        # 2-level prefix to keep directory listings manageable when the
        # cache grows past a few hundred entries.
        return self.cache_dir / key[:2] / f"{key}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(
        self, session_id: str, history: list[tuple[str, str]], question: str
    ) -> dict[str, Any] | None:
        """Return the cached entry or None on miss / when disabled / when
        running in refresh mode."""
        if self.mode in ("off", "refresh"):
            if self.mode == "refresh":
                self.misses += 1
            return None
        path = self._path(self._key(session_id, history, question))
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open(encoding="utf-8") as f:
                self.hits += 1
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt entry: treat as miss and let the run overwrite it.
            self.misses += 1
            return None

    def put(
        self,
        session_id: str,
        history: list[tuple[str, str]],
        question: str,
        result: dict[str, Any],
    ) -> None:
        """Write an entry. Silently no-ops when caching is disabled."""
        if self.mode == "off":
            return
        path = self._path(self._key(session_id, history, question))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self.writes += 1

    def stats(self) -> dict[str, Any]:
        """Snapshot of the cache activity for a single run."""
        return {
            "mode": self.mode,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "chat_model": self.chat_model,
            "temperature": self.temperature,
            "cache_dir": str(self.cache_dir),
        }


def serialise_history(messages: Any) -> list[tuple[str, str]]:
    """Convert a langchain ``InMemoryChatMessageHistory.messages`` list
    into the (user, assistant) pair format used by the cache key.

    Messages outside the human/AI roles are ignored. Pairs are emitted
    only when a human turn is followed by an AI turn; trailing or
    interleaved messages are dropped, since they would not have
    contributed to the chatbot output anyway.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages or []:
        role = getattr(m, "type", "") or m.__class__.__name__.lower()
        content = getattr(m, "content", "")
        if role in ("human", "humanmessage"):
            pending_user = content
        elif role in ("ai", "aimessage"):
            if pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
    return pairs


def turn_result_to_cache(result: Any) -> dict[str, Any]:
    """Serialise a TurnResult-like dataclass into a JSON-safe dict."""
    return asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
