from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import (
    LLM_PROVIDER,
    OLLAMA_CHAT_MODEL,
    OPENROUTER_AGENT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_LIGHTWEIGHT_MODEL,
)


def _active_chat_model() -> str:
    """Return the model ID actually used by DiemBrain, respecting LLM_PROVIDER.

    Used as the cache key so that switching provider correctly invalidates
    cached responses, and as the label in summary.md.
    """
    return OPENROUTER_AGENT_MODEL if LLM_PROVIDER == "openrouter" else OLLAMA_CHAT_MODEL


def _build_judge_llm(force_json: bool = False, force_local: bool = False) -> Any:
    """Create the judge LLM using OPENROUTER_LIGHTWEIGHT_MODEL when available.

    Using a lightweight model different from OPENROUTER_AGENT_MODEL avoids
    self-confirmation bias when judging brain responses. force_json=True adds
    provider-specific JSON-mode parameters required by Ragas. force_local=True
    skips the OpenRouter branch (useful for offline development).
    """
    if LLM_PROVIDER == "openrouter" and not force_local:
        try:
            kwargs: dict[str, Any] = {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": OPENROUTER_API_KEY,
                "model": OPENROUTER_LIGHTWEIGHT_MODEL,
                "temperature": 0.0,
                "timeout": 60,
            }
            if force_json:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
                kwargs["max_tokens"] = 2048
            return ChatOpenAI(**kwargs)
        except Exception as e:
            logging.getLogger("diem.eval").warning(
                f"OpenRouter judge init failed ({e}), falling back to Ollama"
            )
    if force_json:
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.0, format="json", num_ctx=8192)
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0.0)
