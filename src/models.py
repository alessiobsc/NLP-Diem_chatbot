from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_CHAT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_AGENT_MODEL,
    OPENROUTER_MODEL,
)
from src.logger import get_logger

logger = get_logger(__name__)

_QWEN3_EXTRA = {"extra_body": {"thinking": False}}


def _build_chat_model():
    """9b generation model: answer tool, summarize tool, guardrail yes/no checks."""
    if LLM_PROVIDER == "openrouter":
        try:
            logger.info(f"Chat model: OpenRouter {OPENROUTER_MODEL}")
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_MODEL,
                temperature=LLM_TEMPERATURE,
                **_QWEN3_EXTRA,
            )
        except Exception as e:
            logger.warning(f"OpenRouter chat init failed ({e}), falling back to Ollama")
    logger.info(f"Chat model: Ollama {OLLAMA_CHAT_MODEL}")
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)


def _build_agent_model():
    """32b routing model: agent reasoning and tool selection."""
    if LLM_PROVIDER == "openrouter":
        try:
            logger.info(f"Agent model: OpenRouter {OPENROUTER_AGENT_MODEL}")
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_AGENT_MODEL,
                temperature=LLM_TEMPERATURE,
                **_QWEN3_EXTRA,
            )
        except Exception as e:
            logger.warning(f"OpenRouter agent init failed ({e}), falling back to Ollama")
    logger.info(f"Agent model fallback: Ollama {OLLAMA_CHAT_MODEL}")
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)