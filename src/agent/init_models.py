from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.utils.logger import get_logger
from config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_CHAT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_AGENT_MODEL,
    OPENROUTER_LIGHTWEIGHT_MODEL,
)

logger = get_logger(__name__)



def build_lightweight_model():
    """Lightweight model: guardrails, rewrite, summarize, calculate — latency-sensitive tasks."""
    if LLM_PROVIDER == "openrouter":
        try:
            logger.info(f"Lightweight model: OpenRouter {OPENROUTER_LIGHTWEIGHT_MODEL}")
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_LIGHTWEIGHT_MODEL,
                temperature=LLM_TEMPERATURE,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"OpenRouter lightweight init failed ({e}), falling back to Ollama")
    logger.info(f"Lightweight model fallback: Ollama {OLLAMA_CHAT_MODEL}")
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)


def build_agent_model():
    """Routing model: tool selection and reasoning."""
    if LLM_PROVIDER == "openrouter":
        try:
            logger.info(f"Agent model: OpenRouter {OPENROUTER_AGENT_MODEL}")
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_AGENT_MODEL,
                temperature=LLM_TEMPERATURE,
                timeout=120,
            )
        except Exception as e:
            logger.warning(f"OpenRouter agent init failed ({e}), falling back to Ollama")
    logger.info(f"Agent model fallback: Ollama {OLLAMA_CHAT_MODEL}")
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=LLM_TEMPERATURE)
