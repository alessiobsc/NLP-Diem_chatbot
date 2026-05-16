from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.utils.logger import get_logger
from config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_CHAT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_AGENT_MODEL,
    OPENROUTER_MODEL,
)

logger = get_logger(__name__)


def build_chat_model():
    """Generation model: final answer generation, summarize, guardrail checks."""
    if LLM_PROVIDER == "openrouter":
        try:
            logger.info(f"Chat model: OpenRouter {OPENROUTER_MODEL}")
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                model=OPENROUTER_MODEL,
                temperature=LLM_TEMPERATURE,
                timeout=60,
            )
        except Exception as e:
            logger.warning(f"OpenRouter chat init failed ({e}), falling back to Ollama")
    logger.info(f"Chat model: Ollama {OLLAMA_CHAT_MODEL}")
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
