from unittest.mock import patch, MagicMock
import pytest


def test_build_chat_model_openrouter():
    with patch("src.models.LLM_PROVIDER", "openrouter"), \
         patch("src.models.OPENROUTER_API_KEY", "test-key"), \
         patch("src.models.OPENROUTER_MODEL", "qwen/qwen3.5-9b"), \
         patch("src.models.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from src.models import _build_chat_model
        model = _build_chat_model()
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["model"] == "qwen/qwen3.5-9b"
        assert "thinking" in call_kwargs.get("extra_body", {})
        assert call_kwargs["extra_body"]["thinking"] is False


def test_build_agent_model_openrouter():
    with patch("src.models.LLM_PROVIDER", "openrouter"), \
         patch("src.models.OPENROUTER_API_KEY", "test-key"), \
         patch("src.models.OPENROUTER_AGENT_MODEL", "qwen/qwen3-32b"), \
         patch("src.models.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from src.models import _build_agent_model
        model = _build_agent_model()
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["model"] == "qwen/qwen3-32b"
        assert "thinking" in call_kwargs.get("extra_body", {})
        assert call_kwargs["extra_body"]["thinking"] is False


def test_build_chat_model_fallback_ollama():
    with patch("src.models.LLM_PROVIDER", "ollama"), \
         patch("src.models.ChatOllama") as mock_ollama:
        mock_ollama.return_value = MagicMock()
        from src.models import _build_chat_model
        model = _build_chat_model()
        mock_ollama.assert_called_once()
