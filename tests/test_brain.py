from unittest.mock import MagicMock, patch
import pytest


def _make_mock_vectorstore():
    from langchain_chroma import Chroma
    vs = MagicMock(spec=Chroma)
    return vs


def test_diembrain_init_creates_agent():
    with patch("src.brain._build_agent_model") as mock_agent_m, \
         patch("src.brain._build_chat_model") as mock_chat_m, \
         patch("src.brain.build_tools") as mock_tools, \
         patch("src.brain.create_agent") as mock_ca, \
         patch("src.brain.DiemBrain._build_retriever") as mock_ret:
        mock_agent_m.return_value = MagicMock()
        mock_chat_m.return_value = MagicMock()
        mock_tools.return_value = []
        mock_ret.return_value = MagicMock()
        mock_ca.return_value = MagicMock()

        from src.brain import DiemBrain
        brain = DiemBrain(_make_mock_vectorstore())

        mock_ca.assert_called_once()
        assert brain._agent is mock_ca.return_value


def test_chat_eval_returns_answer_and_sources():
    with patch("src.brain._build_agent_model") as mock_agent_m, \
         patch("src.brain._build_chat_model") as mock_chat_m, \
         patch("src.brain.build_tools") as mock_tools, \
         patch("src.brain.create_agent") as mock_ca, \
         patch("src.brain.DiemBrain._build_retriever") as mock_ret:
        from langchain_core.messages import AIMessage

        mock_agent_m.return_value = MagicMock()
        mock_chat_m.return_value = MagicMock()
        mock_tools.return_value = []
        mock_ret.return_value = MagicMock()

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Il DIEM si trova a Fisciano.")]
        }
        mock_ca.return_value = mock_agent

        from src.brain import DiemBrain
        brain = DiemBrain(_make_mock_vectorstore())
        result = brain.chat_eval("Dove si trova il DIEM?", "test-session")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Il DIEM si trova a Fisciano."
        assert isinstance(result["sources"], list)


def test_get_history_returns_empty_for_new_session():
    with patch("src.brain._build_agent_model") as mock_agent_m, \
         patch("src.brain._build_chat_model") as mock_chat_m, \
         patch("src.brain.build_tools") as mock_tools, \
         patch("src.brain.create_agent") as mock_ca, \
         patch("src.brain.DiemBrain._build_retriever") as mock_ret:
        mock_agent_m.return_value = MagicMock()
        mock_chat_m.return_value = MagicMock()
        mock_tools.return_value = []
        mock_ret.return_value = MagicMock()
        mock_ca.return_value = MagicMock()

        from src.brain import DiemBrain
        brain = DiemBrain(_make_mock_vectorstore())
        history = brain.get_history("new-session-xyz")
        assert history == []
