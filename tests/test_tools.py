from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _make_tools():
    from src.tools import build_tools
    mock_retriever = MagicMock()
    mock_model = MagicMock()
    mock_prompt = MagicMock()
    brain_ref = MagicMock()
    brain_ref._last_docs = []
    return build_tools(mock_retriever, mock_model, brain_ref, mock_prompt)


def test_build_tools_returns_four_tools():
    tools = _make_tools()
    assert len(tools) == 4


def test_tool_names():
    tools = _make_tools()
    names = {t.name for t in tools}
    assert names == {"retrieve", "summarize", "calculate", "answer"}


def test_retrieve_updates_last_docs():
    from src.tools import build_tools

    mock_doc = Document(page_content="test content", metadata={"source": "http://example.com"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    mock_model = MagicMock()
    mock_prompt = MagicMock()
    brain_ref = MagicMock()
    brain_ref._last_docs = []

    with patch("src.brain.rerank", return_value=[mock_doc]), \
         patch("src.brain._format_context", return_value={"context": "<document>test</document>"}):
        tools = build_tools(mock_retriever, mock_model, brain_ref, mock_prompt)
        retrieve_tool = next(t for t in tools if t.name == "retrieve")
        result = retrieve_tool.invoke("test query")

    assert brain_ref._last_docs == [mock_doc]
    assert "document" in result


def test_calculate_uses_provided_context():
    from src.tools import build_tools

    mock_retriever = MagicMock()
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="Il voto di laurea è 107.")
    mock_prompt = MagicMock()
    brain_ref = MagicMock()
    brain_ref._last_docs = []

    tools = build_tools(mock_retriever, mock_model, brain_ref, mock_prompt)
    calculate_tool = next(t for t in tools if t.name == "calculate")
    result = calculate_tool.invoke({
        "context": "La formula è media/30 * 110.",
        "operation": "graduation_grade",
        "values": {"average": 28.8}
    })

    mock_model.invoke.assert_called_once()
    assert "107" in result
