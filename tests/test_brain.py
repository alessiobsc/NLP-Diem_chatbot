from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document


def _make_mock_vectorstore():
    from langchain_chroma import Chroma
    return MagicMock(spec=Chroma)


def _make_brain():
    """Create a DiemBrain with all heavy dependencies mocked out.

    _build_graph is patched so ToolNode never sees real tool objects —
    that avoids attribute errors on MagicMock tools during graph compilation.
    """
    with patch("src.brain._build_agent_model") as mock_am, \
         patch("src.brain._build_chat_model") as mock_cm, \
         patch("src.brain.build_tools", return_value=[]), \
         patch("src.brain.DiemBrain._build_retriever"), \
         patch("src.brain.DiemBrain._build_graph") as mock_bg:
        mock_am.return_value = MagicMock()
        mock_am.return_value.bind_tools.return_value = MagicMock()
        mock_cm.return_value = MagicMock()
        mock_bg.return_value = MagicMock()
        from src.brain import DiemBrain
        return DiemBrain(_make_mock_vectorstore())


# ── Routing functions ─────────────────────────────────────────────────────────

def test_route_scope_to_retrieve_when_last_human():
    from src.brain import _route_scope, DiemState
    state = DiemState(
        messages=[HumanMessage(content="Dove si trova il DIEM?")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    assert _route_scope(state) == "retrieve_node"


def test_route_scope_to_end_when_last_ai():
    from src.brain import _route_scope, DiemState
    # scope_guard injected an AIMessage (rejection) → route to END
    state = DiemState(
        messages=[HumanMessage(content="test"), AIMessage(content="rejected")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    assert _route_scope(state) == "__end__"


def test_route_agent_to_tools_when_tool_calls_under_limit():
    from src.brain import _route_agent, DiemState
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve", "args": {}, "id": "x", "type": "tool_call"}],
    )
    state = DiemState(
        messages=[ai_msg], tool_call_count=0, retrieved_context="", last_docs=[],
    )
    assert _route_agent(state) == "tools"


def test_route_agent_to_generate_when_no_tool_calls():
    from src.brain import _route_agent, DiemState
    state = DiemState(
        messages=[AIMessage(content="some response")],
        tool_call_count=0, retrieved_context="ctx", last_docs=[],
    )
    assert _route_agent(state) == "generate"


def test_route_agent_to_generate_at_max_tool_calls():
    from src.brain import _route_agent, DiemState
    from config import MAX_TOOL_CALLS
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "retrieve", "args": {}, "id": "x", "type": "tool_call"}],
    )
    # At cap: even with tool_calls present, must go to generate
    state = DiemState(
        messages=[ai_msg], tool_call_count=MAX_TOOL_CALLS, retrieved_context="", last_docs=[],
    )
    assert _route_agent(state) == "generate"


# ── Node: scope_guard ─────────────────────────────────────────────────────────

def test_node_scope_guard_rejects_oot():
    brain = _make_brain()
    brain._scope_guardrail = MagicMock()
    brain._scope_guardrail.check.return_value = False

    from src.brain import DiemState
    state = DiemState(
        messages=[HumanMessage(content="Chi ha vinto la Coppa del Mondo?")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    result = brain._node_scope_guard(state)
    # Returns an AIMessage with rejection text
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


def test_node_scope_guard_passes_in_scope():
    brain = _make_brain()
    brain._scope_guardrail = MagicMock()
    brain._scope_guardrail.check.return_value = True

    from src.brain import DiemState
    state = DiemState(
        messages=[HumanMessage(content="Quali corsi offre il DIEM?")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    result = brain._node_scope_guard(state)
    # No state change when in scope
    assert result == {}


# ── Node: retrieve_node ───────────────────────────────────────────────────────

def test_node_retrieve_populates_state():
    brain = _make_brain()
    mock_doc = Document(page_content="test", metadata={"source": "http://test.com"})

    with patch("src.brain.rerank", return_value=[mock_doc]), \
         patch("src.brain._format_context", return_value={"context": "<document>test</document>"}):
        brain._retriever = MagicMock()
        brain._retriever.invoke.return_value = [mock_doc]

        from src.brain import DiemState
        state = DiemState(
            messages=[HumanMessage(content="Dove si trova il DIEM?")],
            tool_call_count=3,  # should be reset to 0
            retrieved_context="", last_docs=[],
        )
        result = brain._node_retrieve(state)

    assert result["retrieved_context"] == "<document>test</document>"
    assert result["last_docs"] == [mock_doc]
    assert result["tool_call_count"] == 0  # reset each turn

    # Two messages: fake AIMessage (retrieve tool call) + ToolMessage (context)
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], AIMessage)
    assert isinstance(result["messages"][1], ToolMessage)
    assert result["messages"][0].tool_calls[0]["name"] == "retrieve"
    assert result["messages"][1].content == "<document>test</document>"


# ── Public API ────────────────────────────────────────────────────────────────

def test_diembrain_init_builds_graph():
    brain = _make_brain()
    # _graph is assigned from _build_graph() — mocked but not None
    assert brain._graph is not None


def test_chat_eval_returns_answer_and_sources():
    brain = _make_brain()
    mock_doc = Document(page_content="test", metadata={"source": "http://test.com"})
    brain._graph = MagicMock()
    brain._graph.invoke.return_value = {
        "messages": [AIMessage(content="Il DIEM si trova a Fisciano.")],
        "last_docs": [mock_doc],
    }

    result = brain.chat_eval("Dove si trova il DIEM?", "test-session")
    assert result["answer"] == "Il DIEM si trova a Fisciano."
    assert result["sources"] == [mock_doc]


def test_get_history_returns_empty_for_new_session():
    brain = _make_brain()
    brain._graph = MagicMock()
    brain._graph.get_state.return_value = MagicMock(values={"messages": []})
    assert brain.get_history("new-session-xyz") == []
