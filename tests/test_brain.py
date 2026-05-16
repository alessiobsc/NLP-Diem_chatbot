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


# ── Node: generate ────────────────────────────────────────────────────────────

def test_node_generate_invokes_generation_model():
    brain = _make_brain()
    brain._generation_model = MagicMock()
    brain._generation_model.invoke.return_value = MagicMock(content="Il DIEM si trova a Fisciano.")

    from src.brain import DiemState
    state = DiemState(
        messages=[HumanMessage(content="Dove si trova il DIEM?")],
        tool_call_count=0,
        retrieved_context="<document>Il DIEM è a Fisciano.</document>",
        last_docs=[],
    )
    result = brain._node_generate(state)

    brain._generation_model.invoke.assert_called_once()
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Il DIEM si trova a Fisciano."


# ── Node: output_guard ────────────────────────────────────────────────────────

def test_node_output_guard_passes_clean_content():
    brain = _make_brain()
    brain._offensive_guardrail = MagicMock()
    brain._offensive_guardrail.check.return_value = None  # clean

    from src.brain import DiemState
    state = DiemState(
        messages=[AIMessage(content="Il DIEM si trova a Fisciano.")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    result = brain._node_output_guard(state)
    assert result == {}  # no state change


def test_node_output_guard_replaces_offensive_content():
    brain = _make_brain()
    brain._offensive_guardrail = MagicMock()
    brain._offensive_guardrail.check.return_value = "Non posso fornire questa risposta."

    from src.brain import DiemState
    state = DiemState(
        messages=[AIMessage(content="contenuto offensivo")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    result = brain._node_output_guard(state)
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Non posso fornire questa risposta."


def test_node_output_guard_redacts_pii():
    brain = _make_brain()
    brain._offensive_guardrail = MagicMock()
    brain._offensive_guardrail.check.return_value = None  # not offensive

    from src.brain import DiemState
    state = DiemState(
        messages=[AIMessage(content="Contattami a test@example.com per info.")],
        tool_call_count=0, retrieved_context="", last_docs=[],
    )
    result = brain._node_output_guard(state)
    assert len(result["messages"]) == 1
    assert "test@example.com" not in result["messages"][0].content
    assert "[EMAIL REDACTED]" in result["messages"][0].content


# ── Public API: chat ──────────────────────────────────────────────────────────

def test_chat_returns_answer_with_sources():
    brain = _make_brain()
    mock_doc = Document(page_content="test", metadata={"source": "http://test.com"})
    brain._graph = MagicMock()
    brain._graph.invoke.return_value = {
        "messages": [AIMessage(content="Il DIEM si trova a Fisciano.")],
        "last_docs": [mock_doc],
    }

    result = brain.chat("Dove si trova il DIEM?", "test-session")
    assert "Il DIEM si trova a Fisciano." in result
    assert "http://test.com" in result


def test_chat_returns_rejection_without_sources():
    from src.middleware import _SCOPE_REJECTION
    brain = _make_brain()
    brain._graph = MagicMock()
    brain._graph.invoke.return_value = {
        "messages": [AIMessage(content=_SCOPE_REJECTION)],
        "last_docs": [],
    }

    result = brain.chat("Chi ha vinto il Mondiale?", "test-session")
    # Rejection text is returned; no sources appended
    assert "Sources" not in result
    assert _SCOPE_REJECTION in result


# ── Public API: chat_stream ───────────────────────────────────────────────────

def test_chat_stream_yields_generate_tokens():
    brain = _make_brain()

    # Simulate stream: one chunk from 'generate' node
    generate_chunk = AIMessage(content="Il DIEM è a Fisciano.")
    brain._graph = MagicMock()
    brain._graph.stream.return_value = iter([
        (generate_chunk, {"langgraph_node": "generate"}),
    ])
    brain._last_docs = []

    tokens = list(brain.chat_stream("Dove?", "test-session"))
    assert any("DIEM" in t for t in tokens)


def test_chat_stream_yields_rejection_on_scope_guard():
    from src.middleware import _SCOPE_REJECTION
    brain = _make_brain()

    # Simulate stream: scope_guard emits rejection AIMessage, no generate node runs
    rejection_chunk = AIMessage(content=_SCOPE_REJECTION)
    brain._graph = MagicMock()
    brain._graph.stream.return_value = iter([
        (rejection_chunk, {"langgraph_node": "scope_guard"}),
    ])

    tokens = list(brain.chat_stream("Chi vince il Mondiale?", "test-session"))
    # Must yield something (the rejection), not empty
    assert len(tokens) > 0
    assert any(t.strip() for t in tokens)
