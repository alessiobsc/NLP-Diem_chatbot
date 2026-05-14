from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage


def _make_state(messages):
    return {"messages": messages, "jump_to": None}


def test_scope_guardrail_passes_in_scope():
    from src.middleware import ScopeGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="yes")
    guardrail = ScopeGuardrail(mock_model)
    state = _make_state([HumanMessage(content="Quali corsi offre il DIEM?")])
    result = guardrail.before_agent(state, MagicMock())
    assert result is None


def test_scope_guardrail_rejects_out_of_scope():
    from src.middleware import ScopeGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="no")
    guardrail = ScopeGuardrail(mock_model)
    state = _make_state([HumanMessage(content="Chi è il re di Spagna?")])
    result = guardrail.before_agent(state, MagicMock())
    assert result is not None
    assert result.get("jump_to") == "end"
    assert any(isinstance(m, AIMessage) for m in result.get("messages", []))
    rejection_text = result["messages"][-1].content.lower()
    assert "scope" in rejection_text or "ambito" in rejection_text


def test_offensive_guardrail_passes_clean_content():
    from src.middleware import OffensiveContentGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="no")
    guardrail = OffensiveContentGuardrail(mock_model)
    state = _make_state([
        HumanMessage(content="Dove si trova il DIEM?"),
        AIMessage(content="Il DIEM si trova a Fisciano.", id="msg-1"),
    ])
    result = guardrail.after_agent(state, MagicMock())
    assert result is None


def test_offensive_guardrail_replaces_offensive_content():
    from src.middleware import OffensiveContentGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="yes")
    guardrail = OffensiveContentGuardrail(mock_model)
    state = _make_state([
        HumanMessage(content="test"),
        AIMessage(content="questo testo contiene cazzo e parole offensive", id="msg-1"),
    ])
    result = guardrail.after_agent(state, MagicMock())
    assert result is not None
    assert any(isinstance(m, AIMessage) for m in result.get("messages", []))
    assert result["messages"][-1].content != "questo testo contiene cazzo e parole offensive"


def test_build_middleware_returns_list():
    from src.middleware import build_middleware
    mock_model = MagicMock()
    middlewares = build_middleware(mock_model)
    assert len(middlewares) == 4
