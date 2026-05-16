from unittest.mock import MagicMock


def test_scope_guardrail_check_returns_true_keyword_match():
    """Keyword match fast path: returns True without calling LLM."""
    from src.middleware import ScopeGuardrail
    mock_model = MagicMock()
    g = ScopeGuardrail(mock_model)
    # "diem" is in _SCOPE_KEYWORDS → fast path, no LLM call
    assert g.check("Quali corsi offre il DIEM?") is True
    mock_model.invoke.assert_not_called()


def test_scope_guardrail_check_returns_false_oot():
    """LLM says 'no' → returns False (out of scope)."""
    from src.middleware import ScopeGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="no")
    g = ScopeGuardrail(mock_model)
    # No keyword match → LLM call
    assert g.check("Chi è il re di Spagna?") is False


def test_scope_guardrail_check_returns_true_llm_yes():
    """LLM says 'yes' → returns True (ambiguous but in scope)."""
    from src.middleware import ScopeGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="yes")
    g = ScopeGuardrail(mock_model)
    assert g.check("Cosa si studia qui?") is True


def test_offensive_guardrail_check_returns_none_clean():
    """No badword match → returns None without calling LLM."""
    from src.middleware import OffensiveContentGuardrail
    mock_model = MagicMock()
    g = OffensiveContentGuardrail(mock_model)
    result = g.check("Il DIEM si trova a Fisciano.")
    assert result is None
    mock_model.invoke.assert_not_called()


def test_offensive_guardrail_check_returns_replacement_on_offensive():
    """Badword match + LLM confirms → returns replacement string."""
    from src.middleware import OffensiveContentGuardrail
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="yes")
    g = OffensiveContentGuardrail(mock_model)
    result = g.check("questo testo contiene cazzo e parole offensive")
    assert result is not None
    assert isinstance(result, str)
    assert "cazzo" not in result


def test_redact_pii_redacts_email():
    from src.middleware import redact_pii
    text = "Contattami a mario.rossi@unisa.it per info."
    result = redact_pii(text)
    assert "mario.rossi@unisa.it" not in result
    assert "[EMAIL REDACTED]" in result


def test_redact_pii_blocks_credit_card():
    from src.middleware import redact_pii
    text = "Il numero è 4111 1111 1111 1111."
    result = redact_pii(text)
    assert "4111" not in result


def test_redact_pii_passes_clean_text():
    from src.middleware import redact_pii
    text = "Il DIEM si trova a Fisciano, Campus di Salerno."
    assert redact_pii(text) == text
