from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.llm.provider import fallback_silence, parse_llm_decision
from app.schemas import InterventionCategory, LLMDecision


def test_schema_parsing_valid_json() -> None:
    decision = parse_llm_decision(
        '{"category":"KEEP_SILENCE","should_intervene":false,"confidence":0.91,'
        '"response":null,"reason":"not relevant","trigger_utterance_ids":[14]}'
    )
    assert decision.category == InterventionCategory.KEEP_SILENCE
    assert decision.should_intervene is False


def test_invalid_intervention_without_response_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMDecision(
            category=InterventionCategory.CHIME_IN,
            should_intervene=True,
            confidence=0.5,
            response=None,
            reason="bad",
            trigger_utterance_ids=[1],
        )


def test_invalid_llm_json_fallback_contract() -> None:
    decision = fallback_silence("invalid JSON")
    assert decision.category == InterventionCategory.KEEP_SILENCE
    assert decision.response is None
    assert decision.should_intervene is False

