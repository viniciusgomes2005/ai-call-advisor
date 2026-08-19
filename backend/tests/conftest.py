from __future__ import annotations

from pathlib import Path

import pytest

from app.core.context import ContextManager
from app.core.dedup import Deduplicator
from app.llm.provider import LLMProvider, LLMResult, fallback_silence
from app.meeting import MeetingEngine
from app.schemas import DelegateProfile, InterventionCategory, LLMDecision, ModelInfo
from app.services.logger import EventLogger


class StaticLLMProvider(LLMProvider):
    def __init__(self, decision: LLMDecision | None = None):
        self.decision = decision or fallback_silence("static silence")
        self.prompts: list[str] = []

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="test-model")]

    async def decide_intervention(self, prompt: str, model: str | None = None) -> LLMResult:
        self.prompts.append(prompt)
        return LLMResult(self.decision, model or "test-model", 10, 20, 3, self.decision.model_dump_json())


class HeuristicLLMProvider(StaticLLMProvider):
    async def decide_intervention(self, prompt: str, model: str | None = None) -> LLMResult:
        self.prompts.append(prompt)
        text = prompt.lower()
        if "bob, can you" in text:
            decision = LLMDecision(
                category=InterventionCategory.EXPLICIT_CUE,
                should_intervene=True,
                confidence=0.9,
                response="Authentication is ready; it was completed last week.",
                reason="Bob was directly asked.",
                trigger_utterance_ids=[2],
            )
        elif "anyone from backend" in text:
            decision = LLMDecision(
                category=InterventionCategory.IMPLICIT_CUE,
                should_intervene=True,
                confidence=0.85,
                response="Backend can confirm authentication is integrated.",
                reason="The backend role is implicitly requested.",
                trigger_utterance_ids=[2],
            )
        elif "latency" in text or "wms externo" in text:
            decision = LLMDecision(
                category=InterventionCategory.CHIME_IN,
                should_intervene=True,
                confidence=0.8,
                response="Ask whether this was tested under concurrent requests.",
                reason="Relevant to delegate intent.",
                trigger_utterance_ids=[2],
            )
        else:
            decision = fallback_silence("unrelated")
        return LLMResult(decision, model or "test-model", 10, 20, 3, decision.model_dump_json())


@pytest.fixture
def delegate() -> DelegateProfile:
    return DelegateProfile(
        name="Bob",
        role="Backend Engineer",
        meeting_intent=["Understand the status of the voice feature"],
        shareable_information=[
            {"context": "When backend integration is discussed", "information": "Auth finished last week"}
        ],
    )


@pytest.fixture
def prompt_path() -> Path:
    return Path("prompts/intervention_v1.txt")


def make_test_engine(
    tmp_path: Path,
    delegate: DelegateProfile,
    llm: LLMProvider,
    prompt_path: Path,
    filter_enabled: bool = False,
    cooldown_seconds: int = 10,
    context_max_utterances: int = 40,
) -> MeetingEngine:
    return MeetingEngine(
        delegate=delegate,
        llm_provider=llm,
        context_manager=ContextManager(prompt_path, context_max_utterances, 16000),
        event_logger=EventLogger(tmp_path / "sessions"),
        deduplicator=Deduplicator(cooldown_seconds, filter_enabled),
    )

