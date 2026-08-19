from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.context import PROMPT_VERSION, ContextManager
from app.core.dedup import Deduplicator
from app.llm import LLMProvider
from app.schemas import (
    DelegateProfile,
    InterventionDecision,
    MeetingState,
    PreviousIntervention,
    Utterance,
)
from app.services.logger import EventLogger


class MeetingEngine:
    def __init__(
        self,
        delegate: DelegateProfile,
        llm_provider: LLMProvider,
        context_manager: ContextManager,
        event_logger: EventLogger,
        deduplicator: Deduplicator,
        meeting_id: str | None = None,
        model: str | None = None,
        max_suggestion_age_seconds: int = 15,
    ):
        self.state = MeetingState(meeting_id=meeting_id, delegate=delegate) if meeting_id else MeetingState(delegate=delegate)
        self.llm_provider = llm_provider
        self.context_manager = context_manager
        self.event_logger = event_logger
        self.deduplicator = deduplicator
        self.model = model
        self.max_suggestion_age_seconds = max_suggestion_age_seconds
        self._lock = asyncio.Lock()
        self.event_logger.start_session(self.state)

    async def ingest_utterance(self, utterance: Utterance) -> InterventionDecision:
        pipeline_start = time.perf_counter()
        async with self._lock:
            if any(existing.id == utterance.id for existing in self.state.utterances):
                utterance.id = max((u.id for u in self.state.utterances), default=0) + 1
            self.state.utterances.append(utterance)
            self.context_manager.update_recent_context(self.state)
            self.event_logger.log_utterance(self.state.meeting_id, utterance)
            prompt = self.context_manager.build_prompt(self.state, utterance)

        result = await self.llm_provider.decide_intervention(prompt, self.model)
        pipeline_latency_ms = int((time.perf_counter() - pipeline_start) * 1000)
        decision = InterventionDecision(
            utterance_id=utterance.id,
            category=result.decision.category,
            should_intervene=result.decision.should_intervene,
            confidence=result.decision.confidence,
            response=result.decision.response,
            reason=result.decision.reason,
            trigger_utterance_ids=result.decision.trigger_utterance_ids or [utterance.id],
            model=result.model,
            prompt_version=PROMPT_VERSION,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            llm_latency_ms=result.latency_ms,
            pipeline_latency_ms=pipeline_latency_ms,
            stale=pipeline_latency_ms / 1000 > self.max_suggestion_age_seconds,
        )
        async with self._lock:
            decision = self.deduplicator.apply(self.state, decision)
            if decision.displayed and decision.response:
                self.state.previous_interventions.append(
                    PreviousIntervention(
                        timestamp=datetime.now(timezone.utc),
                        utterance_id=utterance.id,
                        category=decision.category,
                        response=decision.response,
                    )
                )
            self.event_logger.log_decision(self.state.meeting_id, decision)
            self.event_logger.save_state(self.state)
        return decision

    @classmethod
    def from_paths(
        cls,
        delegate: DelegateProfile,
        llm_provider: LLMProvider,
        prompt_path: Path,
        data_dir: Path,
        meeting_id: str | None = None,
        model: str | None = None,
        context_max_utterances: int = 40,
        context_max_chars: int = 16000,
        cooldown_seconds: int = 10,
        filter_enabled: bool = True,
        max_suggestion_age_seconds: int = 15,
    ) -> "MeetingEngine":
        return cls(
            delegate=delegate,
            llm_provider=llm_provider,
            context_manager=ContextManager(prompt_path, context_max_utterances, context_max_chars),
            event_logger=EventLogger(data_dir),
            deduplicator=Deduplicator(cooldown_seconds, filter_enabled),
            meeting_id=meeting_id,
            model=model,
            max_suggestion_age_seconds=max_suggestion_age_seconds,
        )

