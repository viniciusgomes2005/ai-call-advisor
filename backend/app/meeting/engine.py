from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.context import PROMPT_VERSION, ContextManager
from app.core.dedup import Deduplicator
from app.core.insights import detect_meeting_insights
from app.llm import LLMProvider
from app.schemas import (
    DelegateProfile,
    InterventionDecision,
    MeetingInsight,
    MeetingQuestionResponse,
    MeetingState,
    PreviousIntervention,
    SemanticUtterance,
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
        enable_structured_meeting_state: bool = False,
    ):
        self.state = MeetingState(meeting_id=meeting_id, delegate=delegate) if meeting_id else MeetingState(delegate=delegate)
        self.llm_provider = llm_provider
        self.context_manager = context_manager
        self.event_logger = event_logger
        self.deduplicator = deduplicator
        self.model = model
        self.max_suggestion_age_seconds = max_suggestion_age_seconds
        self.enable_structured_meeting_state = enable_structured_meeting_state
        self._lock = asyncio.Lock()
        self.event_logger.start_session(self.state)

    async def answer_question(self, question: str) -> MeetingQuestionResponse:
        async with self._lock:
            prompt = self.context_manager.build_question_prompt(self.state, question)

        result = await self.llm_provider.answer_question(prompt, self.model)
        return MeetingQuestionResponse(
            meeting_id=self.state.meeting_id,
            question=question,
            answer=result.text,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            llm_latency_ms=result.latency_ms,
        )

    async def record_utterance(self, utterance: Utterance) -> list[MeetingInsight]:
        async with self._lock:
            if any(existing.id == utterance.id for existing in self.state.utterances):
                utterance.id = max((u.id for u in self.state.utterances), default=0) + 1
            self.state.utterances.append(utterance)
            insights = detect_meeting_insights(utterance)
            self.state.insights.extend(insights)
            self._update_conversation_state(utterance, insights)
            self.context_manager.update_recent_context(self.state)
            self.event_logger.log_utterance(self.state.meeting_id, utterance)
            for insight in insights:
                self.event_logger.log_insight(self.state.meeting_id, insight)
            self.event_logger.save_state(self.state)
            return insights

    async def ingest_utterance(self, utterance: Utterance) -> InterventionDecision:
        pipeline_start = time.perf_counter()
        async with self._lock:
            if any(existing.id == utterance.id for existing in self.state.utterances):
                utterance.id = max((u.id for u in self.state.utterances), default=0) + 1
            self.state.utterances.append(utterance)
            insights = detect_meeting_insights(utterance)
            self.state.insights.extend(insights)
            self._update_conversation_state(utterance, insights)
            self.context_manager.update_recent_context(self.state)
            self.event_logger.log_utterance(self.state.meeting_id, utterance)
            for insight in insights:
                self.event_logger.log_insight(self.state.meeting_id, insight)
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
            total_suggestion_latency_ms=(
                (utterance.audio_finalize_latency_ms or 0) + (utterance.asr_latency_ms or 0) + pipeline_latency_ms
                if utterance.asr_latency_ms is not None or utterance.audio_finalize_latency_ms is not None
                else None
            ),
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

    async def record_semantic_utterance(
        self, semantic: SemanticUtterance, speaker: str, utterance_id: int
    ) -> tuple[Utterance, list[MeetingInsight]]:
        utterance = self.utterance_from_semantic(semantic, speaker, utterance_id)
        insights = await self.record_utterance(utterance)
        return utterance, insights

    async def ingest_semantic_utterance(
        self, semantic: SemanticUtterance, speaker: str, utterance_id: int
    ) -> tuple[Utterance, InterventionDecision]:
        utterance = self.utterance_from_semantic(semantic, speaker, utterance_id)
        decision = await self.ingest_utterance(utterance)
        return utterance, decision

    @staticmethod
    def utterance_from_semantic(semantic: SemanticUtterance, speaker: str, utterance_id: int) -> Utterance:
        return Utterance(
            id=utterance_id,
            speaker=speaker,
            text=semantic.text,
            source=semantic.source,  # type: ignore[arg-type]
            start=semantic.start,
            end=semantic.end,
            language=semantic.language,
            asr_latency_ms=int(semantic.asr_latency_ms) if semantic.asr_latency_ms is not None else None,
            audio_finalize_latency_ms=(
                int(semantic.audio_finalize_latency_ms) if semantic.audio_finalize_latency_ms is not None else None
            ),
            semantic_id=semantic.id,
            segment_ids=list(semantic.segment_ids),
            assembly_reason=semantic.assembly_reason,
            assembly_latency_ms=semantic.assembly_latency_ms,
        )

    def _update_conversation_state(self, utterance: Utterance, insights: list[MeetingInsight]) -> None:
        if not self.enable_structured_meeting_state:
            return
        state = self.state.conversation_state
        text = utterance.text.strip()
        if text and text not in state.facts:
            state.facts.append(text)
            state.facts = state.facts[-50:]
        for insight in insights:
            if insight.type == "OPEN_QUESTION" and insight.text not in state.open_questions:
                state.open_questions.append(insight.text)
            elif insight.type == "DECISION" and insight.text not in state.decisions:
                state.decisions.append(insight.text)

    def insights_for_utterance(self, utterance_id: int) -> list[MeetingInsight]:
        return [insight for insight in self.state.insights if insight.utterance_id == utterance_id]

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
