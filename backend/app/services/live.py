from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.meeting import MeetingEngine
from app.schemas import InterventionDecision, MeetingEvent, SemanticUtterance, Utterance, utc_now


DecisionCallback = Callable[[InterventionDecision], Awaitable[None]]
EventCallback = Callable[[MeetingEvent], Awaitable[None]]


class LiveMeetingSession:
    def __init__(
        self,
        engine: MeetingEngine,
        max_llm_concurrency: int = 1,
        llm_enabled: bool = True,
        on_decision: DecisionCallback | None = None,
        on_event: EventCallback | None = None,
    ):
        self.engine = engine
        self.semaphore = asyncio.Semaphore(max_llm_concurrency)
        self.llm_enabled = llm_enabled
        self.on_decision = on_decision
        self.on_event = on_event
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_utterance_id = 1

    async def ingest_final_utterance(
        self,
        speaker: str,
        text: str,
        source: str = "UNKNOWN",
        start: float | None = None,
        end: float | None = None,
        language: str | None = None,
        asr_latency_ms: int | None = None,
        audio_finalize_latency_ms: int | None = None,
    ) -> Utterance:
        utterance = Utterance(
            id=self._next_utterance_id,
            speaker=speaker,
            text=text,
            source=source,  # type: ignore[arg-type]
            timestamp=utc_now(),
            start=start,
            end=end,
            language=language,
            asr_latency_ms=asr_latency_ms,
            audio_finalize_latency_ms=audio_finalize_latency_ms,
        )
        self._next_utterance_id += 1
        return await self._dispatch_utterance(utterance)

    async def ingest_semantic_utterance(self, semantic: SemanticUtterance, speaker: str) -> Utterance:
        """Ingest a finalized SemanticUtterance (the assembler's output).

        This is the only path that should reach MeetingEngine/LLM in the live
        pipeline - individual acoustic ASR fragments (transcript.segment /
        semantic_utterance.updated) must never call this. Builds the Utterance
        once and dispatches it exactly once, so there is no risk of the same
        SemanticUtterance being both recorded and ingested.
        """
        utterance = MeetingEngine.utterance_from_semantic(semantic, speaker, self._next_utterance_id)
        self._next_utterance_id += 1
        return await self._dispatch_utterance(utterance)

    async def _dispatch_utterance(self, utterance: Utterance) -> Utterance:
        if self.on_event:
            await self.on_event(
                MeetingEvent(
                    type="utterance.final",
                    meeting_id=self.engine.state.meeting_id,
                    payload=utterance.model_dump(mode="json"),
                )
            )
        if not self.llm_enabled:
            await self._record_without_llm(utterance)
            return utterance
        task = asyncio.create_task(self._decide(utterance))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return utterance

    async def _record_without_llm(self, utterance: Utterance) -> None:
        insights = await self.engine.record_utterance(utterance)
        if self.on_event:
            for insight in insights:
                await self.on_event(
                    MeetingEvent(
                        type="meeting.insight.detected",
                        meeting_id=self.engine.state.meeting_id,
                        payload=insight.model_dump(mode="json"),
                    )
                )

    async def _decide(self, utterance: Utterance) -> None:
        async with self.semaphore:
            if not self.llm_enabled:
                await self._record_without_llm(utterance)
                return
            if self.on_event:
                await self.on_event(
                    MeetingEvent(
                        type="intervention.requested",
                        meeting_id=self.engine.state.meeting_id,
                        payload={"utterance_id": utterance.id},
                    )
                )
            try:
                decision = await self.engine.ingest_utterance(utterance)
            except Exception as exc:
                if not any(existing.id == utterance.id for existing in self.engine.state.utterances):
                    await self.engine.record_utterance(utterance)
                else:
                    self.engine.event_logger.save_state(self.engine.state)
                if self.on_event:
                    await self.on_event(
                        MeetingEvent(
                            type="intervention.error",
                            meeting_id=self.engine.state.meeting_id,
                            payload={"utterance_id": utterance.id, "error": str(exc)},
                        )
                    )
                return
            if self.on_event:
                for insight in self.engine.insights_for_utterance(utterance.id):
                    await self.on_event(
                        MeetingEvent(
                            type="meeting.insight.detected",
                            meeting_id=self.engine.state.meeting_id,
                            payload=insight.model_dump(mode="json"),
                        )
                    )
                await self.on_event(
                    MeetingEvent(
                        type="intervention.decided",
                        meeting_id=self.engine.state.meeting_id,
                        payload=decision.model_dump(mode="json"),
                    )
                )
            if self.on_decision:
                await self.on_decision(decision)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
