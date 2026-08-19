from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.meeting import MeetingEngine
from app.schemas import InterventionDecision, MeetingEvent, Utterance, utc_now


DecisionCallback = Callable[[InterventionDecision], Awaitable[None]]
EventCallback = Callable[[MeetingEvent], Awaitable[None]]


class LiveMeetingSession:
    def __init__(
        self,
        engine: MeetingEngine,
        max_llm_concurrency: int = 1,
        on_decision: DecisionCallback | None = None,
        on_event: EventCallback | None = None,
    ):
        self.engine = engine
        self.semaphore = asyncio.Semaphore(max_llm_concurrency)
        self.on_decision = on_decision
        self.on_event = on_event
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_utterance_id = 1

    async def ingest_final_utterance(self, speaker: str, text: str, source: str = "UNKNOWN") -> Utterance:
        utterance = Utterance(
            id=self._next_utterance_id,
            speaker=speaker,
            text=text,
            source=source,  # type: ignore[arg-type]
            timestamp=utc_now(),
        )
        self._next_utterance_id += 1
        if self.on_event:
            await self.on_event(
                MeetingEvent(
                    type="utterance.final",
                    meeting_id=self.engine.state.meeting_id,
                    payload=utterance.model_dump(mode="json"),
                )
            )
        task = asyncio.create_task(self._decide(utterance))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return utterance

    async def _decide(self, utterance: Utterance) -> None:
        async with self.semaphore:
            if self.on_event:
                await self.on_event(
                    MeetingEvent(
                        type="intervention.requested",
                        meeting_id=self.engine.state.meeting_id,
                        payload={"utterance_id": utterance.id},
                    )
                )
            decision = await self.engine.ingest_utterance(utterance)
            if self.on_event:
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
            await asyncio.gather(*self._tasks)

