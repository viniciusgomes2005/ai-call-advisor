from __future__ import annotations

import asyncio

from app.meeting import MeetingEngine
from app.schemas import InterventionDecision, ReplayRequest


class ReplayService:
    def __init__(self, engine: MeetingEngine):
        self.engine = engine

    async def run(self, request: ReplayRequest) -> list[InterventionDecision]:
        decisions: list[InterventionDecision] = []
        for utterance in sorted(request.utterances, key=lambda item: item.id):
            utterance.source = "REPLAY"
            decisions.append(await self.engine.ingest_utterance(utterance))
            if request.realtime_delay_ms > 0:
                await asyncio.sleep(request.realtime_delay_ms / 1000)
        return decisions

