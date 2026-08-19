from __future__ import annotations

from app.core.events import EventBus
from app.schemas import MeetingEvent
from app.services.live import LiveMeetingSession
from app.schemas import Utterance

from .conftest import StaticLLMProvider, make_test_engine


async def test_event_bus_ordering():
    bus = EventBus()
    seen = []

    async def handler(event: MeetingEvent) -> None:
        seen.append(event.type)

    bus.subscribe("*", handler)
    await bus.publish(MeetingEvent(type="audio.chunk"))
    await bus.publish(MeetingEvent(type="speech.started"))
    await bus.publish(MeetingEvent(type="speech.ended"))
    assert seen == ["audio.chunk", "speech.started", "speech.ended"]


async def test_live_session_does_not_block_new_utterances(tmp_path, delegate, prompt_path):
    events = []

    async def on_event(event: MeetingEvent) -> None:
        events.append(event.type)

    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    session = LiveMeetingSession(engine, max_llm_concurrency=1, on_event=on_event)
    await session.ingest_final_utterance("ME", "first", "LOCAL_MIC_AUDIO")
    await session.ingest_final_utterance("REMOTE", "second", "REMOTE_AUDIO")
    await session.drain()
    assert events.count("utterance.final") == 2
    assert events.count("intervention.decided") == 2
    assert [u.text for u in engine.state.utterances] == ["first", "second"]

