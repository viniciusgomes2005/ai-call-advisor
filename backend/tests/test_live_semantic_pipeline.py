from __future__ import annotations

from datetime import datetime, timezone

from app.core.events import EventBus
from app.meeting.utterance_assembler import UtteranceAssembler, UtteranceAssemblerConfig
from app.schemas import AudioFinalizationReason, MeetingEvent, TranscriptSegment
from app.services.live import LiveMeetingSession

from .conftest import StaticLLMProvider, make_test_engine


def make_segment(
    *, id: str, source: str, start: float, end: float, text: str, reason: AudioFinalizationReason
) -> TranscriptSegment:
    return TranscriptSegment(
        id=id,
        source=source,
        start=start,
        end=end,
        text=text,
        created_at=datetime.now(timezone.utc),
        finalization_reason=reason,
    )


def continuous_speech_segments() -> list[TranscriptSegment]:
    """Three ASR fragments of one continuous 14s utterance: two artificial MAX_DURATION
    cuts (at 6s and 12s, from ASR_MAX_UTTERANCE_MS) followed by a real SILENCE end."""
    return [
        make_segment(
            id="seg_tab_audio_0_0",
            source="TAB_AUDIO",
            start=0.0,
            end=6.0,
            text="Hoje usamos ECC para",
            reason=AudioFinalizationReason.MAX_DURATION,
        ),
        make_segment(
            id="seg_tab_audio_1_0",
            source="TAB_AUDIO",
            start=6.0,
            end=12.0,
            text="controlar a parte financeira e",
            reason=AudioFinalizationReason.MAX_DURATION,
        ),
        make_segment(
            id="seg_tab_audio_2_0",
            source="TAB_AUDIO",
            start=12.0,
            end=14.0,
            text="também um WMS externo.",
            reason=AudioFinalizationReason.SILENCE,
        ),
    ]


async def test_continuous_speech_produces_exactly_one_llm_call(tmp_path, delegate, prompt_path):
    """The main regression test for this refactor: 3 acoustic TranscriptSegments from one
    continuous utterance must collapse into exactly 1 SemanticUtterance, 1 MeetingEngine
    ingest, 1 LLM call and 1 utterance.final event - never one per acoustic fragment."""
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    events: list[str] = []

    async def on_event(event: MeetingEvent) -> None:
        events.append(event.type)

    session = LiveMeetingSession(engine, max_llm_concurrency=1, on_event=on_event)
    assembler = UtteranceAssembler(UtteranceAssemblerConfig(merge_max_gap_ms=1200, finalization_delay_ms=500))

    clock = 0.0
    semantic_utterances = []
    for segment in continuous_speech_segments():
        clock += 0.1
        result = assembler.push(segment, now=clock)
        semantic_utterances.extend(result.finalized)
    clock += 0.6  # past the finalization delay, with no further continuation
    semantic_utterances.extend(assembler.flush_expired(now=clock).finalized)

    assert len(semantic_utterances) == 1
    semantic = semantic_utterances[0]
    assert semantic.segment_count == 3
    assert semantic.text == "Hoje usamos ECC para controlar a parte financeira e também um WMS externo."

    await session.ingest_semantic_utterance(semantic, speaker="REMOTE")
    await session.drain()

    assert events.count("utterance.final") == 1
    assert events.count("intervention.decided") == 1
    assert len(llm.prompts) == 1
    assert len(engine.state.utterances) == 1
    assert engine.state.utterances[0].text == semantic.text
    assert engine.state.utterances[0].segment_ids == [
        "seg_tab_audio_0_0",
        "seg_tab_audio_1_0",
        "seg_tab_audio_2_0",
    ]


async def test_ingest_semantic_utterance_does_not_double_record_when_llm_enabled(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    session = LiveMeetingSession(engine, max_llm_concurrency=1, llm_enabled=True)
    segment = make_segment(
        id="seg_mic_0_0", source="MIC", start=0.0, end=1.0, text="teste unico", reason=AudioFinalizationReason.SILENCE
    )
    assembler = UtteranceAssembler()
    assembler.push(segment)
    finalized = assembler.flush_source("MIC").finalized
    assert finalized

    await session.ingest_semantic_utterance(finalized[0], speaker="ME")
    await session.drain()

    # ingest_utterance (LLM path) must run exactly once - never also through record_utterance.
    assert len(engine.state.utterances) == 1
    assert len(llm.prompts) == 1


async def test_ingest_semantic_utterance_records_once_when_llm_disabled(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    session = LiveMeetingSession(engine, max_llm_concurrency=1, llm_enabled=False)
    segment = make_segment(
        id="seg_mic_0_0", source="MIC", start=0.0, end=1.0, text="teste unico", reason=AudioFinalizationReason.SILENCE
    )
    assembler = UtteranceAssembler()
    assembler.push(segment)
    finalized = assembler.flush_source("MIC").finalized

    await session.ingest_semantic_utterance(finalized[0], speaker="ME")
    await session.drain()

    assert len(engine.state.utterances) == 1
    assert llm.prompts == []  # record path never calls the LLM


async def test_event_bus_ordering_is_unaffected():
    bus = EventBus()
    seen = []

    async def handler(event: MeetingEvent) -> None:
        seen.append(event.type)

    bus.subscribe("*", handler)
    await bus.publish(MeetingEvent(type="transcript.segment"))
    await bus.publish(MeetingEvent(type="semantic_utterance.updated"))
    await bus.publish(MeetingEvent(type="semantic_utterance.final"))
    assert seen == ["transcript.segment", "semantic_utterance.updated", "semantic_utterance.final"]


async def test_ab_benchmark_assembler_on_vs_off_for_continuous_speech(tmp_path, delegate, prompt_path):
    """Simple A/B utility: for the same 3 acoustic fragments of one continuous utterance,
    assembler ON collapses them into 1 LLM call; assembler OFF (baseline/legacy behavior)
    calls the LLM once per fragment."""

    async def run(enabled: bool) -> dict:
        llm = StaticLLMProvider()
        engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
        session = LiveMeetingSession(engine, max_llm_concurrency=1)
        assembler = UtteranceAssembler(UtteranceAssemblerConfig(enabled=enabled, finalization_delay_ms=500))

        semantic_utterances = []
        clock = 0.0
        for segment in continuous_speech_segments():
            clock += 0.1
            semantic_utterances.extend(assembler.push(segment, now=clock).finalized)
        clock += 0.6
        semantic_utterances.extend(assembler.flush_expired(now=clock).finalized)

        for semantic in semantic_utterances:
            await session.ingest_semantic_utterance(semantic, speaker="REMOTE")
        await session.drain()

        return {
            "transcript_segments": 3,
            "semantic_utterances": len(semantic_utterances),
            "llm_calls": len(llm.prompts),
            "merged_segments": sum(max(0, s.segment_count - 1) for s in semantic_utterances),
        }

    off = await run(enabled=False)
    on = await run(enabled=True)

    assert off == {"transcript_segments": 3, "semantic_utterances": 3, "llm_calls": 3, "merged_segments": 0}
    assert on == {"transcript_segments": 3, "semantic_utterances": 1, "llm_calls": 1, "merged_segments": 2}
    assert on["llm_calls"] < off["llm_calls"]
