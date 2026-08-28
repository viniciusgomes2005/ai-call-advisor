from __future__ import annotations

import json
from pathlib import Path

from app.schemas import InterventionCategory, LLMDecision, TranscriptSegment, Utterance
from app.services.replay import ReplayService
from app.schemas import ReplayRequest

from .conftest import HeuristicLLMProvider, StaticLLMProvider, make_test_engine


async def test_keep_silence_behavior(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    decision = await engine.ingest_utterance(Utterance(id=1, speaker="Alice", text="Design review is Thursday."))
    assert decision.category == InterventionCategory.KEEP_SILENCE
    assert decision.response is None


async def test_explicit_cue_fixture(tmp_path, prompt_path):
    case = json.loads(Path("../benchmarks/fixtures/explicit_cue.json").read_text(encoding="utf-8"))
    from app.schemas import BenchmarkCase

    fixture = BenchmarkCase.model_validate(case)
    engine = make_test_engine(tmp_path, fixture.delegate, HeuristicLLMProvider(), prompt_path)
    decisions = await ReplayService(engine).run(ReplayRequest(delegate=fixture.delegate, utterances=fixture.utterances))
    assert decisions[-1].category == InterventionCategory.EXPLICIT_CUE


async def test_implicit_cue_fixture(tmp_path, prompt_path):
    case = json.loads(Path("../benchmarks/fixtures/implicit_cue.json").read_text(encoding="utf-8"))
    from app.schemas import BenchmarkCase

    fixture = BenchmarkCase.model_validate(case)
    engine = make_test_engine(tmp_path, fixture.delegate, HeuristicLLMProvider(), prompt_path)
    decisions = await ReplayService(engine).run(ReplayRequest(delegate=fixture.delegate, utterances=fixture.utterances))
    assert decisions[-1].category == InterventionCategory.IMPLICIT_CUE


async def test_chime_in_fixture(tmp_path, prompt_path):
    case = json.loads(Path("../benchmarks/fixtures/chime_in.json").read_text(encoding="utf-8"))
    from app.schemas import BenchmarkCase

    fixture = BenchmarkCase.model_validate(case)
    engine = make_test_engine(tmp_path, fixture.delegate, HeuristicLLMProvider(), prompt_path)
    decisions = await ReplayService(engine).run(ReplayRequest(delegate=fixture.delegate, utterances=fixture.utterances))
    assert decisions[-1].category == InterventionCategory.CHIME_IN


async def test_deduplication_and_cooldown(tmp_path, delegate, prompt_path):
    decision = LLMDecision(
        category=InterventionCategory.CHIME_IN,
        should_intervene=True,
        confidence=0.8,
        response="Have we measured latency under concurrent requests?",
        reason="latency",
        trigger_utterance_ids=[1],
    )
    llm = StaticLLMProvider(decision)
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path, filter_enabled=True, cooldown_seconds=60)
    first = await engine.ingest_utterance(Utterance(id=1, speaker="Alice", text="Latency is bad."))
    second = await engine.ingest_utterance(Utterance(id=2, speaker="Carol", text="Latency remains bad."))
    assert first.displayed is True
    assert second.filtered is True
    assert second.category == InterventionCategory.KEEP_SILENCE


async def test_context_truncation(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path, context_max_utterances=2)
    for idx in range(1, 5):
        await engine.ingest_utterance(Utterance(id=idx, speaker="Alice", text=f"message {idx}"))
    last_prompt = llm.prompts[-1]
    assert "[1] Alice" not in last_prompt
    assert "[3] Alice" in last_prompt
    assert "[4] Alice" in last_prompt
    assert "Earlier transcript summary" in last_prompt


async def test_replay_never_leaks_future_utterances(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    request = ReplayRequest(
        delegate=delegate,
        utterances=[
            Utterance(id=1, speaker="Alice", text="current status"),
            Utterance(id=2, speaker="Carol", text="future secret utterance"),
        ],
    )
    await ReplayService(engine).run(request)
    assert "future secret utterance" not in llm.prompts[0]
    assert "future secret utterance" in llm.prompts[1]


async def test_answer_question_uses_meeting_context(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    await engine.ingest_utterance(Utterance(id=1, speaker="Alice", text="The voice feature is blocked by latency."))

    answer = await engine.answer_question("What is blocking the voice feature?")

    assert answer.answer == "Static answer from meeting context."
    assert "The voice feature is blocked by latency." in llm.prompts[-1]
    assert "Question: What is blocking the voice feature?" in llm.prompts[-1]


async def test_meeting_insights_are_detected(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)

    await engine.ingest_utterance(
        Utterance(id=1, speaker="Alice", text="Fechado, vamos seguir com o plano e eu vou validar a latência.")
    )
    await engine.ingest_utterance(Utterance(id=2, speaker="Carol", text="Quem vai apresentar os resultados?"))

    insight_types = {insight.type for insight in engine.state.insights}
    assert "DECISION" in insight_types
    assert "ACTION_ITEM" in insight_types
    assert "OPEN_QUESTION" in insight_types


async def test_session_persistence(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    await engine.ingest_utterance(Utterance(id=1, speaker="Alice", text="Quem vai validar a latência?"))
    session_dir = tmp_path / "sessions" / engine.state.meeting_id
    assert (session_dir / "meeting.json").exists()
    assert (session_dir / "utterances.jsonl").exists()
    assert (session_dir / "decisions.jsonl").exists()
    assert (session_dir / "insights.jsonl").exists()
    assert (session_dir / "metrics.json").exists()


async def test_asr_latency_is_carried_into_decision(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)

    decision = await engine.ingest_utterance(
        Utterance(
            id=1,
            speaker="REMOTE",
            text="Atualmente utilizamos SAP ECC.",
            source="TAB_AUDIO",
            asr_latency_ms=700,
            audio_finalize_latency_ms=20,
        )
    )

    assert decision.llm_latency_ms == 3
    assert decision.total_suggestion_latency_ms is not None
    assert decision.total_suggestion_latency_ms >= 720


async def test_session_asr_logging(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)

    engine.event_logger.log_transcript_segment(
        engine.state.meeting_id,
        TranscriptSegment(start=0, end=1, text="SAP ECC", language="pt"),
        {"speaker": "REMOTE", "source": "TAB_AUDIO", "asr_latency_ms": 100},
    )
    engine.event_logger.log_asr_metrics(
        engine.state.meeting_id,
        {"utterance_id": "utt_1", "audio_duration_ms": 1000, "asr_latency_ms": 100},
    )

    session_dir = tmp_path / "sessions" / engine.state.meeting_id
    assert (session_dir / "transcript.jsonl").read_text(encoding="utf-8")
    assert (session_dir / "asr_metrics.jsonl").read_text(encoding="utf-8")
