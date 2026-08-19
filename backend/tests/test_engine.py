from __future__ import annotations

import json
from pathlib import Path

from app.schemas import InterventionCategory, LLMDecision, Utterance
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


async def test_session_persistence(tmp_path, delegate, prompt_path):
    llm = StaticLLMProvider()
    engine = make_test_engine(tmp_path, delegate, llm, prompt_path)
    await engine.ingest_utterance(Utterance(id=1, speaker="Alice", text="hello"))
    session_dir = tmp_path / "sessions" / engine.state.meeting_id
    assert (session_dir / "meeting.json").exists()
    assert (session_dir / "utterances.jsonl").exists()
    assert (session_dir / "decisions.jsonl").exists()
    assert (session_dir / "metrics.json").exists()

