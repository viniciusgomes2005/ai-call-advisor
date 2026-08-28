from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ScreenFrame, ScreenFrameIngest, VisualAnalysis
from app.services.visual import (
    build_screen_frame,
    build_visual_context,
    decode_frame_data,
    screen_frame_filename,
)
from app.vision import NullVisionProvider

from .conftest import StaticLLMProvider, make_test_engine


def make_frame(*, frame_id: str = "frame_0", timestamp: float, change_score: float | None = None) -> ScreenFrame:
    return ScreenFrame(id=frame_id, timestamp=timestamp, width=1280, height=720, change_score=change_score)


# ---------------------------------------------------------------------------
# Pure helpers (app/services/visual.py)
# ---------------------------------------------------------------------------


def test_screen_frame_filename_is_deterministic_and_maps_known_mime_types():
    assert screen_frame_filename(1, "image/jpeg") == "frame_000001.jpg"
    assert screen_frame_filename(2, "image/webp") == "frame_000002.webp"
    assert screen_frame_filename(3, "image/png") == "frame_000003.png"
    assert screen_frame_filename(4, "application/octet-stream") == "frame_000004.jpg"  # unknown -> safe default


def test_decode_frame_data_round_trips_valid_base64():
    import base64

    raw = b"not-really-a-jpeg-but-bytes"
    encoded = base64.b64encode(raw).decode("ascii")

    assert decode_frame_data(encoded) == raw


def test_decode_frame_data_rejects_invalid_base64():
    with pytest.raises(ValueError, match="invalid base64"):
        decode_frame_data("not valid base64!!!")


def test_build_screen_frame_defaults_captured_at_and_source():
    frame = build_screen_frame(
        frame_id="frame_1", timestamp=5.0, captured_at=None, mime_type="image/jpeg", width=100, height=50, change_score=0.5
    )
    assert frame.id == "frame_1"
    assert frame.source == "SCREEN_SHARE"
    assert frame.captured_at is not None
    assert frame.change_score == 0.5


def test_build_visual_context_stays_empty_without_a_vision_analysis():
    frame = make_frame(timestamp=12.3)

    visual = build_visual_context(visual_id="visual_0", frame=frame, analysis=None)

    assert visual.frame_id == frame.id
    assert visual.timestamp == 12.3
    assert visual.summary is None
    assert visual.visible_text == []
    assert visual.entities == []
    assert visual.systems == []
    assert visual.numbers == []


def test_build_visual_context_merges_a_provided_analysis():
    frame = make_frame(timestamp=12.3)
    analysis = VisualAnalysis(summary="a slide", visible_text=["Total: 42"], systems=["SAP"])

    visual = build_visual_context(visual_id="visual_0", frame=frame, analysis=analysis)

    assert visual.summary == "a slide"
    assert visual.visible_text == ["Total: 42"]
    assert visual.systems == ["SAP"]


async def test_null_vision_provider_never_produces_analysis():
    frame = make_frame(timestamp=1.0)

    result = await NullVisionProvider().analyze_frame(b"jpeg-bytes", frame)

    assert result is None


# ---------------------------------------------------------------------------
# ScreenFrameIngest validation (the inbound screen.frame WS message)
# ---------------------------------------------------------------------------


def test_screen_frame_ingest_accepts_a_valid_payload():
    payload = ScreenFrameIngest.model_validate(
        {
            "type": "screen.frame",
            "timestamp": 130.4,
            "captured_at": "2024-01-01T00:00:00Z",
            "mime_type": "image/jpeg",
            "width": 1280,
            "height": 720,
            "change_score": 0.18,
            "data": "aGVsbG8=",
        }
    )
    assert payload.timestamp == 130.4
    assert payload.width == 1280


@pytest.mark.parametrize(
    "overrides",
    [
        {"data": ""},
        {"width": 0},
        {"height": -1},
    ],
)
def test_screen_frame_ingest_rejects_invalid_payloads(overrides):
    base = {"timestamp": 1.0, "width": 100, "height": 100, "data": "aGVsbG8="}
    base.update(overrides)
    with pytest.raises(ValidationError):
        ScreenFrameIngest.model_validate(base)


# ---------------------------------------------------------------------------
# MeetingEngine.record_visual_context / MeetingState
# ---------------------------------------------------------------------------


async def test_record_visual_context_appends_to_state(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    frame = make_frame(timestamp=10.0)
    visual = build_visual_context(visual_id="visual_0", frame=frame)

    await engine.record_visual_context(visual)

    assert [v.id for v in engine.state.recent_visual_contexts] == ["visual_0"]


async def test_record_visual_context_caps_recent_list_and_evicts_oldest(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    engine.visual_context_max_recent = 2

    for index in range(3):
        frame = make_frame(frame_id=f"frame_{index}", timestamp=float(index))
        visual = build_visual_context(visual_id=f"visual_{index}", frame=frame)
        await engine.record_visual_context(visual)

    ids = [v.id for v in engine.state.recent_visual_contexts]
    assert ids == ["visual_1", "visual_2"]  # oldest (visual_0) evicted, order preserved


def test_get_visual_context_near_finds_the_closest_within_tolerance(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    engine.state.recent_visual_contexts = [
        build_visual_context(visual_id="visual_far", frame=make_frame(frame_id="f1", timestamp=50.0)),
        build_visual_context(visual_id="visual_near", frame=make_frame(frame_id="f2", timestamp=130.0)),
        build_visual_context(visual_id="visual_farther", frame=make_frame(frame_id="f3", timestamp=200.0)),
    ]

    # An utterance spanning 128.1s to 132.4s - midpoint 130.25, closest is visual_near.
    found = engine.state.get_visual_context_near(128.1, 132.4, tolerance_seconds=5)
    assert found is not None
    assert found.id == "visual_near"


def test_get_visual_context_near_returns_none_outside_tolerance(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    engine.state.recent_visual_contexts = [
        build_visual_context(visual_id="visual_0", frame=make_frame(frame_id="f1", timestamp=10.0)),
    ]

    assert engine.state.get_visual_context_near(500.0, tolerance_seconds=5) is None


def test_get_visual_context_near_with_no_contexts_returns_none(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    assert engine.state.get_visual_context_near(10.0) is None


# ---------------------------------------------------------------------------
# EventLogger: raw image persistence gated by SAVE_SCREEN_FRAMES, logging, metrics
# ---------------------------------------------------------------------------


async def test_visual_context_logging_writes_jsonl_without_base64_image(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    frame = make_frame(timestamp=130.4, change_score=0.19)
    visual = build_visual_context(visual_id="visual_12", frame=frame)

    engine.event_logger.log_visual_context(engine.state.meeting_id, frame, visual)

    session_dir = tmp_path / "sessions" / engine.state.meeting_id
    log_path = session_dir / "visual_contexts.jsonl"
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip()
    assert "visual_12" in line
    assert "change_score" in line
    assert "0.19" in line
    assert "image_path" not in line  # no path was persisted (SAVE_SCREEN_FRAMES=false path)
    assert '"data"' not in line  # never a raw base64 image blob in the jsonl


def test_save_screen_frame_writes_deterministic_filename(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)

    path = engine.event_logger.save_screen_frame(engine.state.meeting_id, 1, b"jpeg-bytes", "image/jpeg")

    assert path.name == "frame_000001.jpg"
    assert path.read_bytes() == b"jpeg-bytes"
    assert path.parent.name == "frames"


async def test_save_screen_frames_disabled_means_no_file_is_ever_written(tmp_path, delegate, prompt_path):
    """Mirrors the actual gating in routes.py: when SAVE_SCREEN_FRAMES=false the
    caller simply never calls save_screen_frame - so the frames/ dir never appears."""
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    frame = make_frame(timestamp=1.0)
    visual = build_visual_context(visual_id="visual_0", frame=frame)

    save_screen_frames = False
    if save_screen_frames:
        engine.event_logger.save_screen_frame(engine.state.meeting_id, 0, b"jpeg-bytes", "image/jpeg")
    engine.event_logger.log_visual_context(engine.state.meeting_id, frame, visual)

    frames_dir = tmp_path / "sessions" / engine.state.meeting_id / "frames"
    assert not frames_dir.exists()


async def test_visual_context_metrics_accumulate(tmp_path, delegate, prompt_path):
    engine = make_test_engine(tmp_path, delegate, StaticLLMProvider(), prompt_path)
    for index, score in enumerate([0.1, 0.3]):
        frame = make_frame(frame_id=f"frame_{index}", timestamp=float(index), change_score=score)
        visual = build_visual_context(visual_id=f"visual_{index}", frame=frame)
        engine.event_logger.log_visual_context(engine.state.meeting_id, frame, visual)

    import json

    metrics = json.loads((tmp_path / "sessions" / engine.state.meeting_id / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["visual_context_count"] == 2
    assert metrics["screen_frames_accepted"] == 2
    assert metrics["average_change_score"] == pytest.approx(0.2)
