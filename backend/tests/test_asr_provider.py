from __future__ import annotations

import asyncio
import types
from pathlib import Path

import numpy as np
import pytest

from app.asr.provider import (
    FasterWhisperProvider,
    correct_transcript_terms,
    normalize_transcript_text,
    resolve_whisper_runtime,
    should_ignore_transcript,
)
from app.asr.streaming import AudioQueueItem, FinalizedAudio, SpeechSegmenter, SpeechStarted


def test_transcript_filter_ignores_common_whisper_hallucination() -> None:
    assert should_ignore_transcript("Legendas pela comunidade de Amara.org")


def test_transcript_filter_ignores_short_fillers() -> None:
    assert should_ignore_transcript("Hum!")


def test_transcript_filter_keeps_relevant_sentence() -> None:
    text = "Advisor, você recomenda dashboard web, widget flutuante ou extensão do Chrome?"

    assert normalize_transcript_text(f"  {text}  ") == text
    assert not should_ignore_transcript(text)


def test_transcript_terms_are_corrected_for_product_vocabulary() -> None:
    assert correct_transcript_terms("Adivisor, você recomenda With Get Funculate") == (
        "Advisor, você recomenda widget flutuante"
    )
    assert correct_transcript_terms("O estacion do Chrome para o MVP") == "extensão do Chrome para o MVP"


def test_cpu_runtime_configuration(monkeypatch) -> None:
    monkeypatch.setattr("app.asr.provider.cuda_device_count", lambda: 0)
    monkeypatch.setattr("app.asr.provider.cuda_runtime_loadable", lambda: False)

    runtime = resolve_whisper_runtime("auto", "auto")

    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"


def test_cuda_runtime_configuration(monkeypatch) -> None:
    monkeypatch.setattr("app.asr.provider.cuda_device_count", lambda: 1)
    monkeypatch.setattr("app.asr.provider.cuda_runtime_loadable", lambda: True)

    runtime = resolve_whisper_runtime("auto", "auto")

    assert runtime.device == "cuda"
    assert runtime.compute_type == "float16"


async def test_provider_transcribes_pcm_with_mock_model(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, model_name: str, device: str, compute_type: str):
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type

        def transcribe(self, audio, **kwargs):
            assert isinstance(audio, np.ndarray)
            segment = types.SimpleNamespace(
                start=0.0,
                end=1.0,
                text=" Atualmente utilizamos SAP ECC. ",
                avg_logprob=-0.1,
                no_speech_prob=0.1,
                compression_ratio=1.0,
            )
            info = types.SimpleNamespace(language="pt", duration=1.0)
            return iter([segment]), info

    monkeypatch.setattr("app.asr.provider.cuda_device_count", lambda: 0)
    monkeypatch.setattr("app.asr.provider.cuda_runtime_loadable", lambda: False)
    monkeypatch.setattr(FasterWhisperProvider, "_whisper_model_class", staticmethod(lambda: FakeModel))
    provider = FasterWhisperProvider()
    pcm = (np.zeros(16_000, dtype=np.int16)).tobytes()

    result = await provider.transcribe_audio_chunk(pcm)

    assert result.language == "pt"
    assert result.audio_duration_seconds == 1.0
    assert result.processing_time_ms >= 0
    assert result.real_time_factor is not None
    assert result.segments[0].text == "Atualmente utilizamos SAP ECC."
    assert provider.status_payload().status == "ready"


async def test_provider_rejects_empty_audio() -> None:
    provider = FasterWhisperProvider()

    with pytest.raises(ValueError, match="empty audio"):
        await provider.transcribe_audio_chunk(b"")


async def test_provider_rejects_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "audio.txt"
    path.write_text("not audio", encoding="utf-8")
    provider = FasterWhisperProvider()

    with pytest.raises(ValueError, match="unsupported audio file type"):
        await provider.transcribe_file(path)


def test_speech_segmenter_emits_started_and_final_audio() -> None:
    segmenter = SpeechSegmenter(
        min_speech_ms=100,
        silence_end_ms=100,
        max_utterance_ms=15_000,
        rms_threshold=0.01,
    )
    speech = (np.ones(3200, dtype=np.int16) * 2000).tobytes()
    silence = (np.zeros(3200, dtype=np.int16)).tobytes()

    events = []
    events.extend(segmenter.push(AudioQueueItem(source="MIC", data=speech)))
    events.extend(segmenter.push(AudioQueueItem(source="MIC", data=silence)))

    assert any(isinstance(event, SpeechStarted) for event in events)
    final = [event for event in events if isinstance(event, FinalizedAudio)]
    assert len(final) == 1
    assert final[0].source == "MIC"
    assert final[0].duration_seconds > 0


async def test_audio_queue_preserves_chunk_order() -> None:
    queue: asyncio.Queue[AudioQueueItem] = asyncio.Queue()
    await queue.put(AudioQueueItem(source="MIC", data=b"first"))
    await queue.put(AudioQueueItem(source="MIC", data=b"second"))

    first = await queue.get()
    second = await queue.get()

    assert first.data == b"first"
    assert second.data == b"second"
