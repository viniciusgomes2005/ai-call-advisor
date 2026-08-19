from __future__ import annotations

import abc
import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ASRSegment:
    speaker: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    latency_ms: int | None = None


class ASRProvider(abc.ABC):
    @abc.abstractmethod
    async def process_audio_file(self, path: Path, language: str | None = None, speaker: str = "UNKNOWN") -> list[ASRSegment]:
        raise NotImplementedError

    @abc.abstractmethod
    async def process_audio(self, data: bytes, suffix: str = ".wav", language: str | None = None, speaker: str = "UNKNOWN") -> list[ASRSegment]:
        raise NotImplementedError

    @abc.abstractmethod
    def status(self) -> str:
        raise NotImplementedError


class FasterWhisperProvider(ASRProvider):
    def __init__(self, model_name: str = "small", device: str = "auto", compute_type: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_error: Exception | None = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel

            kwargs = {}
            if self.compute_type != "auto":
                kwargs["compute_type"] = self.compute_type
            self._model = WhisperModel(self.model_name, device=self.device, **kwargs)
            return self._model
        except Exception as exc:
            self._load_error = exc
            raise

    def status(self) -> str:
        if self._model is not None:
            return "ok"
        if self._load_error is not None:
            return "error"
        try:
            import faster_whisper  # noqa: F401

            return "ok"
        except Exception:
            return "unavailable"

    async def process_audio_file(self, path: Path, language: str | None = None, speaker: str = "UNKNOWN") -> list[ASRSegment]:
        return await asyncio.to_thread(self._process_sync, path, language, speaker)

    async def process_audio(self, data: bytes, suffix: str = ".wav", language: str | None = None, speaker: str = "UNKNOWN") -> list[ASRSegment]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
            file.write(data)
            temp_path = Path(file.name)
        try:
            return await self.process_audio_file(temp_path, language=language, speaker=speaker)
        finally:
            temp_path.unlink(missing_ok=True)

    def _process_sync(self, path: Path, language: str | None, speaker: str) -> list[ASRSegment]:
        start = time.perf_counter()
        model = self._ensure_model()
        segments, _info = model.transcribe(str(path), language=language, vad_filter=True)
        output: list[ASRSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                output.append(
                    ASRSegment(
                        speaker=speaker,
                        text=text,
                        start_ms=int(segment.start * 1000),
                        end_ms=int(segment.end * 1000),
                        latency_ms=int((time.perf_counter() - start) * 1000),
                    )
                )
        return output

