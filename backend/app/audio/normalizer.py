from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class NormalizedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class AudioNormalizer:
    target_sample_rate = TARGET_SAMPLE_RATE

    def from_pcm16(self, data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> NormalizedAudio:
        if not data:
            raise ValueError("empty audio")
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            raise ValueError("empty audio")
        if sample_rate != self.target_sample_rate:
            samples = self._resample_linear(samples, sample_rate, self.target_sample_rate)
        return NormalizedAudio(
            samples=samples.astype(np.float32, copy=False),
            sample_rate=self.target_sample_rate,
            duration_seconds=float(samples.size / self.target_sample_rate),
        )

    def from_file(self, path: Path) -> NormalizedAudio:
        try:
            return self._from_media_file(path)
        except ImportError:
            if path.suffix.lower() == ".wav":
                return self._from_wav_stdlib(path)
            raise

    def from_bytes(self, data: bytes, suffix: str = ".wav") -> NormalizedAudio:
        if suffix.lower() == ".pcm":
            return self.from_pcm16(data)
        try:
            import av
        except ImportError:
            if suffix.lower() == ".wav":
                return self._from_wav_bytes_stdlib(data)
            raise

        with av.open(io.BytesIO(data), format=self._av_format_for_suffix(suffix)) as container:
            return self._decode_container(container, f"audio{suffix}")

    def _from_media_file(self, path: Path) -> NormalizedAudio:
        import av

        with av.open(str(path)) as container:
            return self._decode_container(container, str(path))

    def _decode_container(self, container, source_name: str) -> NormalizedAudio:
        chunks: list[np.ndarray] = []
        input_rate: int | None = None
        for frame in container.decode(audio=0):
            input_rate = int(frame.sample_rate)
            array = frame.to_ndarray()
            if array.ndim == 2:
                array = array.mean(axis=0)
            if np.issubdtype(array.dtype, np.integer):
                max_value = float(np.iinfo(array.dtype).max)
                array = array.astype(np.float32) / max_value
            else:
                array = array.astype(np.float32)
            chunks.append(array.reshape(-1))
        if not chunks:
            raise ValueError(f"no audio frames decoded from {source_name}")
        samples = np.concatenate(chunks).astype(np.float32, copy=False)
        rate = input_rate or self.target_sample_rate
        if rate != self.target_sample_rate:
            samples = self._resample_linear(samples, rate, self.target_sample_rate)
        return NormalizedAudio(
            samples=samples,
            sample_rate=self.target_sample_rate,
            duration_seconds=float(samples.size / self.target_sample_rate),
        )

    def _from_wav_stdlib(self, path: Path) -> NormalizedAudio:
        return self._from_wav_bytes_stdlib(path.read_bytes())

    def _from_wav_bytes_stdlib(self, data: bytes) -> NormalizedAudio:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2:
            raise ValueError("only 16-bit PCM WAV is supported without PyAV")
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if sample_rate != self.target_sample_rate:
            samples = self._resample_linear(samples, sample_rate, self.target_sample_rate)
        return NormalizedAudio(
            samples=samples.astype(np.float32, copy=False),
            sample_rate=self.target_sample_rate,
            duration_seconds=float(samples.size / self.target_sample_rate),
        )

    @staticmethod
    def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("invalid sample rate")
        if samples.size == 0 or source_rate == target_rate:
            return samples.astype(np.float32, copy=False)
        duration = samples.size / source_rate
        target_size = max(1, int(round(duration * target_rate)))
        old_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        new_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
        return np.interp(new_positions, old_positions, samples).astype(np.float32)

    @staticmethod
    def _av_format_for_suffix(suffix: str) -> str | None:
        suffix = suffix.lower().lstrip(".")
        if suffix in {"webm", "mp3", "wav"}:
            return suffix
        if suffix in {"m4a", "mp4"}:
            return "mp4"
        return None
