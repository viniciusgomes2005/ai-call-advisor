from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.schemas import AudioFinalizationReason


@dataclass(slots=True)
class AudioQueueItem:
    source: str
    data: bytes
    sample_rate: int = 16_000
    received_at: float = field(default_factory=time.perf_counter)


@dataclass(slots=True)
class SpeechStarted:
    source: str
    start_seconds: float


@dataclass(slots=True)
class FinalizedAudio:
    source: str
    data: bytes
    sample_rate: int
    start_seconds: float
    end_seconds: float
    finalization_reason: AudioFinalizationReason = AudioFinalizationReason.SILENCE
    finalized_at: float = field(default_factory=time.perf_counter)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


class SpeechSegmenter:
    def __init__(
        self,
        *,
        min_speech_ms: int,
        silence_end_ms: int,
        max_utterance_ms: int,
        rms_threshold: float,
        sample_rate: int = 16_000,
    ):
        self.min_speech_ms = min_speech_ms
        self.silence_end_ms = silence_end_ms
        self.max_utterance_ms = max_utterance_ms
        self.rms_threshold = rms_threshold
        self.sample_rate = sample_rate
        self.timeline_seconds = 0.0
        self.in_speech = False
        self.speech_start_seconds = 0.0
        self.speech_ms = 0
        self.trailing_silence_ms = 0
        self.buffer = bytearray()

    def push(self, item: AudioQueueItem) -> list[SpeechStarted | FinalizedAudio]:
        data = item.data
        if not data:
            return []
        chunk_seconds = (len(data) / 2) / item.sample_rate
        chunk_ms = int(chunk_seconds * 1000)
        rms = self._rms(data)
        is_speech = rms >= self.rms_threshold
        events: list[SpeechStarted | FinalizedAudio] = []

        if is_speech and not self.in_speech:
            self.in_speech = True
            self.speech_start_seconds = self.timeline_seconds
            self.speech_ms = 0
            self.trailing_silence_ms = 0
            self.buffer.clear()
            events.append(SpeechStarted(source=item.source, start_seconds=self.speech_start_seconds))

        if self.in_speech:
            self.buffer.extend(data)
            self.speech_ms += chunk_ms
            if is_speech:
                self.trailing_silence_ms = 0
            else:
                self.trailing_silence_ms += chunk_ms

            utterance_ms = int((self.timeline_seconds + chunk_seconds - self.speech_start_seconds) * 1000)
            is_silence_end = self.speech_ms >= self.min_speech_ms and self.trailing_silence_ms >= self.silence_end_ms
            is_max_duration = utterance_ms >= self.max_utterance_ms
            if is_silence_end or is_max_duration:
                # Silence takes priority: if both conditions land on the same chunk, it is a
                # real end of speech, not an artificial cut - the assembler should be allowed
                # to finalize the semantic utterance.
                reason = AudioFinalizationReason.SILENCE if is_silence_end else AudioFinalizationReason.MAX_DURATION
                events.append(
                    self.finalize(item.source, item.sample_rate, self.timeline_seconds + chunk_seconds, reason)
                )

        self.timeline_seconds += chunk_seconds
        return events

    def flush(
        self,
        source: str,
        sample_rate: int = 16_000,
        reason: AudioFinalizationReason = AudioFinalizationReason.MANUAL_FLUSH,
    ) -> FinalizedAudio | None:
        if not self.in_speech or not self.buffer:
            return None
        return self.finalize(source, sample_rate, self.timeline_seconds, reason)

    def finalize(
        self,
        source: str,
        sample_rate: int,
        end_seconds: float,
        reason: AudioFinalizationReason = AudioFinalizationReason.SILENCE,
    ) -> FinalizedAudio:
        event = FinalizedAudio(
            source=source,
            data=bytes(self.buffer),
            sample_rate=sample_rate,
            start_seconds=self.speech_start_seconds,
            end_seconds=end_seconds,
            finalization_reason=reason,
        )
        self.in_speech = False
        self.speech_ms = 0
        self.trailing_silence_ms = 0
        self.buffer.clear()
        return event

    @staticmethod
    def _rms(data: bytes) -> float:
        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        values = samples.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(values * values)))
