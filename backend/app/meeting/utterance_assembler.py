from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from app.schemas import SemanticUtterance, TranscriptSegment


AssemblyReason = Literal[
    "silence_timeout",
    "large_gap",
    "hard_duration_limit",
    "hard_char_limit",
    "source_change",
    "manual_flush",
    "meeting_end",
    "assembler_disabled",
]


@dataclass(slots=True)
class UtteranceAssemblerConfig:
    enabled: bool = True
    merge_max_gap_ms: int = 1200
    hard_max_duration_ms: int = 20000
    hard_max_chars: int = 500
    finalization_delay_ms: int = 500


@dataclass(slots=True)
class AssemblyLog:
    event: str
    payload: dict


@dataclass(slots=True)
class AssemblyResult:
    updated: SemanticUtterance | None = None
    finalized: list[SemanticUtterance] = field(default_factory=list)
    logs: list[AssemblyLog] = field(default_factory=list)


@dataclass(slots=True)
class _Buffer:
    semantic_id: str
    source: str
    segments: list[TranscriptSegment]
    started_at: float
    updated_at: float


class TranscriptOrderingBuffer:
    def __init__(self):
        self._expected: dict[str, int] = {}
        self._pending: dict[str, dict[int, list[TranscriptSegment]]] = {}
        self._first_pending_at: dict[str, dict[int, float]] = {}

    def push_batch(
        self,
        source: str,
        sequence: int,
        segments: list[TranscriptSegment],
        *,
        received_at: float | None = None,
    ) -> list[TranscriptSegment]:
        pending = self._pending.setdefault(source, {})
        pending[sequence] = sorted(segments, key=lambda item: (item.start, item.end, item.id))
        self._first_pending_at.setdefault(source, {})[sequence] = received_at if received_at is not None else time.perf_counter()
        expected = self._expected.setdefault(source, 0)
        ready: list[TranscriptSegment] = []
        while expected in pending:
            ready.extend(pending.pop(expected))
            self._first_pending_at[source].pop(expected, None)
            expected += 1
        self._expected[source] = expected
        return ready

    def pending_count(self) -> int:
        return sum(len(batch) for batches in self._pending.values() for batch in batches.values())

    def oldest_pending_age_ms(self, now: float | None = None) -> int:
        clock = now if now is not None else time.perf_counter()
        timestamps = [stamp for source_items in self._first_pending_at.values() for stamp in source_items.values()]
        if not timestamps:
            return 0
        return max(0, int((clock - min(timestamps)) * 1000))


class UtteranceAssembler:
    continuation_words = {
        "and",
        "but",
        "because",
        "for",
        "with",
        "that",
        "when",
        "if",
        "then",
        "so",
        "or",
        "to",
        "of",
        "the",
        "e",
        "mas",
        "porque",
        "para",
        "pra",
        "com",
        "que",
        "quando",
        "se",
        "entao",
        "então",
        "ou",
        "de",
        "da",
        "do",
        "dos",
        "das",
        "um",
        "uma",
    }

    terminal_punctuation = (".", "?", "!", "…")
    continuation_punctuation = (",", ":", ";", "-", "–")

    def __init__(self, config: UtteranceAssemblerConfig | None = None):
        self.config = config or UtteranceAssemblerConfig()
        self._buffers: dict[str, _Buffer] = {}
        self._next_id = 1
        self._segments_per_utterance: list[int] = []
        self._assembly_latencies_ms: list[float] = []
        self.number_of_merged_segments = 0
        self.number_of_single_segment_utterances = 0

    def push(self, segment: TranscriptSegment, *, now: float | None = None) -> AssemblyResult:
        clock = now if now is not None else time.perf_counter()
        if not self.config.enabled:
            semantic = self._semantic_from_segments(self._new_utterance_id(), [segment], "assembler_disabled", clock)
            return AssemblyResult(finalized=[semantic])

        source = segment.source
        current = self._buffers.get(source)
        if current is None:
            current = _Buffer(
                semantic_id=self._new_utterance_id(),
                source=source,
                segments=[segment],
                started_at=clock,
                updated_at=clock,
            )
            self._buffers[source] = current
            semantic = self._semantic_from_segments(current.semantic_id, [segment], None, clock)
            return AssemblyResult(
                updated=semantic,
                logs=[AssemblyLog("utterance.assembly.started", self._log_payload(semantic))],
            )

        finalized: list[SemanticUtterance] = []
        logs: list[AssemblyLog] = []
        if self._should_merge(current, segment):
            current.segments.append(segment)
            current.segments.sort(key=lambda item: (item.start, item.end, item.id))
            current.updated_at = clock
            updated = self._semantic_from_segments(current.semantic_id, current.segments, None, clock)
            logs.append(AssemblyLog("utterance.segment.merged", self._log_payload(updated, segment_id=segment.id)))
            hard_reason = self._hard_limit_reason(current.segments)
            if hard_reason:
                finalized.append(self._finalize(source, hard_reason, clock))
                updated = None
            return AssemblyResult(updated=updated, finalized=finalized, logs=logs)

        reason = self._split_reason(current, segment)
        finalized.append(self._finalize(source, reason, clock))
        current = _Buffer(
            semantic_id=self._new_utterance_id(),
            source=source,
            segments=[segment],
            started_at=clock,
            updated_at=clock,
        )
        self._buffers[source] = current
        updated = self._semantic_from_segments(current.semantic_id, [segment], None, clock)
        logs.append(AssemblyLog("utterance.assembly.started", self._log_payload(updated)))
        return AssemblyResult(updated=updated, finalized=finalized, logs=logs)

    def flush_expired(self, *, now: float | None = None) -> AssemblyResult:
        clock = now if now is not None else time.perf_counter()
        finalized: list[SemanticUtterance] = []
        for source, current in list(self._buffers.items()):
            age_ms = int((clock - current.updated_at) * 1000)
            if age_ms >= self.config.finalization_delay_ms:
                finalized.append(self._finalize(source, "silence_timeout", clock))
        return AssemblyResult(finalized=finalized)

    def flush_source(self, source: str, reason: AssemblyReason = "manual_flush", *, now: float | None = None) -> AssemblyResult:
        if source not in self._buffers:
            return AssemblyResult()
        clock = now if now is not None else time.perf_counter()
        return AssemblyResult(finalized=[self._finalize(source, reason, clock)])

    def flush_all(self, reason: AssemblyReason = "meeting_end", *, now: float | None = None) -> AssemblyResult:
        clock = now if now is not None else time.perf_counter()
        finalized = [self._finalize(source, reason, clock) for source in list(self._buffers)]
        return AssemblyResult(finalized=finalized)

    def pending_transcript_segments(self) -> int:
        return sum(len(buffer.segments) for buffer in self._buffers.values())

    def oldest_pending_segment_age_ms(self, *, now: float | None = None) -> int:
        if not self._buffers:
            return 0
        clock = now if now is not None else time.perf_counter()
        oldest = min(buffer.started_at for buffer in self._buffers.values())
        return max(0, int((clock - oldest) * 1000))

    def metrics(self) -> dict[str, float | int | None]:
        return {
            "average_segments_per_utterance": (
                statistics.fmean(self._segments_per_utterance) if self._segments_per_utterance else None
            ),
            "p50_segments_per_utterance": self._percentile(self._segments_per_utterance, 50),
            "p95_segments_per_utterance": self._percentile(self._segments_per_utterance, 95),
            "number_of_merged_segments": self.number_of_merged_segments,
            "number_of_single_segment_utterances": self.number_of_single_segment_utterances,
            "utterance_assembly_latency_ms": (
                statistics.fmean(self._assembly_latencies_ms) if self._assembly_latencies_ms else None
            ),
        }

    @classmethod
    def looks_incomplete(cls, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if lowered.endswith(cls.continuation_punctuation):
            return True
        if lowered.endswith(cls.terminal_punctuation):
            return False
        words = [word.strip(".,!?;:()[]{}\"'").lower() for word in lowered.split()]
        if not words:
            return False
        return words[-1] in cls.continuation_words or len(words) <= 3

    def _should_merge(self, current: _Buffer, segment: TranscriptSegment) -> bool:
        previous = current.segments[-1]
        gap_ms = int(max(0.0, segment.start - previous.end) * 1000)
        if gap_ms <= self.config.merge_max_gap_ms:
            return True
        if self.looks_incomplete(previous.text) and gap_ms <= int(self.config.merge_max_gap_ms * 1.5):
            return True
        return False

    def _split_reason(self, current: _Buffer, segment: TranscriptSegment) -> AssemblyReason:
        hard_reason = self._hard_limit_reason(current.segments)
        if hard_reason:
            return hard_reason
        previous = current.segments[-1]
        gap_ms = int(max(0.0, segment.start - previous.end) * 1000)
        if gap_ms > self.config.merge_max_gap_ms:
            return "large_gap"
        return "silence_timeout"

    def _hard_limit_reason(self, segments: list[TranscriptSegment]) -> AssemblyReason | None:
        if not segments:
            return None
        duration_ms = int(max(0.0, segments[-1].end - segments[0].start) * 1000)
        if duration_ms >= self.config.hard_max_duration_ms:
            return "hard_duration_limit"
        if len(self._join_text(segments)) >= self.config.hard_max_chars:
            return "hard_char_limit"
        return None

    def _finalize(self, source: str, reason: AssemblyReason, now: float) -> SemanticUtterance:
        current = self._buffers.pop(source)
        semantic = self._semantic_from_segments(current.semantic_id, current.segments, reason, now)
        count = len(current.segments)
        self._segments_per_utterance.append(count)
        if count == 1:
            self.number_of_single_segment_utterances += 1
        else:
            self.number_of_merged_segments += count - 1
        if semantic.assembly_latency_ms is not None:
            self._assembly_latencies_ms.append(semantic.assembly_latency_ms)
        return semantic

    def _semantic_from_segments(
        self,
        semantic_id: str,
        segments: list[TranscriptSegment],
        reason: AssemblyReason | None,
        now: float,
    ) -> SemanticUtterance:
        created_at = min((segment.created_at for segment in segments), default=datetime.now(timezone.utc))
        updated_at = max((segment.created_at for segment in segments), default=created_at)
        latency_ms = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() * 1000)
        asr_latencies = [segment.asr_latency_ms for segment in segments if segment.asr_latency_ms is not None]
        return SemanticUtterance(
            id=semantic_id,
            source=segments[0].source,
            text=self._join_text(segments),
            start=segments[0].start,
            end=segments[-1].end,
            segment_ids=[segment.id for segment in segments],
            created_at=created_at,
            updated_at=updated_at,
            language=next((segment.language for segment in segments if segment.language), None),
            asr_latency_ms=max(asr_latencies) if asr_latencies else None,
            audio_finalize_latency_ms=None,
            assembly_reason=reason,
            assembly_latency_ms=latency_ms if reason else None,
            segment_count=len(segments),
        )

    def _new_utterance_id(self) -> str:
        value = f"utt_{self._next_id}"
        self._next_id += 1
        return value

    def _log_payload(self, semantic: SemanticUtterance, **extra: object) -> dict:
        payload = semantic.model_dump(mode="json")
        payload.update(extra)
        return payload

    @staticmethod
    def _join_text(segments: list[TranscriptSegment]) -> str:
        text = ""
        for segment in segments:
            piece = segment.text.strip()
            if not piece:
                continue
            if not text:
                text = piece
            elif piece[:1] in ".,!?;:":
                text += piece
            else:
                text += f" {piece}"
        return text.strip()

    @staticmethod
    def _percentile(values: list[int], percentile: int) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        index = round((percentile / 100) * (len(ordered) - 1))
        return float(ordered[index])
