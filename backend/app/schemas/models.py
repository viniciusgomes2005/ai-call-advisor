from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InterventionCategory(StrEnum):
    EXPLICIT_CUE = "EXPLICIT_CUE"
    IMPLICIT_CUE = "IMPLICIT_CUE"
    CHIME_IN = "CHIME_IN"
    KEEP_SILENCE = "KEEP_SILENCE"


class MeetingInsightType(StrEnum):
    OPEN_QUESTION = "OPEN_QUESTION"
    ACTION_ITEM = "ACTION_ITEM"
    DECISION = "DECISION"


class ShareableInformation(BaseModel):
    context: str
    information: str


class DelegateProfile(BaseModel):
    name: str
    role: str
    meeting_intents: list[str] = Field(default_factory=list, alias="meeting_intent")
    shareable_information: list[ShareableInformation] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Utterance(BaseModel):
    id: int
    speaker: str
    text: str
    timestamp: datetime = Field(default_factory=utc_now)
    source: Literal[
        "REPLAY",
        "MANUAL",
        "FILE",
        "MIC",
        "TAB_AUDIO",
        "REMOTE",
        "REMOTE_AUDIO",
        "LOCAL_MIC",
        "LOCAL_MIC_AUDIO",
        "FILE_AUDIO",
        "UNKNOWN",
    ] = "UNKNOWN"
    start: float | None = None
    end: float | None = None
    language: str | None = None
    asr_latency_ms: int | None = None
    audio_finalize_latency_ms: int | None = None
    semantic_id: str | None = None
    segment_ids: list[str] = Field(default_factory=list)
    assembly_reason: str | None = None
    assembly_latency_ms: float | None = None

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("utterance text cannot be empty")
        return value


class LLMDecision(BaseModel):
    category: InterventionCategory
    should_intervene: bool
    confidence: float = Field(ge=0, le=1)
    response: str | None = None
    reason: str
    trigger_utterance_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_silence_contract(self) -> "LLMDecision":
        if self.category == InterventionCategory.KEEP_SILENCE:
            self.should_intervene = False
            self.response = None
        elif self.should_intervene and not self.response:
            raise ValueError("intervention decisions require a response")
        return self


class InterventionDecision(LLMDecision):
    utterance_id: int
    timestamp: datetime = Field(default_factory=utc_now)
    model: str | None = None
    prompt_version: str = "intervention_v1"
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_latency_ms: int | None = None
    pipeline_latency_ms: int | None = None
    total_suggestion_latency_ms: int | None = None
    stale: bool = False
    displayed: bool = False
    filtered: bool = False
    filter_reason: str | None = None


class PreviousIntervention(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    utterance_id: int
    category: InterventionCategory
    response: str


class MeetingInsight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: MeetingInsightType
    utterance_id: int
    speaker: str
    text: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime = Field(default_factory=utc_now)


class MeetingState(BaseModel):
    meeting_id: str = Field(default_factory=lambda: str(uuid4()))
    delegate: DelegateProfile
    utterances: list[Utterance] = Field(default_factory=list)
    recent_context: list[Utterance] = Field(default_factory=list)
    summary: str = ""
    previous_interventions: list[PreviousIntervention] = Field(default_factory=list)
    insights: list[MeetingInsight] = Field(default_factory=list)
    conversation_state: "ConversationState" = Field(default_factory=lambda: ConversationState())


class ReplayRequest(BaseModel):
    delegate: DelegateProfile
    utterances: list[Utterance]
    meeting_id: str | None = None
    realtime_delay_ms: int = 0


class ManualUtteranceRequest(BaseModel):
    speaker: str = "UNKNOWN"
    text: str
    source: Literal[
        "MANUAL",
        "REMOTE_AUDIO",
        "LOCAL_MIC_AUDIO",
        "FILE_AUDIO",
        "FILE",
        "MIC",
        "TAB_AUDIO",
        "REMOTE",
        "UNKNOWN",
    ] = "MANUAL"


class MeetingQuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def non_empty_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


class MeetingQuestionResponse(BaseModel):
    meeting_id: str
    question: str
    answer: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_latency_ms: int | None = None


class CreateMeetingRequest(BaseModel):
    delegate: DelegateProfile
    meeting_id: str | None = None


class MeetingResponse(BaseModel):
    meeting_id: str
    delegate: DelegateProfile


class ModelInfo(BaseModel):
    id: str
    owned_by: str | None = None


class HealthResponse(BaseModel):
    backend: Literal["ok"] = "ok"
    lm_studio: Literal["ok", "error"]
    model: str | None = None
    asr: Literal["ok", "unavailable", "error"]
    error: str | None = None


class ConversationState(BaseModel):
    current_topics: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    id: str = Field(default_factory=lambda: f"seg_{uuid4().hex}")
    source: str = "UNKNOWN"
    start: float
    end: float
    text: str
    created_at: datetime = Field(default_factory=utc_now)
    asr_latency_ms: float | None = None
    language: str | None = None
    confidence: float | None = None


class SemanticUtterance(BaseModel):
    id: str = Field(default_factory=lambda: f"utt_{uuid4().hex}")
    source: str
    text: str
    start: float
    end: float
    segment_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    language: str | None = None
    asr_latency_ms: float | None = None
    audio_finalize_latency_ms: float | None = None
    assembly_reason: str | None = None
    assembly_latency_ms: float | None = None
    segment_count: int = 0


class ASRTranscriptionResponse(BaseModel):
    language: str | None = None
    duration: float | None = None
    audio_duration_seconds: float | None = None
    processing_time_seconds: float
    processing_time_ms: int
    real_time_factor: float | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    provider: str = "faster-whisper"
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None


class ASRStatusResponse(BaseModel):
    status: Literal["idle", "loading", "ready", "error", "unavailable"]
    provider: str = "faster-whisper"
    model: str
    device: str | None = None
    compute_type: str | None = None
    language: str | None = None
    detail: str | None = None


class MeetingEvent(BaseModel):
    type: str
    meeting_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class BenchmarkExpected(BaseModel):
    category: InterventionCategory
    key_points: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    case_id: str
    delegate: DelegateProfile
    utterances: list[Utterance]
    evaluation_at_utterance: int
    expected: BenchmarkExpected
