from __future__ import annotations

import base64
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.dependencies import make_asr_provider, make_engine, make_llm_provider
from app.asr.streaming import AudioQueueItem, FinalizedAudio, SpeechSegmenter, SpeechStarted
from app.meeting.utterance_assembler import AssemblyResult, TranscriptOrderingBuffer, UtteranceAssembler, UtteranceAssemblerConfig
from app.schemas import (
    ASRStatusResponse,
    ASRTranscriptionResponse,
    AudioFinalizationReason,
    CreateMeetingRequest,
    DelegateProfile,
    HealthResponse,
    ManualUtteranceRequest,
    MeetingQuestionRequest,
    MeetingEvent,
    MeetingResponse,
    ReplayRequest,
    TranscriptSegment,
    Utterance,
)
from app.services.logger import EventLogger
from app.services.live import LiveMeetingSession
from app.services.replay import ReplayService
from app.settings import get_settings

router = APIRouter()
ENGINES = {}
logger = logging.getLogger(__name__)


def speaker_for_source(source: str) -> str:
    return "ME" if source in {"MIC", "LOCAL_MIC", "LOCAL_MIC_AUDIO"} else "REMOTE"


def normalize_audio_source(source: str) -> str:
    if source in {"LOCAL_MIC", "LOCAL_MIC_AUDIO"}:
        return "MIC"
    if source in {"REMOTE_AUDIO", "REMOTE"}:
        return "TAB_AUDIO"
    if source in {"MIC", "TAB_AUDIO", "FILE"}:
        return source
    return "UNKNOWN"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    llm = make_llm_provider()
    asr = make_asr_provider()
    try:
        models = await llm.list_models()
        selected = settings.llm_model or (models[0].id if models else None)
        return HealthResponse(lm_studio="ok", model=selected, asr=asr.status())
    except (httpx.HTTPError, OSError) as exc:
        return HealthResponse(lm_studio="error", model=settings.llm_model or None, asr=asr.status(), error=str(exc))


@router.get("/api/asr/status", response_model=ASRStatusResponse)
async def asr_status() -> ASRStatusResponse:
    return make_asr_provider().status_payload()


@router.post("/api/asr/transcribe", response_model=ASRTranscriptionResponse)
async def transcribe_audio_file(file: Annotated[UploadFile, File()], language: str | None = None):
    suffix = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(await file.read())
        path = Path(temp.name)
    try:
        settings = get_settings()
        result = await make_asr_provider().transcribe_file(path, language=language or settings.whisper_language)
        session_id = str(uuid4())
        logger = EventLogger(settings.data_dir)
        logger.save_audio_metadata(
            session_id,
            {
                "filename": file.filename,
                "source": "FILE",
                "content_type": file.content_type,
                "audio_duration_seconds": result.audio_duration_seconds,
                "save_raw_audio": settings.save_raw_audio,
            },
        )
        for index, segment in enumerate(result.segments, start=1):
            logger.log_transcript_segment(
                session_id,
                segment,
                {
                    "id": f"seg_{index}",
                    "speaker": None,
                    "source": "FILE",
                    "asr_latency_ms": result.processing_time_ms,
                    "model": result.model,
                },
            )
        logger.log_asr_metrics(
            session_id,
            {
                "source": "FILE",
                "audio_duration_ms": int((result.audio_duration_seconds or 0) * 1000),
                "asr_latency_ms": result.processing_time_ms,
                "real_time_factor": result.real_time_factor,
                "model": result.model,
                "device": result.device,
                "compute_type": result.compute_type,
            },
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Whisper transcription failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


@router.get("/models")
async def models():
    try:
        return {"models": [item.model_dump() for item in await make_llm_provider().list_models()]}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"LM Studio unavailable: {exc}") from exc


@router.post("/meetings", response_model=MeetingResponse)
async def create_meeting(request: CreateMeetingRequest, model: str | None = None) -> MeetingResponse:
    engine = make_engine(request.delegate, request.meeting_id, model)
    ENGINES[engine.state.meeting_id] = engine
    return MeetingResponse(meeting_id=engine.state.meeting_id, delegate=engine.state.delegate)


@router.post("/meetings/{meeting_id}/utterances")
async def manual_utterance(meeting_id: str, request: ManualUtteranceRequest):
    engine = ENGINES.get(meeting_id)
    if not engine:
        raise HTTPException(status_code=404, detail="meeting not found")
    utterance = Utterance(
        id=max((u.id for u in engine.state.utterances), default=0) + 1,
        speaker=request.speaker,
        text=request.text,
        source=request.source,
    )
    decision = await engine.ingest_utterance(utterance)
    return {
        "utterance": utterance.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "insights": [insight.model_dump(mode="json") for insight in engine.insights_for_utterance(utterance.id)],
    }


@router.post("/meetings/{meeting_id}/questions")
async def ask_question(meeting_id: str, request: MeetingQuestionRequest):
    engine = ENGINES.get(meeting_id)
    if not engine:
        raise HTTPException(status_code=404, detail="meeting not found")
    answer = await engine.answer_question(request.question)
    return answer.model_dump(mode="json")


@router.post("/replay")
async def replay(request: ReplayRequest, model: str | None = None):
    engine = make_engine(request.delegate, request.meeting_id, model)
    ENGINES[engine.state.meeting_id] = engine
    decisions = await ReplayService(engine).run(request)
    return {
        "meeting_id": engine.state.meeting_id,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
        "insights": [insight.model_dump(mode="json") for insight in engine.state.insights],
    }


@router.post("/meetings/{meeting_id}/audio")
async def upload_audio(meeting_id: str, file: Annotated[UploadFile, File()], speaker: str = "UNKNOWN"):
    engine = ENGINES.get(meeting_id)
    if not engine:
        raise HTTPException(status_code=404, detail="meeting not found")
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(await file.read())
        path = Path(temp.name)
    try:
        asr = make_asr_provider()
        settings = get_settings()
        result = await asr.transcribe_file(path, language=settings.whisper_language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Whisper transcription failed: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)
    outputs = []
    engine.event_logger.save_audio_metadata(
        engine.state.meeting_id,
        {
            "filename": file.filename,
            "source": "FILE",
            "content_type": file.content_type,
            "save_raw_audio": get_settings().save_raw_audio,
        },
    )
    engine.event_logger.log_asr_metrics(
        engine.state.meeting_id,
        {
            "source": "FILE",
            "audio_duration_ms": int((result.audio_duration_seconds or 0) * 1000),
            "asr_latency_ms": result.processing_time_ms,
            "real_time_factor": result.real_time_factor,
            "model": result.model,
            "device": result.device,
            "compute_type": result.compute_type,
        },
    )
    for segment in result.segments:
        engine.event_logger.log_transcript_segment(
            engine.state.meeting_id,
            segment,
            {"speaker": speaker, "source": "FILE", "asr_latency_ms": result.processing_time_ms, "model": result.model},
        )
        utterance = Utterance(
            id=max((u.id for u in engine.state.utterances), default=0) + 1,
            speaker=speaker,
            text=segment.text,
            source="FILE",
            start=segment.start,
            end=segment.end,
            language=segment.language,
            asr_latency_ms=result.processing_time_ms,
        )
        decision = await engine.ingest_utterance(utterance)
        outputs.append(
            {
                "utterance": utterance.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
                "insights": [
                    insight.model_dump(mode="json") for insight in engine.insights_for_utterance(utterance.id)
                ],
            }
        )
    return {"segments": outputs}


@router.websocket("/ws/audio")
@router.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    session: LiveMeetingSession | None = None
    assembler: UtteranceAssembler | None = None
    settings = get_settings()
    audio_queue: asyncio.Queue[AudioQueueItem | None] = asyncio.Queue()
    segmenters: dict[str, SpeechSegmenter] = {}
    asr_tasks: set[asyncio.Task[None]] = set()
    asr_sequence: dict[str, int] = {}
    ordering_buffer = TranscriptOrderingBuffer()
    assembly_flush_task: asyncio.Task[None] | None = None
    send_lock = asyncio.Lock()
    disconnected = False
    chunk_stats: dict[str, dict[str, int | float]] = {}

    def current_meeting_id() -> str | None:
        return session.engine.state.meeting_id if session else None

    def log_audio_debug(kind: str, payload: dict) -> None:
        meeting_id = current_meeting_id()
        event = {"event": kind, "timestamp": time.time(), **payload}
        logger.info("live audio debug: meeting=%s event=%s payload=%s", meeting_id or "-", kind, payload)
        if session and meeting_id:
            session.engine.event_logger.log_audio_debug(meeting_id, event)

    async def send_live_state(payload: dict) -> None:
        if not session:
            return
        await send_event(
            MeetingEvent(
                type="meeting.state.updated",
                meeting_id=session.engine.state.meeting_id,
                payload=payload,
            )
        )

    async def send_event(event: MeetingEvent) -> None:
        nonlocal disconnected
        if disconnected:
            return
        async with send_lock:
            try:
                await websocket.send_json(event.model_dump(mode="json"))
            except (RuntimeError, WebSocketDisconnect):
                disconnected = True

    def segmenter_for(source: str) -> SpeechSegmenter:
        if source not in segmenters:
            segmenters[source] = SpeechSegmenter(
                min_speech_ms=settings.asr_min_speech_ms,
                silence_end_ms=settings.asr_silence_end_ms,
                max_utterance_ms=settings.asr_max_utterance_ms,
                rms_threshold=settings.asr_vad_rms_threshold,
            )
        return segmenters[source]

    def check_backlog() -> None:
        if not session or assembler is None:
            return
        pending_transcript_segments = ordering_buffer.pending_count() + assembler.pending_transcript_segments()
        oldest_pending_segment_age_ms = max(
            ordering_buffer.oldest_pending_age_ms(), assembler.oldest_pending_segment_age_ms()
        )
        if pending_transcript_segments <= settings.utterance_backlog_warning_threshold:
            return
        payload = {
            "pending_audio_segments": audio_queue.qsize() + len(asr_tasks),
            "pending_transcript_segments": pending_transcript_segments,
            "oldest_pending_segment_age_ms": oldest_pending_segment_age_ms,
            "threshold": settings.utterance_backlog_warning_threshold,
        }
        logger.warning("Utterance backlog above threshold: %s", payload)
        log_audio_debug("backlog.warning", payload)

    async def handle_assembly_result(result: AssemblyResult) -> None:
        if not session:
            return
        meeting_id = session.engine.state.meeting_id
        for entry in result.logs:
            session.engine.event_logger.log_utterance_assembly_event(meeting_id, entry.event, entry.payload)
        if result.updated:
            await send_event(
                MeetingEvent(
                    type="semantic_utterance.updated",
                    meeting_id=meeting_id,
                    payload=result.updated.model_dump(mode="json"),
                )
            )
        # Only a finalized SemanticUtterance is allowed to reach MeetingEngine/LLM -
        # never an in-progress `updated` utterance or a raw transcript.segment.
        for semantic in result.finalized:
            session.engine.event_logger.log_semantic_utterance(meeting_id, semantic)
            session.engine.event_logger.log_utterance_assembly_event(
                meeting_id, "utterance.finalized", semantic.model_dump(mode="json")
            )
            await send_event(
                MeetingEvent(type="semantic_utterance.final", meeting_id=meeting_id, payload=semantic.model_dump(mode="json"))
            )
            speaker = speaker_for_source(semantic.source)
            utterance = await session.ingest_semantic_utterance(semantic, speaker)
            await send_live_state(
                {
                    "utterance_count": utterance.id,
                    "last_assembly_reason": semantic.assembly_reason,
                    "last_assembly_latency_ms": semantic.assembly_latency_ms,
                    "segments_per_utterance": semantic.segment_count,
                    "last_transcript_empty": False,
                }
            )

    async def assembly_flush_worker() -> None:
        interval_seconds = max(0.05, settings.utterance_finalization_delay_ms / 1000 / 2)
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                if not session or assembler is None:
                    continue
                result = assembler.flush_expired()
                if result.finalized:
                    await handle_assembly_result(result)
                check_backlog()
            except asyncio.CancelledError:
                return
            except Exception:
                # This background task must keep running for the whole meeting - one bad
                # tick (e.g. a transient logging/event error) must not silently kill the
                # only thing that finalizes utterances after a real silence.
                logger.exception("assembly_flush_worker tick failed")

    async def transcribe_final_audio(final_audio: FinalizedAudio, sequence: int) -> None:
        if not session or assembler is None:
            return
        source = normalize_audio_source(final_audio.source)
        finalization_reason = str(final_audio.finalization_reason)
        asr_started = time.perf_counter()
        log_audio_debug(
            "asr.started",
            {
                "source": source,
                "sequence": sequence,
                "duration_ms": int(final_audio.duration_seconds * 1000),
                "bytes": len(final_audio.data),
                "queue_size": audio_queue.qsize(),
                "inflight_asr": len(asr_tasks),
                "finalization_reason": finalization_reason,
            },
        )
        await send_event(
            MeetingEvent(
                type="asr.started",
                meeting_id=session.engine.state.meeting_id,
                payload={
                    "source": source,
                    "sequence": sequence,
                    "duration_ms": int(final_audio.duration_seconds * 1000),
                    "queue_size": audio_queue.qsize(),
                    "inflight_asr": len(asr_tasks),
                    "finalization_reason": finalization_reason,
                },
            )
        )
        transcript_segments: list[TranscriptSegment] = []
        try:
            result = await make_asr_provider().transcribe_audio_chunk(
                final_audio.data,
                sample_rate=final_audio.sample_rate,
                language=settings.whisper_language,
                audio_format="pcm_s16le",
            )
            asr_wall_latency_ms = int((time.perf_counter() - asr_started) * 1000)
            asr_latency_ms = result.processing_time_ms
            asr_queue_latency_ms = max(0, asr_wall_latency_ms - asr_latency_ms)
            audio_finalize_latency_ms = int((asr_started - final_audio.finalized_at) * 1000)
            text = " ".join(segment.text for segment in result.segments).strip()
            metrics = {
                "source": source,
                "sequence": sequence,
                "audio_duration_ms": int(final_audio.duration_seconds * 1000),
                "audio_finalize_latency_ms": audio_finalize_latency_ms,
                "asr_latency_ms": asr_latency_ms,
                "asr_wall_latency_ms": asr_wall_latency_ms,
                "asr_queue_latency_ms": asr_queue_latency_ms,
                "real_time_factor": result.real_time_factor,
                "segment_count": len(result.segments),
                "text_chars": len(text),
                "model": result.model,
                "device": result.device,
                "compute_type": result.compute_type,
                "finalization_reason": finalization_reason,
            }
            session.engine.event_logger.log_asr_metrics(session.engine.state.meeting_id, metrics)
            log_audio_debug("asr.completed", metrics)
            await send_event(
                MeetingEvent(
                    type="asr.completed",
                    meeting_id=session.engine.state.meeting_id,
                    payload=metrics,
                )
            )
            if not result.segments:
                await send_event(
                    MeetingEvent(
                        type="transcript.empty",
                        meeting_id=session.engine.state.meeting_id,
                        payload=metrics,
                    )
                )
                await send_live_state(
                    {
                        "last_asr_latency_ms": asr_latency_ms,
                        "last_asr_queue_latency_ms": asr_queue_latency_ms,
                        "last_transcript_empty": True,
                    }
                )
                return
            # Absolute meeting-timeline timestamps: this chunk started at
            # final_audio.start_seconds, whisper's segment.start/end are relative to it.
            for index, segment in enumerate(result.segments):
                transcript_segments.append(
                    TranscriptSegment(
                        id=f"seg_{source.lower()}_{sequence}_{index}",
                        source=source,
                        start=final_audio.start_seconds + segment.start,
                        end=final_audio.start_seconds + segment.end,
                        text=segment.text,
                        asr_latency_ms=asr_latency_ms,
                        audio_finalize_latency_ms=audio_finalize_latency_ms,
                        finalization_reason=final_audio.finalization_reason,
                        language=segment.language or result.language,
                        confidence=segment.confidence,
                    )
                )
        except Exception as exc:
            log_audio_debug(
                "asr.error",
                {
                    "source": source,
                    "sequence": sequence,
                    "duration_ms": int(final_audio.duration_seconds * 1000),
                    "bytes": len(final_audio.data),
                    "error": str(exc),
                },
            )
            logger.exception("Live ASR failed for source=%s sequence=%s", source, sequence)
            await send_event(
                MeetingEvent(
                    type="transcript.error",
                    meeting_id=session.engine.state.meeting_id,
                    payload={"source": source, "sequence": sequence, "error": str(exc)},
                )
            )
        finally:
            # Whether ASR succeeded, failed, or returned nothing, this sequence number
            # must be marked complete - otherwise a failed/empty chunk would block the
            # ordering buffer forever for every later sequence of the same source.
            ready_segments = ordering_buffer.push_batch(source, sequence, transcript_segments)
            for segment in ready_segments:
                session.engine.event_logger.log_transcript_segment(
                    session.engine.state.meeting_id,
                    segment,
                    {"speaker": speaker_for_source(segment.source), "sequence": sequence},
                )
                session.engine.event_logger.log_transcript_event(
                    session.engine.state.meeting_id, "transcript.segment.created", segment
                )
                await send_event(
                    MeetingEvent(
                        type="transcript.segment",
                        meeting_id=session.engine.state.meeting_id,
                        payload=segment.model_dump(mode="json"),
                    )
                )
                await handle_assembly_result(assembler.push(segment))
            check_backlog()

    def schedule_asr(final_audio: FinalizedAudio) -> None:
        source = normalize_audio_source(final_audio.source)
        sequence = asr_sequence.get(source, 0)
        asr_sequence[source] = sequence + 1
        log_audio_debug(
            "asr.scheduled",
            {
                "source": source,
                "sequence": sequence,
                "duration_ms": int(final_audio.duration_seconds * 1000),
                "bytes": len(final_audio.data),
                "queue_size": audio_queue.qsize(),
                "finalization_reason": str(final_audio.finalization_reason),
            },
        )
        task = asyncio.create_task(transcribe_final_audio(final_audio, sequence))
        asr_tasks.add(task)
        task.add_done_callback(asr_tasks.discard)

    async def audio_worker() -> None:
        while True:
            item = await audio_queue.get()
            try:
                if item is None:
                    for source, segmenter in list(segmenters.items()):
                        final_audio = segmenter.flush(source, reason=AudioFinalizationReason.MEETING_END)
                        if final_audio:
                            schedule_asr(final_audio)
                    return
                for event in segmenter_for(item.source).push(item):
                    if isinstance(event, SpeechStarted):
                        source = normalize_audio_source(event.source)
                        log_audio_debug("speech.started", {"source": source, "start": event.start_seconds})
                        await send_event(
                            MeetingEvent(
                                type="speech.started",
                                meeting_id=current_meeting_id(),
                                payload={"source": source, "start": event.start_seconds},
                            )
                        )
                    else:
                        source = normalize_audio_source(event.source)
                        log_audio_debug(
                            "speech.ended",
                            {
                                "source": source,
                                "start": event.start_seconds,
                                "end": event.end_seconds,
                                "duration_ms": int(event.duration_seconds * 1000),
                                "bytes": len(event.data),
                            },
                        )
                        await send_event(
                            MeetingEvent(
                                type="speech.ended",
                                meeting_id=current_meeting_id(),
                                payload={
                                    "source": source,
                                    "start": event.start_seconds,
                                    "end": event.end_seconds,
                                    "duration_ms": int(event.duration_seconds * 1000),
                                },
                            )
                        )
                        schedule_asr(event)
            finally:
                audio_queue.task_done()

    worker_task = asyncio.create_task(audio_worker())

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            event_type = message.get("type")
            if event_type == "meeting.start":
                delegate_payload = DelegateProfile.model_validate(message["delegate"])
                model = message.get("model")
                engine = make_engine(delegate_payload, message.get("meeting_id"), model)
                ENGINES[engine.state.meeting_id] = engine
                session = LiveMeetingSession(
                    engine,
                    max_llm_concurrency=get_settings().max_llm_concurrency,
                    llm_enabled=bool(message.get("llm_enabled", True)),
                    on_event=send_event,
                )
                assembler = UtteranceAssembler(
                    UtteranceAssemblerConfig(
                        enabled=settings.utterance_assembly_enabled,
                        merge_max_gap_ms=settings.utterance_merge_max_gap_ms,
                        hard_max_duration_ms=settings.utterance_hard_max_duration_ms,
                        hard_max_chars=settings.utterance_hard_max_chars,
                        finalization_delay_ms=settings.utterance_finalization_delay_ms,
                    )
                )
                assembly_flush_task = asyncio.create_task(assembly_flush_worker())
                logger.info(
                    "Utterance assembly enabled=%s merge_max_gap_ms=%s finalization_delay_ms=%s",
                    settings.utterance_assembly_enabled,
                    settings.utterance_merge_max_gap_ms,
                    settings.utterance_finalization_delay_ms,
                )
                logger.info(
                    "Live meeting started: meeting=%s model=%s llm_enabled=%s vad_threshold=%s silence_end_ms=%s",
                    engine.state.meeting_id,
                    model or settings.llm_model or "-",
                    session.llm_enabled,
                    settings.asr_vad_rms_threshold,
                    settings.asr_silence_end_ms,
                )
                engine.event_logger.save_audio_metadata(
                    engine.state.meeting_id,
                    {
                        "sample_rate": 16000,
                        "format": "pcm_s16le",
                        "sources": ["MIC", "TAB_AUDIO"],
                        "save_raw_audio": settings.save_raw_audio,
                    },
                )
                await send_event(
                    MeetingEvent(
                        type="meeting.state.updated",
                        meeting_id=engine.state.meeting_id,
                        payload={"meeting_id": engine.state.meeting_id, "llm_enabled": session.llm_enabled},
                    )
                )
            elif event_type == "llm.set_enabled":
                if not session:
                    raise RuntimeError("meeting.start must be sent first")
                session.llm_enabled = bool(message.get("enabled", True))
                log_audio_debug("llm.set_enabled", {"enabled": session.llm_enabled})
                await send_event(
                    MeetingEvent(
                        type="meeting.state.updated",
                        meeting_id=session.engine.state.meeting_id,
                        payload={"llm_enabled": session.llm_enabled},
                    )
                )
            elif event_type == "utterance.final":
                if not session:
                    raise RuntimeError("meeting.start must be sent first")
                payload = message["payload"]
                utterance = await session.ingest_final_utterance(
                    speaker=payload.get("speaker", "UNKNOWN"),
                    text=payload["text"],
                    source=payload.get("source", "UNKNOWN"),
                )
                await send_event(
                    MeetingEvent(
                        type="meeting.state.updated",
                        meeting_id=session.engine.state.meeting_id,
                        payload={"utterance_count": utterance.id},
                    )
                )
            elif event_type == "audio.chunk":
                source = normalize_audio_source(message.get("source", "TAB_AUDIO"))
                data = base64.b64decode(message.get("data", ""))
                sample_rate = int(message.get("sample_rate", 16000))
                await audio_queue.put(AudioQueueItem(source=source, data=data, sample_rate=sample_rate))
                stats = chunk_stats.setdefault(source, {"chunks": 0, "bytes": 0, "last_log_at": 0.0})
                stats["chunks"] = int(stats["chunks"]) + 1
                stats["bytes"] = int(stats["bytes"]) + len(data)
                now = time.perf_counter()
                if now - float(stats["last_log_at"]) >= 5:
                    payload = {
                        "source": source,
                        "chunks": stats["chunks"],
                        "bytes": stats["bytes"],
                        "queue_size": audio_queue.qsize(),
                        "sample_rate": sample_rate,
                    }
                    log_audio_debug("audio.chunks", payload)
                    stats["chunks"] = 0
                    stats["bytes"] = 0
                    stats["last_log_at"] = now
                await send_event(
                    MeetingEvent(
                        type="audio.chunk",
                        meeting_id=current_meeting_id(),
                        payload={"source": source, "bytes": len(data), "queue_size": audio_queue.qsize()},
                    )
                )
            elif event_type == "audio.flush":
                source = normalize_audio_source(message.get("source", "TAB_AUDIO"))
                final_audio = segmenter_for(source).flush(source, reason=AudioFinalizationReason.MANUAL_FLUSH)
                if final_audio:
                    schedule_asr(final_audio)
            elif event_type == "meeting.stop":
                await audio_queue.put(None)
                await worker_task
                if asr_tasks:
                    await asyncio.gather(*asr_tasks, return_exceptions=True)
                if assembly_flush_task:
                    assembly_flush_task.cancel()
                if assembler is not None:
                    # Any buffer still open (e.g. mid-sentence, or artificially cut by
                    # MAX_DURATION and never followed by a real silence) must not be lost.
                    await handle_assembly_result(assembler.flush_all(reason="meeting_end"))
                if session:
                    await session.drain()
                    await send_event(MeetingEvent(type="meeting.ended", meeting_id=session.engine.state.meeting_id))
                break
    except WebSocketDisconnect:
        disconnected = True
        worker_task.cancel()
        if asr_tasks:
            await asyncio.gather(*asr_tasks, return_exceptions=True)
        if assembly_flush_task:
            assembly_flush_task.cancel()
        if assembler is not None:
            await handle_assembly_result(assembler.flush_all(reason="meeting_end"))
        if session:
            await session.drain()
    finally:
        if not worker_task.done():
            worker_task.cancel()
        if assembly_flush_task and not assembly_flush_task.done():
            assembly_flush_task.cancel()
