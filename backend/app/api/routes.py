from __future__ import annotations

import base64
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.api.dependencies import make_asr_provider, make_engine, make_llm_provider
from app.schemas import (
    CreateMeetingRequest,
    DelegateProfile,
    HealthResponse,
    ManualUtteranceRequest,
    MeetingEvent,
    MeetingResponse,
    ReplayRequest,
    Utterance,
)
from app.services.live import LiveMeetingSession
from app.services.replay import ReplayService
from app.settings import get_settings

router = APIRouter()
ENGINES = {}


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
    return {"utterance": utterance.model_dump(mode="json"), "decision": decision.model_dump(mode="json")}


@router.post("/replay")
async def replay(request: ReplayRequest, model: str | None = None):
    engine = make_engine(request.delegate, request.meeting_id, model)
    ENGINES[engine.state.meeting_id] = engine
    decisions = await ReplayService(engine).run(request)
    return {
        "meeting_id": engine.state.meeting_id,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
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
        segments = await asr.process_audio_file(path, language=get_settings().language, speaker=speaker)
    finally:
        path.unlink(missing_ok=True)
    outputs = []
    for segment in segments:
        utterance = Utterance(
            id=max((u.id for u in engine.state.utterances), default=0) + 1,
            speaker=segment.speaker,
            text=segment.text,
            source="FILE_AUDIO",
        )
        decision = await engine.ingest_utterance(utterance)
        outputs.append({"utterance": utterance.model_dump(mode="json"), "decision": decision.model_dump(mode="json")})
    return {"segments": outputs}


@router.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    session: LiveMeetingSession | None = None
    audio_buffers: dict[str, bytearray] = {"REMOTE_AUDIO": bytearray(), "LOCAL_MIC_AUDIO": bytearray()}
    transcription_tasks: set[asyncio.Task] = set()
    send_lock = asyncio.Lock()

    async def send_event(event: MeetingEvent) -> None:
        async with send_lock:
            await websocket.send_json(event.model_dump(mode="json"))

    async def transcribe_buffer(source: str, data: bytes) -> None:
        if not session or not data:
            return
        speaker = "ME" if source == "LOCAL_MIC_AUDIO" else "REMOTE"
        try:
            segments = await make_asr_provider().process_audio(
                data,
                suffix=".webm",
                language=get_settings().language,
                speaker=speaker,
            )
            for segment in segments:
                await send_event(
                    MeetingEvent(
                        type="transcript.partial",
                        meeting_id=session.engine.state.meeting_id,
                        payload={"speaker": speaker, "text": segment.text, "source": source},
                    )
                )
                await session.ingest_final_utterance(speaker=speaker, text=segment.text, source=source)
        except Exception as exc:
            await send_event(
                MeetingEvent(
                    type="transcript.error",
                    meeting_id=session.engine.state.meeting_id,
                    payload={"source": source, "error": str(exc)},
                )
            )

    def schedule_transcription(source: str) -> None:
        data = bytes(audio_buffers.get(source, b""))
        audio_buffers[source] = bytearray()
        if not data:
            return
        task = asyncio.create_task(transcribe_buffer(source, data))
        transcription_tasks.add(task)
        task.add_done_callback(transcription_tasks.discard)

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
                    on_event=send_event,
                )
                await send_event(
                    MeetingEvent(
                        type="meeting.state.updated",
                        meeting_id=engine.state.meeting_id,
                        payload={"meeting_id": engine.state.meeting_id},
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
                source = message.get("source", "REMOTE_AUDIO")
                data = base64.b64decode(message.get("data", ""))
                audio_buffers.setdefault(source, bytearray()).extend(data)
                await send_event(
                    MeetingEvent(
                        type="audio.chunk",
                        meeting_id=session.engine.state.meeting_id if session else None,
                        payload={"source": source, "bytes": len(data)},
                    )
                )
                if len(audio_buffers[source]) >= 240_000:
                    schedule_transcription(source)
            elif event_type == "audio.flush":
                source = message.get("source", "REMOTE_AUDIO")
                schedule_transcription(source)
            elif event_type == "meeting.stop":
                for source in list(audio_buffers):
                    schedule_transcription(source)
                if transcription_tasks:
                    await asyncio.gather(*transcription_tasks)
                if session:
                    await session.drain()
                    await send_event(MeetingEvent(type="meeting.ended", meeting_id=session.engine.state.meeting_id))
                break
    except WebSocketDisconnect:
        if transcription_tasks:
            await asyncio.gather(*transcription_tasks)
        if session:
            await session.drain()
