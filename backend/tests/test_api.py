from __future__ import annotations

import httpx

from app.main import app
from app.schemas import ASRStatusResponse, ASRTranscriptionResponse, ModelInfo, TranscriptSegment


class FakeLLM:
    async def list_models(self):
        return [ModelInfo(id="local-test-model")]


class FakeASR:
    def status(self) -> str:
        return "ok"

    def status_payload(self):
        return ASRStatusResponse(status="ready", model="large-v3-turbo", device="cpu", compute_type="int8")

    async def transcribe_file(self, path, language=None):
        return ASRTranscriptionResponse(
            language="pt",
            duration=1.0,
            audio_duration_seconds=1.0,
            processing_time_seconds=0.1,
            processing_time_ms=100,
            real_time_factor=0.1,
            segments=[TranscriptSegment(start=0.0, end=1.0, text="Teste SAP ECC.", language="pt")],
            model="large-v3-turbo",
            device="cpu",
            compute_type="int8",
        )


async def test_health_endpoint_asgi(monkeypatch):
    import app.api.routes as routes

    monkeypatch.setattr(routes, "make_llm_provider", lambda: FakeLLM())
    monkeypatch.setattr(routes, "make_asr_provider", lambda: FakeASR())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["lm_studio"] == "ok"
    assert response.json()["model"] == "local-test-model"


async def test_asr_status_endpoint(monkeypatch):
    import app.api.routes as routes

    monkeypatch.setattr(routes, "make_asr_provider", lambda: FakeASR())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/asr/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model"] == "large-v3-turbo"


async def test_asr_transcribe_endpoint(monkeypatch):
    import app.api.routes as routes

    monkeypatch.setattr(routes, "make_asr_provider", lambda: FakeASR())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/asr/transcribe",
            files={"file": ("sample.wav", b"fake wav bytes", "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "pt"
    assert payload["processing_time_ms"] == 100
    assert payload["segments"][0]["text"] == "Teste SAP ECC."
