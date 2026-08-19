from __future__ import annotations

import httpx

from app.main import app
from app.schemas import ModelInfo


class FakeLLM:
    async def list_models(self):
        return [ModelInfo(id="local-test-model")]


class FakeASR:
    def status(self) -> str:
        return "ok"


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

