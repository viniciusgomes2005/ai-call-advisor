from __future__ import annotations

import abc
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import LLMDecision, ModelInfo, InterventionCategory


@dataclass(slots=True)
class LLMResult:
    decision: LLMDecision
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    raw_response: str


@dataclass(slots=True)
class LLMTextResult:
    text: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    @abc.abstractmethod
    async def decide_intervention(self, prompt: str, model: str | None = None) -> LLMResult:
        raise NotImplementedError

    @abc.abstractmethod
    async def answer_question(self, prompt: str, model: str | None = None) -> LLMTextResult:
        raise NotImplementedError


def fallback_silence(reason: str = "The model response could not be parsed.") -> LLMDecision:
    return LLMDecision(
        category=InterventionCategory.KEEP_SILENCE,
        should_intervene=False,
        confidence=0.0,
        response=None,
        reason=reason,
        trigger_utterance_ids=[],
    )


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def parse_llm_decision(text: str) -> LLMDecision:
    data = extract_json_object(text)
    if data.get("category") == "SILENCE":
        data["category"] = "KEEP_SILENCE"
    if data.get("category") == "CHIME IN":
        data["category"] = "CHIME_IN"
    return LLMDecision.model_validate(data)


class LMStudioProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, default_model: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout

    async def list_models(self) -> list[ModelInfo]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        return [ModelInfo(id=item["id"], owned_by=item.get("owned_by")) for item in payload.get("data", [])]

    async def decide_intervention(self, prompt: str, model: str | None = None) -> LLMResult:
        selected_model = model or self.default_model
        if not selected_model:
            models = await self.list_models()
            selected_model = models[0].id if models else ""
        if not selected_model:
            return LLMResult(fallback_silence("No LM Studio model is loaded."), None, None, None, 0, "")

        start = time.perf_counter()
        raw_response = ""
        usage: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": selected_model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON. Do not include markdown."},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            raw_response = payload["choices"][0]["message"]["content"]

        try:
            decision = parse_llm_decision(raw_response)
        except (json.JSONDecodeError, ValidationError, ValueError):
            repair_prompt = (
                "Repair the following content into valid JSON matching exactly this schema: "
                '{"category":"KEEP_SILENCE|EXPLICIT_CUE|IMPLICIT_CUE|CHIME_IN",'
                '"should_intervene":boolean,"confidence":number,"response":string|null,'
                '"reason":string,"trigger_utterance_ids":[integer]}. '
                "Return only JSON.\n\n"
                f"{raw_response}"
            )
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    repair = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": selected_model,
                            "temperature": 0,
                            "messages": [{"role": "user", "content": repair_prompt}],
                            "response_format": {"type": "json_object"},
                        },
                    )
                    repair.raise_for_status()
                    repair_payload = repair.json()
                    raw_response = repair_payload["choices"][0]["message"]["content"]
                    decision = parse_llm_decision(raw_response)
            except Exception:
                decision = fallback_silence()

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResult(
            decision=decision,
            model=selected_model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            raw_response=raw_response,
        )

    async def answer_question(self, prompt: str, model: str | None = None) -> LLMTextResult:
        selected_model = model or self.default_model
        if not selected_model:
            models = await self.list_models()
            selected_model = models[0].id if models else ""
        if not selected_model:
            return LLMTextResult("No LM Studio model is loaded.", None, None, None, 0)

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": selected_model,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer concisely using only the meeting context provided. "
                                "If the answer is not in the context, say that it is not available yet."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()

        usage = payload.get("usage") or {}
        text = payload["choices"][0]["message"]["content"].strip()
        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMTextResult(
            text=text,
            model=selected_model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )
