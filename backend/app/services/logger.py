from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.schemas import InterventionDecision, MeetingInsight, MeetingState, Utterance


class EventLogger:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def start_session(self, state: MeetingState) -> Path:
        session_dir = self.base_dir / state.meeting_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(session_dir / "meeting.json", state)
        metrics = {"utterance_count": 0, "decision_count": 0, "intervention_count": 0, "insight_count": 0}
        self._write_json(session_dir / "metrics.json", metrics)
        return session_dir

    def log_utterance(self, meeting_id: str, utterance: Utterance) -> None:
        session_dir = self.base_dir / meeting_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(session_dir / "utterances.jsonl", utterance)

    def log_decision(self, meeting_id: str, decision: InterventionDecision) -> None:
        session_dir = self.base_dir / meeting_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(session_dir / "decisions.jsonl", decision)
        self._update_metrics(session_dir, decision)

    def log_insight(self, meeting_id: str, insight: MeetingInsight) -> None:
        session_dir = self.base_dir / meeting_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._append_jsonl(session_dir / "insights.jsonl", insight)
        path = session_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["insight_count"] = int(metrics.get("insight_count", 0)) + 1
        self._write_json(path, metrics)

    def save_state(self, state: MeetingState) -> None:
        self._write_json(self.base_dir / state.meeting_id / "meeting.json", state)

    def _update_metrics(self, session_dir: Path, decision: InterventionDecision) -> None:
        path = session_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["decision_count"] = int(metrics.get("decision_count", 0)) + 1
        metrics["utterance_count"] = max(int(metrics.get("utterance_count", 0)), decision.utterance_id)
        if decision.displayed:
            metrics["intervention_count"] = int(metrics.get("intervention_count", 0)) + 1
        latencies = metrics.setdefault("llm_latencies_ms", [])
        if decision.llm_latency_ms is not None:
            latencies.append(decision.llm_latency_ms)
        self._write_json(path, metrics)

    @staticmethod
    def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, BaseModel):
            data = value.model_dump(mode="json", by_alias=True)
        else:
            data = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, value: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(value.model_dump(mode="json"), ensure_ascii=False) + "\n")
