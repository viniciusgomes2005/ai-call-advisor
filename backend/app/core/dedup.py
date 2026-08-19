from __future__ import annotations

import re
from datetime import datetime, timezone

from app.schemas import InterventionCategory, InterventionDecision, MeetingState


def normalize_text(text: str) -> set[str]:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())
    stop = {"the", "a", "an", "and", "or", "to", "of", "for", "de", "da", "do", "e", "o", "a"}
    return {word for word in words if len(word) > 2 and word not in stop}


class Deduplicator:
    def __init__(self, cooldown_seconds: int = 10, enabled: bool = True):
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled

    def apply(self, state: MeetingState, decision: InterventionDecision) -> InterventionDecision:
        if not self.enabled or not decision.should_intervene:
            decision.displayed = decision.should_intervene
            return decision
        now = decision.timestamp or datetime.now(timezone.utc)
        if state.previous_interventions:
            latest = state.previous_interventions[-1]
            elapsed = (now - latest.timestamp).total_seconds()
            if elapsed < self.cooldown_seconds:
                return self._filter(decision, f"cooldown active ({elapsed:.1f}s)")
            if decision.response and self._similar(decision.response, latest.response):
                return self._filter(decision, "similar to recent intervention")
        decision.displayed = decision.category != InterventionCategory.KEEP_SILENCE
        return decision

    @staticmethod
    def _filter(decision: InterventionDecision, reason: str) -> InterventionDecision:
        decision.filtered = True
        decision.filter_reason = reason
        decision.displayed = False
        decision.category = InterventionCategory.KEEP_SILENCE
        decision.should_intervene = False
        decision.response = None
        decision.reason = f"Filtered: {reason}"
        return decision

    @staticmethod
    def _similar(left: str, right: str) -> bool:
        a = normalize_text(left)
        b = normalize_text(right)
        if not a or not b:
            return False
        return len(a & b) / len(a | b) >= 0.65

