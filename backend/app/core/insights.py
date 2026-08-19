from __future__ import annotations

import re

from app.schemas import MeetingInsight, MeetingInsightType, Utterance


QUESTION_PATTERNS = [
    r"\?",
    r"\b(quem|como|quando|onde|por que|porque|qual|quais|algu[eé]m)\b",
    r"\b(who|what|when|where|why|how|can|could|should|anyone)\b",
]

ACTION_PATTERNS = [
    r"\b(vou|vamos|preciso|precisamos|precisa|ficou de|pr[oó]ximo passo|a[cç][aã]o)\b",
    r"\b(i will|we will|todo|to do|action item|next step|follow up|can you|could you|please)\b",
]

DECISION_PATTERNS = [
    r"\b(decidimos|decidido|fechado|aprovado|combinado|vamos seguir|fica definido)\b",
    r"\b(we decided|decided|decision|approved|agreed|settled|we will go with)\b",
]


def detect_meeting_insights(utterance: Utterance) -> list[MeetingInsight]:
    text = utterance.text.strip()
    normalized = text.lower()
    insights: list[MeetingInsight] = []

    if _matches(normalized, DECISION_PATTERNS):
        insights.append(
            MeetingInsight(
                type=MeetingInsightType.DECISION,
                utterance_id=utterance.id,
                speaker=utterance.speaker,
                text=text,
                reason="Detected wording that indicates a decision or agreement.",
                confidence=0.76,
            )
        )

    if _matches(normalized, ACTION_PATTERNS):
        insights.append(
            MeetingInsight(
                type=MeetingInsightType.ACTION_ITEM,
                utterance_id=utterance.id,
                speaker=utterance.speaker,
                text=text,
                reason="Detected wording that indicates a follow-up or owner action.",
                confidence=0.7,
            )
        )

    if _matches(normalized, QUESTION_PATTERNS):
        insights.append(
            MeetingInsight(
                type=MeetingInsightType.OPEN_QUESTION,
                utterance_id=utterance.id,
                speaker=utterance.speaker,
                text=text,
                reason="Detected wording that indicates an open question.",
                confidence=0.72,
            )
        )

    return insights


def _matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
