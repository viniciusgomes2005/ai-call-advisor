from __future__ import annotations

from pathlib import Path

from app.schemas import MeetingState, Utterance


PROMPT_VERSION = "intervention_v1"


class ContextManager:
    def __init__(
        self,
        prompt_path: Path,
        max_utterances: int = 40,
        max_chars: int = 16000,
    ):
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.max_utterances = max_utterances
        self.max_chars = max_chars

    def update_recent_context(self, state: MeetingState) -> None:
        if len(state.utterances) <= self.max_utterances:
            state.recent_context = list(state.utterances)
            return
        older = state.utterances[: -self.max_utterances]
        if not state.summary:
            state.summary = self._summarize_old_context(older)
        state.recent_context = state.utterances[-self.max_utterances :]

    def _summarize_old_context(self, utterances: list[Utterance]) -> str:
        speakers = sorted({u.speaker for u in utterances})
        first_id = utterances[0].id if utterances else None
        last_id = utterances[-1].id if utterances else None
        return (
            f"Earlier transcript summary: utterances {first_id}-{last_id}; "
            f"speakers involved: {', '.join(speakers)}. "
            "Full historical details were truncated for context limits."
        )

    def build_prompt(self, state: MeetingState, latest_utterance: Utterance) -> str:
        self.update_recent_context(state)
        transcript = self._format_transcript(state.recent_context)
        if len(transcript) > self.max_chars:
            transcript = transcript[-self.max_chars :]
        previous = "\n".join(
            f"- after utterance {item.utterance_id}: {item.category} - {item.response}"
            for item in state.previous_interventions[-10:]
        ) or "None"
        shareable = "\n".join(
            f"- Context: {item.context}\n  Information: {item.information}"
            for item in state.delegate.shareable_information
        ) or "None"
        intents = "\n".join(f"- {intent}" for intent in state.delegate.meeting_intents) or "None"
        return self.prompt_template.format(
            delegate_name=state.delegate.name,
            delegate_role=state.delegate.role,
            meeting_intents=intents,
            shareable_information=shareable,
            previous_interventions=previous,
            summary=state.summary or "None",
            transcript=transcript,
            latest_utterance_id=latest_utterance.id,
            latest_speaker=latest_utterance.speaker,
            latest_text=latest_utterance.text,
        )

    def build_question_prompt(self, state: MeetingState, question: str) -> str:
        self.update_recent_context(state)
        transcript = self._format_transcript(state.recent_context)
        if len(transcript) > self.max_chars:
            transcript = transcript[-self.max_chars :]
        shareable = "\n".join(
            f"- Context: {item.context}\n  Information: {item.information}"
            for item in state.delegate.shareable_information
        ) or "None"
        intents = "\n".join(f"- {intent}" for intent in state.delegate.meeting_intents) or "None"
        previous = "\n".join(
            f"- after utterance {item.utterance_id}: {item.category} - {item.response}"
            for item in state.previous_interventions[-10:]
        ) or "None"
        return (
            f"Delegate: {state.delegate.name}\n"
            f"Role: {state.delegate.role}\n\n"
            f"Meeting intents:\n{intents}\n\n"
            f"Shareable information:\n{shareable}\n\n"
            f"Previous assistant interventions:\n{previous}\n\n"
            f"Earlier summary:\n{state.summary or 'None'}\n\n"
            f"Recent transcript:\n{transcript or 'None'}\n\n"
            f"Question: {question}\n\n"
            "Answer in the same language as the question. Be brief and practical. "
            "Do not invent facts outside the transcript or shareable information."
        )

    @staticmethod
    def _format_transcript(utterances: list[Utterance]) -> str:
        return "\n".join(f"[{u.id}] {u.speaker}: {u.text}" for u in utterances)
