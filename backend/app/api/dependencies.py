from __future__ import annotations

from pathlib import Path

from app.asr import FasterWhisperProvider
from app.core.dedup import Deduplicator
from app.core.context import ContextManager
from app.llm import LMStudioProvider
from app.meeting import MeetingEngine
from app.services.logger import EventLogger
from app.settings import get_settings
from app.schemas import DelegateProfile


def prompt_path() -> Path:
    return Path("prompts/intervention_v1.txt")


def make_llm_provider() -> LMStudioProvider:
    settings = get_settings()
    return LMStudioProvider(settings.llm_base_url, settings.llm_api_key, settings.llm_model)


def make_asr_provider() -> FasterWhisperProvider:
    settings = get_settings()
    return FasterWhisperProvider(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)


def make_engine(delegate: DelegateProfile, meeting_id: str | None = None, model: str | None = None) -> MeetingEngine:
    settings = get_settings()
    return MeetingEngine(
        delegate=delegate,
        llm_provider=make_llm_provider(),
        context_manager=ContextManager(prompt_path(), settings.context_max_utterances, settings.context_max_chars),
        event_logger=EventLogger(settings.data_dir),
        deduplicator=Deduplicator(settings.intervention_cooldown_seconds, settings.enable_intervention_filter),
        meeting_id=meeting_id,
        model=model or settings.llm_model or None,
        max_suggestion_age_seconds=settings.max_suggestion_age_seconds,
    )

