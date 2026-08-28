from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = ""

    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_language: str | None = None
    whisper_initial_prompt: str = ""
    whisper_hotwords: str = ""
    whisper_vad_min_silence_duration_ms: int = 500
    whisper_beam_size: int = 1
    whisper_condition_on_previous_text: bool = False
    asr_use_domain_prompt: bool = False
    asr_load_on_startup: bool = True
    asr_warmup: bool = False
    asr_parallel_workers: int = 2
    asr_min_speech_ms: int = 500
    asr_silence_end_ms: int = 450
    asr_max_utterance_ms: int = 6000
    asr_vad_rms_threshold: float = 0.012
    save_raw_audio: bool = False
    language: str = "pt"

    silence_end_ms: int = 700
    min_utterance_ms: int = 700
    max_utterance_ms: int = 15000

    intervention_cooldown_seconds: int = 10
    enable_intervention_filter: bool = True

    max_llm_concurrency: int = 1
    max_suggestion_age_seconds: int = 15

    context_max_utterances: int = 40
    context_max_chars: int = 16000

    log_level: str = "INFO"
    data_dir: Path = Field(default_factory=lambda: Path("data/sessions"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: str | None) -> str:
        if not value:
            return "pt"
        primary = str(value).split(":", maxsplit=1)[0]
        primary = primary.replace("_", "-").split("-", maxsplit=1)[0].strip().lower()
        return primary or "pt"

    @field_validator("whisper_language", mode="before")
    @classmethod
    def normalize_optional_language(cls, value: str | None) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        primary = str(value).split(":", maxsplit=1)[0]
        primary = primary.replace("_", "-").split("-", maxsplit=1)[0].strip().lower()
        return primary or None

    @field_validator("asr_parallel_workers", mode="before")
    @classmethod
    def normalize_parallel_workers(cls, value: int | str | None) -> int:
        if value is None or str(value).strip() == "":
            return 1
        return max(1, int(value))


@lru_cache
def get_settings() -> Settings:
    return Settings()
