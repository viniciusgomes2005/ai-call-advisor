from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = ""

    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()

