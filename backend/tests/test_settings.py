from __future__ import annotations

from app.settings import Settings


def test_language_env_locale_list_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("LANGUAGE", "pt_BR:en")

    settings = Settings(_env_file=None)

    assert settings.language == "pt"


def test_language_region_tag_is_normalized() -> None:
    settings = Settings(language="pt-BR")

    assert settings.language == "pt"


def test_whisper_defaults_match_local_asr_target() -> None:
    settings = Settings(_env_file=None)

    assert settings.whisper_model == "small"
    assert settings.whisper_device == "auto"
    assert settings.whisper_compute_type == "auto"
    assert settings.whisper_language is None
    assert settings.whisper_beam_size == 1
    assert settings.whisper_condition_on_previous_text is False
    assert settings.asr_parallel_workers == 2
    assert settings.asr_min_speech_ms == 500
    assert settings.asr_silence_end_ms == 450
    assert settings.asr_max_utterance_ms == 6000


def test_asr_parallel_workers_is_clamped() -> None:
    settings = Settings(asr_parallel_workers=0)

    assert settings.asr_parallel_workers == 1
