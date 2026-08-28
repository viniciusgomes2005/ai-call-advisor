from __future__ import annotations

import abc
import asyncio
import ctypes
import logging
import math
import os
import re
import shutil
import subprocess
import sysconfig
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.audio.normalizer import AudioNormalizer
from app.schemas import ASRStatusResponse, ASRTranscriptionResponse, TranscriptSegment


logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm"}
SAP_TERMS = ["SAP", "ECC", "S/4HANA", "BAPI", "IDoc", "EWM", "Ariba", "Fiori", "ABAP", "OData"]
SAP_DOMAIN_PROMPT = "This meeting discusses SAP, ECC, S/4HANA, BAPI, IDoc, EWM, Ariba, Fiori, ABAP and OData."

IGNORED_TRANSCRIPT_FRAGMENTS = (
    "legendas pela comunidade",
    "amara.org",
    "transcricao e legendas",
    "transcrição e legendas",
    "subtitles by",
    "captioned by",
    "obrigado por assistir",
)

IGNORED_SHORT_UTTERANCES = {
    "ah",
    "aham",
    "hm",
    "hum",
    "hmm",
    "ok",
    "ta",
    "tá",
}

TRANSCRIPT_REPLACEMENTS = (
    (re.compile(r"\badivisor\b", re.IGNORECASE), "Advisor"),
    (re.compile(r"\bwith get funculate\b", re.IGNORECASE), "widget flutuante"),
    (re.compile(r"\bwidget funculate\b", re.IGNORECASE), "widget flutuante"),
    (re.compile(r"\bwidget flutante\b", re.IGNORECASE), "widget flutuante"),
    (re.compile(r"\bo estacion do chrome\b", re.IGNORECASE), "extensão do Chrome"),
    (re.compile(r"\bo esclimição do chrome\b", re.IGNORECASE), "extensão do Chrome"),
    (re.compile(r"\ba estens[aã]o do chrome\b", re.IGNORECASE), "a extensão do Chrome"),
)


@dataclass(slots=True)
class WhisperRuntimeConfig:
    device: str
    compute_type: str
    cuda_available: bool


@dataclass(slots=True)
class ASRSegment:
    speaker: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    latency_ms: int | None = None
    language: str | None = None
    confidence: float | None = None


def normalize_transcript_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def correct_transcript_terms(text: str) -> str:
    corrected = normalize_transcript_text(text)
    for pattern, replacement in TRANSCRIPT_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return corrected


def should_ignore_transcript(text: str) -> bool:
    normalized = normalize_transcript_text(text)
    lowered = normalized.lower()
    asciiish = lowered.translate(
        str.maketrans({"ç": "c", "ã": "a", "á": "a", "à": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"})
    )
    stripped = re.sub(r"[^\w\s]", "", lowered).strip()
    alpha_count = sum(character.isalpha() for character in normalized)
    if alpha_count < 4:
        return True
    if stripped in IGNORED_SHORT_UTTERANCES:
        return True
    return any(fragment in lowered or fragment in asciiish for fragment in IGNORED_TRANSCRIPT_FRAGMENTS)


def segment_is_low_confidence(segment: object) -> bool:
    no_speech_prob = getattr(segment, "no_speech_prob", None)
    avg_logprob = getattr(segment, "avg_logprob", None)
    compression_ratio = getattr(segment, "compression_ratio", None)
    if isinstance(no_speech_prob, (int, float)) and no_speech_prob > 0.65:
        return True
    if isinstance(avg_logprob, (int, float)) and avg_logprob < -1.15:
        return True
    if isinstance(compression_ratio, (int, float)) and compression_ratio > 2.4:
        return True
    return False


def confidence_from_segment(segment: object) -> float | None:
    avg_logprob = getattr(segment, "avg_logprob", None)
    if not isinstance(avg_logprob, (int, float)):
        return None
    return max(0.0, min(1.0, math.exp(float(avg_logprob))))


def cuda_device_count() -> int:
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=1, check=False)
        if result.returncode == 0 and "GPU" in result.stdout:
            return len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception as exc:
        logger.info("Could not inspect CUDA availability through nvidia-smi: %s", exc)
        return 0
    return 0


def cuda_runtime_loadable() -> bool:
    preload_nvidia_runtime_libraries()
    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError as exc:
        logger.warning("CUDA device detected but libcublas.so.12 is not loadable: %s", exc)
        return False


def preload_nvidia_runtime_libraries() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    lib_dirs = [
        site_packages / "nvidia" / "cuda_nvrtc" / "lib",
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    existing_dirs = [str(path) for path in lib_dirs if path.exists()]
    if existing_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        known = [item for item in current.split(":") if item]
        merged = existing_dirs + [item for item in known if item not in existing_dirs]
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    for path in (
        site_packages / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so.12",
        site_packages / "nvidia" / "cublas" / "lib" / "libcublas.so.12",
        site_packages / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9",
        site_packages / "nvidia" / "cudnn" / "lib" / "libcudnn_ops.so.9",
    ):
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.debug("Could not preload NVIDIA library %s: %s", path, exc)


def resolve_whisper_runtime(device: str = "auto", compute_type: str = "auto") -> WhisperRuntimeConfig:
    requested_device = device.strip().lower() or "auto"
    requested_compute = compute_type.strip().lower() or "auto"
    cuda_available = cuda_device_count() > 0
    cuda_usable = cuda_available and cuda_runtime_loadable()

    if requested_device == "auto":
        if cuda_usable:
            resolved_device = "cuda"
            logger.info("Detected NVIDIA CUDA device")
        else:
            resolved_device = "cpu"
            logger.info("CUDA unavailable or missing runtime libraries")
    else:
        resolved_device = requested_device
        if resolved_device == "cuda" and not cuda_usable:
            logger.warning("CUDA was requested but the CUDA runtime is not fully available")

    if requested_compute == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute = requested_compute

    logger.info(
        "Using %s with %s",
        resolved_device.upper() if resolved_device == "cpu" else resolved_device,
        resolved_compute,
    )
    return WhisperRuntimeConfig(resolved_device, resolved_compute, cuda_available)


async def run_blocking(func, *args):
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="faster-whisper") as executor:
        future = executor.submit(func, *args)
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()


class ASRProvider(abc.ABC):
    @abc.abstractmethod
    async def load(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def transcribe_file(self, path: Path, language: str | None = None) -> ASRTranscriptionResponse:
        raise NotImplementedError

    @abc.abstractmethod
    async def transcribe_audio_chunk(
        self,
        data: bytes,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        audio_format: str = "pcm_s16le",
    ) -> ASRTranscriptionResponse:
        raise NotImplementedError

    @abc.abstractmethod
    def status(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def status_payload(self) -> ASRStatusResponse:
        raise NotImplementedError


class FasterWhisperProvider(ASRProvider):
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
        initial_prompt: str = "",
        hotwords: str = "",
        language: str | None = None,
        beam_size: int = 5,
        vad_min_silence_duration_ms: int = 500,
        condition_on_previous_text: bool = True,
        use_domain_prompt: bool = False,
        warmup: bool = False,
        parallel_workers: int = 1,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.initial_prompt = initial_prompt
        self.hotwords = hotwords
        self.language = language
        self.beam_size = beam_size
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.condition_on_previous_text = condition_on_previous_text
        self.use_domain_prompt = use_domain_prompt
        self.warmup_enabled = warmup
        self.parallel_workers = max(1, parallel_workers)
        self.normalizer = AudioNormalizer()
        self.runtime = resolve_whisper_runtime(device, compute_type)
        self._model: Any | None = None
        self._load_error: Exception | None = None
        self._loading = False
        self._load_lock = asyncio.Lock()
        self._transcribe_semaphore = asyncio.Semaphore(self.parallel_workers)

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            self._loading = True
            try:
                self._model = await run_blocking(self._load_sync)
                self._load_error = None
                if self.warmup_enabled:
                    await run_blocking(self._warmup_sync)
            except Exception as exc:
                self._load_error = exc
                raise
            finally:
                self._loading = False

    def _load_sync(self):
        WhisperModel = self._whisper_model_class()
        candidates = [(self.runtime.device, self.runtime.compute_type)]
        if self.requested_device == "auto" and self.runtime.device == "cuda":
            candidates.extend([("cuda", "int8_float16"), ("cpu", "int8")])

        last_error: Exception | None = None
        for device, compute_type in candidates:
            try:
                logger.info("Loading Whisper %s with %s on %s", self.model_name, compute_type, device)
                model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
                self._validate_model_sync(model)
                self.runtime = WhisperRuntimeConfig(
                    device=device,
                    compute_type=compute_type,
                    cuda_available=self.runtime.cuda_available,
                )
                return model
            except Exception as exc:
                last_error = exc
                logger.exception("Failed loading Whisper %s with %s on %s", self.model_name, compute_type, device)
                if self.requested_device != "auto":
                    break
        assert last_error is not None
        raise last_error

    def _warmup_sync(self) -> None:
        if self._model is None:
            return
        self._validate_model_sync(self._model)

    def _validate_model_sync(self, model) -> None:
        samples = np.zeros(1_600, dtype=np.float32)
        list(model.transcribe(samples, language=self.language, beam_size=1, vad_filter=False)[0])

    @staticmethod
    def _whisper_model_class():
        from faster_whisper import WhisperModel

        return WhisperModel

    def status(self) -> str:
        payload = self.status_payload()
        if payload.status in {"ready", "loading", "idle"}:
            return "ok"
        return payload.status

    def status_payload(self) -> ASRStatusResponse:
        if self._model is not None:
            return ASRStatusResponse(
                status="ready",
                model=self.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                language=self.language,
                detail=f"parallel_workers={self.parallel_workers}",
            )
        if self._loading:
            return ASRStatusResponse(
                status="loading",
                model=self.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                language=self.language,
            )
        if self._load_error is not None:
            return ASRStatusResponse(
                status="error",
                model=self.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                language=self.language,
                detail=str(self._load_error),
            )
        try:
            self._whisper_model_class()
        except Exception as exc:
            return ASRStatusResponse(
                status="unavailable",
                model=self.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                language=self.language,
                detail=str(exc),
            )
        return ASRStatusResponse(
            status="idle",
            model=self.model_name,
            device=self.runtime.device,
            compute_type=self.runtime.compute_type,
            language=self.language,
            detail="model is available and will load on first transcription",
        )

    async def process_audio_file(
        self, path: Path, language: str | None = None, speaker: str = "UNKNOWN"
    ) -> list[ASRSegment]:
        result = await self.transcribe_file(path, language=language)
        return self._legacy_segments(result, speaker)

    async def process_audio(
        self,
        data: bytes,
        suffix: str = ".wav",
        language: str | None = None,
        speaker: str = "UNKNOWN",
    ) -> list[ASRSegment]:
        if suffix == ".pcm":
            result = await self.transcribe_audio_chunk(data, language=language, audio_format="pcm_s16le")
        else:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
                file.write(data)
                temp_path = Path(file.name)
            try:
                result = await self.transcribe_file(temp_path, language=language)
            finally:
                temp_path.unlink(missing_ok=True)
        return self._legacy_segments(result, speaker)

    async def transcribe_file(self, path: Path, language: str | None = None) -> ASRTranscriptionResponse:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio file type: {suffix or '<none>'}")
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError("empty audio file")
        if self._model is None:
            await self.load()
        async with self._transcribe_semaphore:
            return await run_blocking(self._transcribe_path_sync, path, language)

    async def transcribe_audio_chunk(
        self,
        data: bytes,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        audio_format: str = "pcm_s16le",
    ) -> ASRTranscriptionResponse:
        if not data:
            raise ValueError("empty audio")
        if self._model is None:
            await self.load()
        async with self._transcribe_semaphore:
            return await run_blocking(self._transcribe_chunk_sync, data, sample_rate, language, audio_format)

    def _transcribe_path_sync(self, path: Path, language: str | None) -> ASRTranscriptionResponse:
        normalized = self.normalizer.from_file(path)
        return self._transcribe_samples_sync(normalized.samples, normalized.duration_seconds, language)

    def _transcribe_chunk_sync(
        self, data: bytes, sample_rate: int, language: str | None, audio_format: str
    ) -> ASRTranscriptionResponse:
        if audio_format != "pcm_s16le":
            suffix = audio_format if audio_format.startswith(".") else f".{audio_format}"
            normalized = self.normalizer.from_bytes(data, suffix=suffix)
        else:
            normalized = self.normalizer.from_pcm16(data, sample_rate=sample_rate)
        return self._transcribe_samples_sync(normalized.samples, normalized.duration_seconds, language)

    def _transcribe_samples_sync(
        self, samples: np.ndarray, duration_seconds: float, language: str | None
    ) -> ASRTranscriptionResponse:
        if samples.size == 0:
            raise ValueError("empty audio")
        model = self._model
        if model is None:
            try:
                self._loading = True
                model = self._load_sync()
                self._model = model
                self._load_error = None
            except Exception as exc:
                self._load_error = exc
                raise
            finally:
                self._loading = False
        started = time.perf_counter()
        selected_language = language if language is not None else self.language
        logger.info(
            "Transcribing audio with Whisper: duration=%.2fs language=%s device=%s compute_type=%s samples=%d",
            duration_seconds,
            selected_language or "auto",
            self.runtime.device,
            self.runtime.compute_type,
            samples.size,
        )
        segments_iter, info = model.transcribe(
            samples,
            language=selected_language,
            initial_prompt=self._initial_prompt(),
            hotwords=self.hotwords or None,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.vad_min_silence_duration_ms},
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        detected_language = getattr(info, "language", None)
        audio_duration = float(getattr(info, "duration", duration_seconds) or duration_seconds)
        segments: list[TranscriptSegment] = []
        for segment in segments_iter:
            text = correct_transcript_terms(segment.text)
            if text and not should_ignore_transcript(text) and not segment_is_low_confidence(segment):
                segments.append(
                    TranscriptSegment(
                        start=float(segment.start),
                        end=float(segment.end),
                        text=text,
                        language=detected_language,
                        confidence=confidence_from_segment(segment),
                    )
                )
        processing_time_seconds = time.perf_counter() - started
        real_time_factor = processing_time_seconds / audio_duration if audio_duration > 0 else None
        logger.info(
            "Whisper transcription finished: segments=%d duration=%.2fs latency_ms=%d rtf=%s language=%s",
            len(segments),
            audio_duration,
            int(processing_time_seconds * 1000),
            f"{real_time_factor:.2f}" if real_time_factor is not None else "-",
            detected_language or "unknown",
        )
        return ASRTranscriptionResponse(
            language=detected_language,
            duration=audio_duration,
            audio_duration_seconds=audio_duration,
            processing_time_seconds=processing_time_seconds,
            processing_time_ms=int(processing_time_seconds * 1000),
            real_time_factor=real_time_factor,
            segments=segments,
            model=self.model_name,
            device=self.runtime.device,
            compute_type=self.runtime.compute_type,
        )

    def _initial_prompt(self) -> str | None:
        parts = []
        if self.initial_prompt:
            parts.append(self.initial_prompt)
        if self.use_domain_prompt:
            parts.append(SAP_DOMAIN_PROMPT)
        return "\n".join(parts) or None

    @staticmethod
    def _legacy_segments(result: ASRTranscriptionResponse, speaker: str) -> list[ASRSegment]:
        return [
            ASRSegment(
                speaker=speaker,
                text=segment.text,
                start_ms=int(segment.start * 1000),
                end_ms=int(segment.end * 1000),
                latency_ms=result.processing_time_ms,
                language=segment.language,
                confidence=segment.confidence,
            )
            for segment in result.segments
        ]
