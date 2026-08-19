from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(slots=True)
class AudioChunk:
    source: str
    data: bytes
    sample_rate: int | None = None
    timestamp_ms: int | None = None


class MeetingAudioSource(ABC):
    @abstractmethod
    async def chunks(self) -> AsyncIterator[AudioChunk]:
        raise NotImplementedError


class BrowserTabAudioSource(MeetingAudioSource):
    """Marker abstraction for Chrome tab capture events delivered by WebSocket."""

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        if False:
            yield AudioChunk(source="REMOTE_AUDIO", data=b"")


class FileReplaySource(MeetingAudioSource):
    def __init__(self, path: str):
        self.path = path

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        with open(self.path, "rb") as file:
            yield AudioChunk(source="FILE_AUDIO", data=file.read())

