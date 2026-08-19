from __future__ import annotations

from collections import defaultdict
from typing import Awaitable, Callable

from app.schemas import MeetingEvent


EventHandler = Callable[[MeetingEvent], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    async def publish(self, event: MeetingEvent) -> None:
        for handler in self._subscribers.get(event.type, []):
            await handler(event)
        for handler in self._subscribers.get("*", []):
            await handler(event)

