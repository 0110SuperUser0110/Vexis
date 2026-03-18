from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


class EventBus:
    """
    Lightweight event queue for the VEXIS prototype.
    """

    def __init__(self) -> None:
        self._queue: Deque[Event] = deque()

    def publish(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._queue.append(Event(event_type=event_type, payload=payload or {}))

    def has_events(self) -> bool:
        return len(self._queue) > 0

    def next_event(self) -> Optional[Event]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def size(self) -> int:
        return len(self._queue)