from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class VoiceQueueItem:
    text: str
    speech_type: str = "status"
    created_at: datetime = field(default_factory=utc_now)


class VoiceQueue:
    def __init__(self) -> None:
        self._queue: Deque[VoiceQueueItem] = deque()

    def enqueue(self, text: str, speech_type: str = "status") -> None:
        self._queue.append(VoiceQueueItem(text=text, speech_type=speech_type))

    def dequeue(self) -> Optional[VoiceQueueItem]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0