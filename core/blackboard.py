from __future__ import annotations

from typing import List, Optional

from core.schemas import BlackboardEntry


class Blackboard:
    """
    Shared internal workspace for module observations and internal notes.
    """

    def __init__(self) -> None:
        self._entries: List[BlackboardEntry] = []

    def post(self, entry: BlackboardEntry) -> None:
        self._entries.append(entry)

    def all_entries(self) -> List[BlackboardEntry]:
        return list(self._entries)

    def highest_priority(self) -> Optional[BlackboardEntry]:
        if not self._entries:
            return None
        return sorted(self._entries, key=lambda e: e.priority, reverse=True)[0]

    def recent(self, limit: int = 10) -> List[BlackboardEntry]:
        return self._entries[-limit:]

    def size(self) -> int:
        return len(self._entries)