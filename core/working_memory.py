from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class WorkingMemoryItem:
    item_id: str
    content: str
    item_type: str
    priority: float = 0.5
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    resolved: bool = False
    notes: Optional[str] = None


class WorkingMemory:
    """
    Active short-term workspace for VEXIS.
    Tracks what is currently being considered.
    """

    def __init__(self) -> None:
        self._items: Dict[str, WorkingMemoryItem] = {}

    def add_item(self, item: WorkingMemoryItem) -> None:
        self._items[item.item_id] = item

    def get_item(self, item_id: str) -> Optional[WorkingMemoryItem]:
        return self._items.get(item_id)

    def list_active_items(self) -> List[WorkingMemoryItem]:
        return [item for item in self._items.values() if not item.resolved]

    def mark_resolved(self, item_id: str, notes: Optional[str] = None) -> bool:
        item = self._items.get(item_id)
        if item is None:
            return False

        item.resolved = True
        item.updated_at = utc_now()
        item.notes = notes
        return True

    def highest_priority_item(self) -> Optional[WorkingMemoryItem]:
        active_items = self.list_active_items()
        if not active_items:
            return None
        return sorted(active_items, key=lambda x: x.priority, reverse=True)[0]

    def size(self) -> int:
        return len(self._items)