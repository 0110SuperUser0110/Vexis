from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass
class TemporalMarker:
    label: str
    timestamp: datetime = field(default_factory=local_now)
    sequence_id: int = 0
    previous_label: Optional[str] = None
    next_expected_label: Optional[str] = None


@dataclass
class CurrentTimeContext:
    timestamp: datetime = field(default_factory=local_now)
    iso_datetime: str = ""
    date_label: str = ""
    time_label: str = ""
    weekday_name: str = ""
    day_part: str = ""
    relative_day_label: str = "today"