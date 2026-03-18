from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from core.temporal_state import CurrentTimeContext


def local_now() -> datetime:
    return datetime.now().astimezone()


class TemporalEngine:
    """
    Handles prototype temporal awareness:
    - current local date and time
    - day of week
    - morning / afternoon / evening / night
    - before / after
    - yesterday / today / tomorrow
    - elapsed time
    """

    @staticmethod
    def now() -> datetime:
        return local_now()

    @staticmethod
    def today_label(reference: Optional[datetime] = None) -> str:
        ref = reference or local_now()
        return ref.strftime("%Y-%m-%d")

    @staticmethod
    def time_label(reference: Optional[datetime] = None) -> str:
        ref = reference or local_now()
        return ref.strftime("%I:%M %p")

    @staticmethod
    def weekday_name(reference: Optional[datetime] = None) -> str:
        ref = reference or local_now()
        return ref.strftime("%A")

    @staticmethod
    def day_part(reference: Optional[datetime] = None) -> str:
        ref = reference or local_now()
        hour = ref.hour

        if 6 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    @staticmethod
    def yesterday(reference: Optional[datetime] = None) -> datetime:
        ref = reference or local_now()
        return ref - timedelta(days=1)

    @staticmethod
    def tomorrow(reference: Optional[datetime] = None) -> datetime:
        ref = reference or local_now()
        return ref + timedelta(days=1)

    @staticmethod
    def is_before(a: datetime, b: datetime) -> bool:
        return a < b

    @staticmethod
    def is_after(a: datetime, b: datetime) -> bool:
        return a > b

    @staticmethod
    def elapsed_seconds(start: datetime, end: Optional[datetime] = None) -> float:
        ref = end or local_now()
        return (ref - start).total_seconds()

    @staticmethod
    def relative_day_label(target: datetime, reference: Optional[datetime] = None) -> str:
        ref = reference or local_now()
        target_date = target.date()
        ref_date = ref.date()

        if target_date == ref_date:
            return "today"
        if target_date == (ref_date - timedelta(days=1)):
            return "yesterday"
        if target_date == (ref_date + timedelta(days=1)):
            return "tomorrow"
        if target_date < ref_date:
            return "past"
        return "future"

    @staticmethod
    def current_time_context(reference: Optional[datetime] = None) -> CurrentTimeContext:
        ref = reference or local_now()

        return CurrentTimeContext(
            timestamp=ref,
            iso_datetime=ref.isoformat(),
            date_label=ref.strftime("%Y-%m-%d"),
            time_label=ref.strftime("%I:%M %p"),
            weekday_name=ref.strftime("%A"),
            day_part=TemporalEngine.day_part(ref),
            relative_day_label="today",
        )