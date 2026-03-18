from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass
class TimeContext:
    utc_timestamp: str
    local_timestamp: str
    timezone_name: str
    utc_offset: str
    unix_timestamp: float
    day_of_week: str
    part_of_day: str
    is_weekend: bool
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int

    def to_dict(self) -> dict:
        return asdict(self)


class TimeContextBuilder:
    """
    Builds full temporal context for VEXIS events.

    Design:
    - UTC is the objective ordering clock
    - local time is the experienced time
    - timezone and offset preserve context
    - semantic time fields support temporal reasoning
    """

    def __init__(self, default_timezone: str = "America/Chicago") -> None:
        self.default_timezone = default_timezone

    def now(self, timezone_name: str | None = None) -> TimeContext:
        tz_name = timezone_name or self.default_timezone

        utc_now = datetime.now(UTC).replace(microsecond=0)
        local_now, resolved_tz_name = self._resolve_local_time(utc_now, tz_name)

        return TimeContext(
            utc_timestamp=utc_now.isoformat().replace("+00:00", "Z"),
            local_timestamp=local_now.replace(tzinfo=None).isoformat(),
            timezone_name=resolved_tz_name,
            utc_offset=self._format_offset(local_now),
            unix_timestamp=utc_now.timestamp(),
            day_of_week=local_now.strftime("%A"),
            part_of_day=self._part_of_day(local_now.hour),
            is_weekend=local_now.weekday() >= 5,
            year=local_now.year,
            month=local_now.month,
            day=local_now.day,
            hour=local_now.hour,
            minute=local_now.minute,
            second=local_now.second,
        )

    def from_datetime(
        self,
        dt: datetime,
        timezone_name: str | None = None,
    ) -> TimeContext:
        tz_name = timezone_name or self.default_timezone

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)

        dt = dt.replace(microsecond=0)
        local_dt, resolved_tz_name = self._resolve_local_time(dt, tz_name)

        return TimeContext(
            utc_timestamp=dt.isoformat().replace("+00:00", "Z"),
            local_timestamp=local_dt.replace(tzinfo=None).isoformat(),
            timezone_name=resolved_tz_name,
            utc_offset=self._format_offset(local_dt),
            unix_timestamp=dt.timestamp(),
            day_of_week=local_dt.strftime("%A"),
            part_of_day=self._part_of_day(local_dt.hour),
            is_weekend=local_dt.weekday() >= 5,
            year=local_dt.year,
            month=local_dt.month,
            day=local_dt.day,
            hour=local_dt.hour,
            minute=local_dt.minute,
            second=local_dt.second,
        )

    def enrich_event(
        self,
        event: dict,
        timezone_name: str | None = None,
    ) -> dict:
        enriched = dict(event)
        enriched["time_context"] = self.now(timezone_name=timezone_name).to_dict()
        return enriched

    def delta_summary(self, earlier_unix: float, later_unix: float) -> dict:
        seconds = max(0.0, later_unix - earlier_unix)

        return {
            "seconds": int(seconds),
            "minutes": round(seconds / 60, 2),
            "hours": round(seconds / 3600, 2),
            "days": round(seconds / 86400, 3),
        }

    def _resolve_local_time(self, utc_dt: datetime, tz_name: str) -> tuple[datetime, str]:
        try:
            tz = ZoneInfo(tz_name)
            return utc_dt.astimezone(tz), tz_name
        except ZoneInfoNotFoundError:
            # Fallback to the machine's local timezone if tzdata / named zone is unavailable.
            local_dt = utc_dt.astimezone()
            machine_tz_name = getattr(local_dt.tzinfo, "key", None) or str(local_dt.tzinfo) or "Local"
            return local_dt, machine_tz_name

    def _format_offset(self, dt: datetime) -> str:
        offset = dt.utcoffset()
        if offset is None:
            return "+00:00"

        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{sign}{hours:02d}:{minutes:02d}"

    def _part_of_day(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"