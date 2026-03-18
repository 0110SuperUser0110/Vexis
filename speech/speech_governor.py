from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SpeechPolicy:
    user_present: bool = True
    muted: bool = False
    allow_reflective: bool = False
    allow_creative: bool = False
    unsolicited_cooldown_seconds: int = 300


class SpeechGovernor:
    """
    Decides whether VEXIS is allowed to speak.
    """

    def __init__(self, policy: Optional[SpeechPolicy] = None) -> None:
        self.policy = policy or SpeechPolicy()
        self._last_unsolicited_at: Optional[datetime] = None

    def can_speak(self, speech_type: str, user_present: Optional[bool] = None) -> tuple[bool, str]:
        present = self.policy.user_present if user_present is None else user_present

        if self.policy.muted:
            return False, "speech is muted"

        if speech_type == "response":
            return True, "direct response allowed"

        if not present:
            return False, "user not present"

        if speech_type == "reflective" and not self.policy.allow_reflective:
            return False, "reflective speech disabled"

        if speech_type == "creative" and not self.policy.allow_creative:
            return False, "creative speech disabled"

        now = utc_now()
        if self._last_unsolicited_at is not None:
            delta = now - self._last_unsolicited_at
            if delta < timedelta(seconds=self.policy.unsolicited_cooldown_seconds):
                return False, "unsolicited speech cooldown active"

        return True, "speech allowed"

    def mark_spoken(self, speech_type: str) -> None:
        if speech_type != "response":
            self._last_unsolicited_at = utc_now()