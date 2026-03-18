from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.autonomy_engine import AutonomyAction
from core.schemas import VexisState


@dataclass
class OutputDecision:
    allow_gui_output: bool
    allow_speech_output: bool
    reason: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_gui_output": self.allow_gui_output,
            "allow_speech_output": self.allow_speech_output,
            "reason": self.reason,
            "text": self.text,
            "metadata": self.metadata,
        }


class OutputGate:
    """
    Controls what internal VEX actions may become visible or spoken.

    Important:
    - cognition and autonomy continue whether or not output is allowed
    - mute should only affect audible output
    - internal actions may still be logged even if not spoken
    """

    def __init__(
        self,
        speech_enabled: bool = True,
        spontaneous_output_enabled: bool = True,
        minimum_idle_seconds_for_spontaneous_output: int = 45,
    ) -> None:
        self.speech_enabled = speech_enabled
        self.spontaneous_output_enabled = spontaneous_output_enabled
        self.minimum_idle_seconds_for_spontaneous_output = minimum_idle_seconds_for_spontaneous_output

        self._last_user_activity_at = time.time()
        self._last_spoken_output_at = 0.0

    def note_user_activity(self) -> None:
        self._last_user_activity_at = time.time()

    def note_spoken_output(self) -> None:
        self._last_spoken_output_at = time.time()

    def set_speech_enabled(self, enabled: bool) -> None:
        self.speech_enabled = enabled

    def set_spontaneous_output_enabled(self, enabled: bool) -> None:
        self.spontaneous_output_enabled = enabled

    def evaluate_action(
        self,
        action: AutonomyAction,
        state: VexisState,
        current_time: Optional[float] = None,
    ) -> OutputDecision:
        now = current_time or time.time()

        if action.internal_only:
            return OutputDecision(
                allow_gui_output=False,
                allow_speech_output=False,
                reason="internal_only_action",
                text=action.speech_text or action.description,
                metadata={"action_type": action.action_type},
            )

        if not self.spontaneous_output_enabled:
            return OutputDecision(
                allow_gui_output=False,
                allow_speech_output=False,
                reason="spontaneous_output_disabled",
                text=action.speech_text or action.description,
                metadata={"action_type": action.action_type},
            )

        if state.speech.is_speaking:
            return OutputDecision(
                allow_gui_output=False,
                allow_speech_output=False,
                reason="speech_already_active",
                text=action.speech_text or action.description,
                metadata={"action_type": action.action_type},
            )

        if state.cognition.is_thinking:
            return OutputDecision(
                allow_gui_output=False,
                allow_speech_output=False,
                reason="core_busy_thinking",
                text=action.speech_text or action.description,
                metadata={"action_type": action.action_type},
            )

        idle_time = now - self._last_user_activity_at
        if idle_time < self.minimum_idle_seconds_for_spontaneous_output:
            return OutputDecision(
                allow_gui_output=False,
                allow_speech_output=False,
                reason="user_not_idle_long_enough",
                text=action.speech_text or action.description,
                metadata={
                    "action_type": action.action_type,
                    "idle_time_seconds": round(idle_time, 2),
                },
            )

        text = action.speech_text or action.description

        return OutputDecision(
            allow_gui_output=True,
            allow_speech_output=self.speech_enabled and action.should_speak,
            reason="spontaneous_output_allowed",
            text=text,
            metadata={
                "action_type": action.action_type,
                "idle_time_seconds": round(idle_time, 2),
                "speech_enabled": self.speech_enabled,
            },
        )

    def evaluate_direct_response(
        self,
        text: str,
        state: VexisState,
    ) -> OutputDecision:
        """
        Used for direct user-facing responses to actual prompts.
        These should almost always be allowed to appear in the GUI.
        Speech is controlled separately by speech_enabled.
        """
        return OutputDecision(
            allow_gui_output=True,
            allow_speech_output=self.speech_enabled and bool(text.strip()),
            reason="direct_response",
            text=text,
            metadata={
                "speech_enabled": self.speech_enabled,
                "is_speaking": state.speech.is_speaking,
            },
        )