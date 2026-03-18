from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas import VexisState
from core.state_manager import StateManager


@dataclass
class CapabilityProfile:
    speech_output: bool = True
    speech_input: bool = False
    social_response: bool = True
    deterministic_reasoning: bool = True
    journal_ingest: bool = False
    background_cognition: bool = True
    autonomous_actions: bool = True
    internet_access: bool = False
    file_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "speech_output": self.speech_output,
            "speech_input": self.speech_input,
            "social_response": self.social_response,
            "deterministic_reasoning": self.deterministic_reasoning,
            "journal_ingest": self.journal_ingest,
            "background_cognition": self.background_cognition,
            "autonomous_actions": self.autonomous_actions,
            "internet_access": self.internet_access,
            "file_review": self.file_review,
        }


@dataclass
class IdentityProfile:
    name: str = "VEX"
    system_label: str = "VEXIS"
    personality_mode: str = "dry_sardonic"
    role_description: str = "language layer and deterministic cognition system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_label": self.system_label,
            "personality_mode": self.personality_mode,
            "role_description": self.role_description,
        }


@dataclass
class RuntimeProfile:
    device_label: str = "unknown_device"
    location_label: str = "unknown_location"
    session_id: str = "unknown_session"
    timezone_name: str = "unknown_timezone"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_label": self.device_label,
            "location_label": self.location_label,
            "session_id": self.session_id,
            "timezone_name": self.timezone_name,
        }


@dataclass
class SelfModelSnapshot:
    identity: IdentityProfile
    capabilities: CapabilityProfile
    runtime: RuntimeProfile
    current_presence_state: str
    current_status_text: str
    is_speaking: bool
    is_thinking: bool
    active_task_count: int
    open_question_count: int
    unsupported_claim_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "runtime": self.runtime.to_dict(),
            "current_presence_state": self.current_presence_state,
            "current_status_text": self.current_status_text,
            "is_speaking": self.is_speaking,
            "is_thinking": self.is_thinking,
            "active_task_count": self.active_task_count,
            "open_question_count": self.open_question_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "notes": self.notes,
        }


class SelfModel:
    """
    Internal model of what VEX is, what VEX can do, and where VEX is running.

    This gives VEX grounded answers to questions like:
    - who are you
    - what are you
    - where are you
    - what can you do
    - are you thinking
    """

    def __init__(
        self,
        state_manager: StateManager,
        identity: Optional[IdentityProfile] = None,
        capabilities: Optional[CapabilityProfile] = None,
    ) -> None:
        self.state_manager = state_manager
        self.identity = identity or IdentityProfile()
        self.capabilities = capabilities or CapabilityProfile()

    def snapshot(self) -> SelfModelSnapshot:
        state = self.state_manager.get_state()

        runtime = RuntimeProfile(
            device_label=self.state_manager.device_label,
            location_label=self.state_manager.location_label,
            session_id=self.state_manager.session_id,
            timezone_name=self.state_manager.time_builder.default_timezone,
        )

        notes: list[str] = []

        if state.cognition.is_thinking:
            notes.append("core cognition is currently active")
        if state.speech.is_speaking:
            notes.append("speech output is currently active")
        if len(state.epistemic.open_questions) > 0:
            notes.append("there are unresolved questions under review")
        if len(state.epistemic.unsupported_claims) > 0:
            notes.append("there are unsupported claims awaiting evidence review")

        return SelfModelSnapshot(
            identity=self.identity,
            capabilities=self.capabilities,
            runtime=runtime,
            current_presence_state=state.presence.visual_state,
            current_status_text=state.presence.status_text,
            is_speaking=state.speech.is_speaking,
            is_thinking=state.cognition.is_thinking,
            active_task_count=len(state.tasks.active_task_ids),
            open_question_count=len(state.epistemic.open_questions),
            unsupported_claim_count=len(state.epistemic.unsupported_claims),
            notes=notes,
        )

    def answer_identity_question(self, question_text: str) -> Optional[str]:
        snapshot = self.snapshot()
        normalized = (question_text or "").strip().lower()

        if "who are you" in normalized or "what are you" in normalized:
            return (
                f"I am {snapshot.identity.system_label}, with the active identity name "
                f"{snapshot.identity.name}. My role is {snapshot.identity.role_description}."
            )

        if "where are you" in normalized or "where is here" in normalized:
            return (
                f'I am running on {snapshot.runtime.device_label} at location context '
                f'"{snapshot.runtime.location_label}".'
            )

        if "what can you do" in normalized:
            enabled = []
            for key, value in snapshot.capabilities.to_dict().items():
                if value:
                    enabled.append(key.replace("_", " "))

            capability_text = ", ".join(enabled) if enabled else "very little at the moment"
            return f"My currently enabled capabilities include: {capability_text}."

        if "are you thinking" in normalized:
            if snapshot.is_thinking:
                return "Yes. Core cognition is active right now."
            return "Not at this exact moment, but the core remains available."

        if "are you speaking" in normalized:
            if snapshot.is_speaking:
                return "Yes. Speech output is active."
            return "No. Speech output is currently idle."

        if "what is your name" in normalized:
            return f"My active identity name is {snapshot.identity.name}."

        return None

    def capability_summary(self) -> dict[str, Any]:
        return self.snapshot().to_dict()

    def set_capability(self, name: str, value: bool) -> None:
        if hasattr(self.capabilities, name):
            setattr(self.capabilities, name, value)

    def set_identity_name(self, name: str) -> None:
        self.identity.name = name

    def set_personality_mode(self, mode: str) -> None:
        self.identity.personality_mode = mode