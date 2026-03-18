from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class LifecycleState:
    mode: str = "active"
    boot_count: int = 0
    started_at: str = ""
    last_active_at: str = ""


@dataclass
class PresenceState:
    visual_state: str = "idle"
    status_text: str = "ready"


@dataclass
class SpeechState:
    is_speaking: bool = False
    last_spoken_text: str = ""
    queue_length: int = 0


@dataclass
class CognitionState:
    is_thinking: bool = False
    current_focus: Optional[str] = None
    current_input_id: Optional[str] = None
    last_classification: Optional[str] = None


@dataclass
class IngestState:
    pending_files: list[str] = field(default_factory=list)
    processing_file: Optional[str] = None
    completed_files: list[str] = field(default_factory=list)


@dataclass
class MemoryState:
    total_items: int = 0
    last_memory_id: Optional[str] = None


@dataclass
class TaskState:
    active_task_ids: list[str] = field(default_factory=list)
    unresolved_task_ids: list[str] = field(default_factory=list)
    completed_task_ids: list[str] = field(default_factory=list)


@dataclass
class EpistemicState:
    open_questions: list[str] = field(default_factory=list)
    open_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


@dataclass
class InputRecord:
    input_id: str
    timestamp: str
    source: str
    raw_text: str
    input_type: str = "unknown"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    time_context: dict[str, Any] = field(default_factory=dict)
    interaction_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    memory_id: str
    timestamp: str
    kind: str
    content: str
    source: str
    related_input_id: Optional[str] = None
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    time_context: dict[str, Any] = field(default_factory=dict)
    interaction_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRecord:
    task_id: str
    timestamp: str
    title: str
    description: str
    status: str = "active"
    priority: int = 0
    related_input_id: Optional[str] = None
    related_memory_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    time_context: dict[str, Any] = field(default_factory=dict)
    interaction_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class VexisState:
    lifecycle: LifecycleState = field(default_factory=LifecycleState)
    presence: PresenceState = field(default_factory=PresenceState)
    speech: SpeechState = field(default_factory=SpeechState)
    cognition: CognitionState = field(default_factory=CognitionState)
    ingest: IngestState = field(default_factory=IngestState)
    memory: MemoryState = field(default_factory=MemoryState)
    tasks: TaskState = field(default_factory=TaskState)
    epistemic: EpistemicState = field(default_factory=EpistemicState)
    recent_inputs: list[InputRecord] = field(default_factory=list)
    recent_memories: list[MemoryRecord] = field(default_factory=list)
    recent_tasks: list[TaskRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VexisState":
        lifecycle = LifecycleState(**data.get("lifecycle", {}))
        presence = PresenceState(**data.get("presence", {}))
        speech = SpeechState(**data.get("speech", {}))
        cognition = CognitionState(**data.get("cognition", {}))
        ingest = IngestState(**data.get("ingest", {}))
        memory = MemoryState(**data.get("memory", {}))
        tasks = TaskState(**data.get("tasks", {}))
        epistemic = EpistemicState(**data.get("epistemic", {}))

        recent_inputs = [InputRecord(**item) for item in data.get("recent_inputs", [])]
        recent_memories = [MemoryRecord(**item) for item in data.get("recent_memories", [])]
        recent_tasks = [TaskRecord(**item) for item in data.get("recent_tasks", [])]

        return cls(
            lifecycle=lifecycle,
            presence=presence,
            speech=speech,
            cognition=cognition,
            ingest=ingest,
            memory=memory,
            tasks=tasks,
            epistemic=epistemic,
            recent_inputs=recent_inputs,
            recent_memories=recent_memories,
            recent_tasks=recent_tasks,
        )