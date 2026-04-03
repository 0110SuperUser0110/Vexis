from __future__ import annotations

from dataclasses import dataclass, field
import re

from core.resolution_engine import ResolutionEngine
from core.schemas import MemoryRecord
from core.self_model import SelfModel
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


_INVALID_SELF_FACT_PATTERNS = (
    re.compile(r"^vex (running_on|location_label) [a-z0-9_+\-]+$"),
    re.compile(r"^which property your$"),
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _is_invalid_self_fact(record: MemoryRecord) -> bool:
    if record.kind != "fact":
        return False
    content = _normalize_text(record.content)
    return any(pattern.match(content) for pattern in _INVALID_SELF_FACT_PATTERNS)


@dataclass
class RuntimeRepairReport:
    removed_invalid_memory_ids: list[str] = field(default_factory=list)
    removed_open_questions: list[str] = field(default_factory=list)
    added_resolved_questions: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.removed_invalid_memory_ids
            or self.removed_open_questions
            or self.added_resolved_questions
        )


def repair_runtime_state(
    state_manager: StateManager,
    memory_store: MemoryStore,
) -> RuntimeRepairReport:
    report = RuntimeRepairReport()
    memories = memory_store.load_memories()
    invalid_memory_ids = {record.memory_id for record in memories if _is_invalid_self_fact(record)}

    if invalid_memory_ids:
        filtered_memories = [record for record in memories if record.memory_id not in invalid_memory_ids]
        memory_store.replace_memories(filtered_memories)
        memories = filtered_memories
        report.removed_invalid_memory_ids.extend(sorted(invalid_memory_ids))

    resolved_questions = {
        _normalize_text(record.content)
        for record in memories
        if record.kind == "resolved_question" and record.content
    }
    resolution_engine = ResolutionEngine(self_model=SelfModel(state_manager))
    state = state_manager.get_state()

    for question in list(state.epistemic.open_questions):
        normalized_question = _normalize_text(question)
        if not normalized_question:
            state_manager.remove_open_question(question)
            report.removed_open_questions.append(question)
            continue

        if normalized_question in resolved_questions:
            state_manager.remove_open_question(question)
            report.removed_open_questions.append(question)
            continue

        resolution = resolution_engine.resolve_question(question, state, memories)
        if not resolution.resolved:
            continue

        state_manager.remove_open_question(question)
        report.removed_open_questions.append(question)

        if resolution.should_store_resolution and normalized_question not in resolved_questions:
            record = state_manager.add_memory(
                kind="resolved_question",
                content=question,
                source="resolution_engine",
                status="resolved",
                metadata={"source_type": "startup_repair"},
            )
            memory_store.save_memory(record)
            memories.append(record)
            resolved_questions.add(normalized_question)
            report.added_resolved_questions.append(question)

    if report.changed:
        state = state_manager.load_state()
        if invalid_memory_ids:
            state.recent_memories = [
                record for record in state.recent_memories if record.memory_id not in invalid_memory_ids
            ]
        state.recent_memories = state.recent_memories[-100:]
        state.memory.total_items = len(memory_store.load_memories())
        if state.recent_memories:
            state.memory.last_memory_id = state.recent_memories[-1].memory_id
        state_manager.save_state(state)

    return report
