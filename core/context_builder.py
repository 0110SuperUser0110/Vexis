from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.schemas import InputRecord, MemoryRecord, TaskRecord, VexisState
from memory.memory_store import MemoryStore


@dataclass
class ContextBundle:
    current_input: InputRecord
    related_memories: list[MemoryRecord] = field(default_factory=list)
    recent_inputs: list[InputRecord] = field(default_factory=list)
    recent_tasks: list[TaskRecord] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    open_claims: list[str] = field(default_factory=list)


class ContextBuilder:
    """
    Builds lightweight working context for the response engine.

    First-pass strategy:
    - related memory search by keyword overlap
    - recent input/task recall
    - epistemic queue snapshot
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def build(self, state: VexisState, current_input: InputRecord) -> ContextBundle:
        related_memories = self._related_memories(
            query=current_input.raw_text,
            exclude_input_id=current_input.input_id,
            limit=5,
        )

        recent_inputs = self._recent_inputs(exclude_input_id=current_input.input_id, limit=5)
        recent_tasks = self.memory_store.get_recent_tasks(limit=5)

        return ContextBundle(
            current_input=current_input,
            related_memories=related_memories,
            recent_inputs=recent_inputs,
            recent_tasks=recent_tasks,
            open_questions=state.epistemic.open_questions[-10:],
            open_claims=state.epistemic.open_claims[-10:],
        )

    def _related_memories(
        self,
        query: str,
        exclude_input_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        matches = self.memory_store.search_memories(query, limit=limit + 3)
        filtered: list[MemoryRecord] = []

        for match in matches:
            if exclude_input_id and match.related_input_id == exclude_input_id:
                continue
            filtered.append(match)
            if len(filtered) >= limit:
                break

        return filtered

    def _recent_inputs(
        self,
        exclude_input_id: Optional[str] = None,
        limit: int = 5,
    ) -> list[InputRecord]:
        items = self.memory_store.get_recent_inputs(limit=limit + 3)
        filtered: list[InputRecord] = []

        for item in reversed(items):
            if exclude_input_id and item.input_id == exclude_input_id:
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break

        return filtered