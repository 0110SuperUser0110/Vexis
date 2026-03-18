from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.schemas import InputRecord, MemoryRecord, TaskRecord


class MemoryStore:
    def __init__(self, base_dir: str = "data/memory") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.inputs_path = self.base_dir / "inputs.json"
        self.memories_path = self.base_dir / "memories.json"
        self.tasks_path = self.base_dir / "tasks.json"

        self._ensure_file(self.inputs_path)
        self._ensure_file(self.memories_path)
        self._ensure_file(self.tasks_path)

    def _ensure_file(self, path: Path) -> None:
        if not path.exists():
            path.write_text("[]", encoding="utf-8")

    def _read_json_list(self, path: Path) -> list[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def _write_json_list(self, path: Path, data: list[dict[str, Any]]) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def save_input(self, record: InputRecord) -> None:
        items = self._read_json_list(self.inputs_path)
        items.append(record.__dict__)
        self._write_json_list(self.inputs_path, items)

    def save_memory(self, record: MemoryRecord) -> None:
        items = self._read_json_list(self.memories_path)
        items.append(record.__dict__)
        self._write_json_list(self.memories_path, items)

    def save_task(self, record: TaskRecord) -> None:
        items = self._read_json_list(self.tasks_path)
        items.append(record.__dict__)
        self._write_json_list(self.tasks_path, items)

    def load_inputs(self) -> list[InputRecord]:
        return [InputRecord(**item) for item in self._read_json_list(self.inputs_path)]

    def load_memories(self) -> list[MemoryRecord]:
        return [MemoryRecord(**item) for item in self._read_json_list(self.memories_path)]

    def load_tasks(self) -> list[TaskRecord]:
        return [TaskRecord(**item) for item in self._read_json_list(self.tasks_path)]

    def find_input_by_id(self, input_id: str) -> Optional[InputRecord]:
        for item in self.load_inputs():
            if item.input_id == input_id:
                return item
        return None

    def find_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        for item in self.load_memories():
            if item.memory_id == memory_id:
                return item
        return None

    def find_task_by_id(self, task_id: str) -> Optional[TaskRecord]:
        for item in self.load_tasks():
            if item.task_id == task_id:
                return item
        return None

    def get_recent_inputs(self, limit: int = 10) -> list[InputRecord]:
        return self.load_inputs()[-limit:]

    def get_recent_memories(self, limit: int = 10) -> list[MemoryRecord]:
        return self.load_memories()[-limit:]

    def get_recent_tasks(self, limit: int = 10) -> list[TaskRecord]:
        return self.load_tasks()[-limit:]

    def search_memories(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        """
        Simple keyword-overlap search.
        Good enough for first-pass recall before LLM or embeddings.
        """
        query_terms = self._normalize_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.load_memories():
            haystack_terms = self._normalize_terms(
                " ".join(
                    [
                        record.kind,
                        record.content,
                        record.source,
                        " ".join(f"{k}:{v}" for k, v in record.metadata.items()),
                    ]
                )
            )
            overlap = len(query_terms.intersection(haystack_terms))
            if overlap > 0:
                scored.append((overlap, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def get_latest_memory_by_kind(self, kind: str) -> Optional[MemoryRecord]:
        matches = [m for m in self.load_memories() if m.kind == kind]
        if not matches:
            return None
        return matches[-1]

    def _normalize_terms(self, text: str) -> set[str]:
        cleaned = []
        for ch in text.lower():
            cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
        return {part for part in "".join(cleaned).split() if len(part) > 1}