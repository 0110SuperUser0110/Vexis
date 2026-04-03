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
    - source expansion for uploaded knowledge so broad prompts can see beyond title pages
    - recent input/task recall
    - epistemic queue snapshot
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        related_memory_limit: int = 8,
        recent_input_limit: int = 5,
        recent_task_limit: int = 5,
    ) -> None:
        self.memory_store = memory_store
        self.related_memory_limit = related_memory_limit
        self.recent_input_limit = recent_input_limit
        self.recent_task_limit = recent_task_limit

    def build(self, state: VexisState, current_input: InputRecord) -> ContextBundle:
        related_memories = self._related_memories(
            query=current_input.raw_text,
            exclude_input_id=current_input.input_id,
            limit=self.related_memory_limit,
        )

        recent_inputs = self._recent_inputs(
            exclude_input_id=current_input.input_id,
            limit=self.recent_input_limit,
        )
        recent_tasks = self.memory_store.get_recent_tasks(limit=self.recent_task_limit)

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

        if filtered:
            filtered = self._expand_knowledge_context(query=query, memories=filtered, limit=limit)

        return filtered[:limit]

    def _expand_knowledge_context(
        self,
        query: str,
        memories: list[MemoryRecord],
        limit: int,
    ) -> list[MemoryRecord]:
        knowledge_matches = [
            memory
            for memory in memories
            if memory.kind in {"knowledge_chunk", "knowledge_source"}
        ]
        if not knowledge_matches or limit <= 3:
            return memories

        source_key = self._dominant_source_key(knowledge_matches)
        if not source_key:
            return memories

        same_source_chunks = [
            memory
            for memory in self.memory_store.load_memories()
            if memory.kind == "knowledge_chunk" and self._memory_source_key(memory) == source_key
        ]
        if not same_source_chunks:
            return memories

        same_source_chunks.sort(key=lambda memory: int(memory.metadata.get("chunk_index", 0)))
        query_terms = self._terms(query)
        broad_request = len(query_terms) <= 2
        candidate_chunks = [
            memory
            for memory in same_source_chunks
            if int(memory.metadata.get("chunk_index", 0)) >= 8
        ] or same_source_chunks

        ranked_chunks = self._rank_source_chunks(
            candidate_chunks,
            query_terms=query_terms,
            broad_request=broad_request,
        )

        reserve = min(3, max(1, (limit + 1) // 3))
        base = memories[: max(0, limit - reserve)]
        existing_ids = {memory.memory_id for memory in base}

        for chunk in ranked_chunks:
            if chunk.memory_id in existing_ids:
                continue
            base.append(chunk)
            existing_ids.add(chunk.memory_id)
            if len(base) >= limit:
                break

        return base

    def _rank_source_chunks(
        self,
        chunks: list[MemoryRecord],
        query_terms: set[str],
        broad_request: bool,
    ) -> list[MemoryRecord]:
        if broad_request:
            representatives = self._representative_chunks(chunks, count=3)
            seen_ids = {chunk.memory_id for chunk in representatives}
            remainder = [chunk for chunk in chunks if chunk.memory_id not in seen_ids]
            return representatives + remainder

        scored: list[tuple[float, MemoryRecord]] = []
        for chunk in chunks:
            chunk_terms = self._terms(
                " ".join(
                    [
                        chunk.content,
                        str(chunk.metadata.get("section_name", "")),
                    ]
                )
            )
            overlap = len(query_terms.intersection(chunk_terms))
            score = float(overlap)
            if int(chunk.metadata.get("chunk_index", 0)) >= 8:
                score += 0.25
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        if any(score > 0 for score, _ in scored):
            return [chunk for _, chunk in scored]

        representatives = self._representative_chunks(chunks, count=3)
        seen_ids = {chunk.memory_id for chunk in representatives}
        remainder = [chunk for chunk in chunks if chunk.memory_id not in seen_ids]
        return representatives + remainder

    def _representative_chunks(
        self,
        chunks: list[MemoryRecord],
        count: int = 3,
    ) -> list[MemoryRecord]:
        if len(chunks) <= count:
            return chunks

        positions = {0}
        if count > 1:
            positions.add(len(chunks) // 3)
            positions.add((2 * len(chunks)) // 3)
        positions = {min(max(position, 0), len(chunks) - 1) for position in positions}
        return [chunks[position] for position in sorted(positions)]

    def _dominant_source_key(self, memories: list[MemoryRecord]) -> str:
        counts: dict[str, int] = {}
        for memory in memories:
            source_key = self._memory_source_key(memory)
            if not source_key:
                continue
            counts[source_key] = counts.get(source_key, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda item: item[1])[0]

    def _memory_source_key(self, memory: MemoryRecord) -> str:
        metadata = memory.metadata or {}
        return str(metadata.get("source_path") or metadata.get("title") or "")

    def _terms(self, text: str) -> set[str]:
        cleaned = []
        for ch in text.lower():
            cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
        return {part for part in "".join(cleaned).split() if len(part) > 2}

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
