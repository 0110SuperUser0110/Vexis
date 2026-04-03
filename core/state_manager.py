from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Optional
from uuid import uuid4

from core.schemas import (
    InputRecord,
    MemoryRecord,
    TaskRecord,
    VexisState,
    utc_now_iso,
)
from core.time_context import TimeContextBuilder


class StateManager:
    _lock = RLock()

    def __init__(
        self,
        state_path: str = "data/state/vexis_state.json",
        default_timezone: str = "America/Chicago",
        device_label: str = "BIGFATBETTY",
        location_label: str = "local_desktop",
        session_id: str = "local_session",
        speaker_id: str = "primary_user",
    ) -> None:
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.time_builder = TimeContextBuilder(default_timezone=default_timezone)

        self.device_label = device_label
        self.location_label = location_label
        self.session_id = session_id
        self.speaker_id = speaker_id

        self.state = self.load_state()

    def load_state(self) -> VexisState:
        with self._lock:
            state = self._read_state_from_disk()
            self.state = state
            return state

    def _read_state_from_disk(self) -> VexisState:
        if not self.state_path.exists():
            state = VexisState()
            now = utc_now_iso()
            state.lifecycle.started_at = now
            state.lifecycle.last_active_at = now
            self.state = state
            self._write_state_to_disk(state)
            return state

        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return VexisState.from_dict(data)
        except Exception:
            state = VexisState()
            now = utc_now_iso()
            state.lifecycle.started_at = now
            state.lifecycle.last_active_at = now
            self.state = state
            self._write_state_to_disk(state)
            return state

    def save_state(self, state: Optional[VexisState] = None) -> None:
        with self._lock:
            if state is not None:
                self.state = state

            self.state.lifecycle.last_active_at = utc_now_iso()
            self._write_state_to_disk(self.state)

    def _write_state_to_disk(self, state: VexisState) -> None:
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _mutate_state(self, mutator) -> None:
        with self._lock:
            self.state = self._read_state_from_disk()
            mutator(self.state)
            self.state.lifecycle.last_active_at = utc_now_iso()
            self._write_state_to_disk(self.state)

    def increment_boot_count(self) -> int:
        next_count = 0

        def mutate(state: VexisState) -> None:
            nonlocal next_count
            state.lifecycle.boot_count += 1
            next_count = state.lifecycle.boot_count

        self._mutate_state(mutate)
        return next_count

    def set_visual_state(self, visual_state: str, status_text: Optional[str] = None) -> None:
        def mutate(state: VexisState) -> None:
            state.presence.visual_state = visual_state
            if status_text is not None:
                state.presence.status_text = status_text

        self._mutate_state(mutate)

    def set_thinking(
        self,
        is_thinking: bool,
        current_focus: Optional[str] = None,
        current_input_id: Optional[str] = None,
        last_classification: Optional[str] = None,
    ) -> None:
        def mutate(state: VexisState) -> None:
            state.cognition.is_thinking = is_thinking
            if current_focus is not None:
                state.cognition.current_focus = current_focus
            if current_input_id is not None:
                state.cognition.current_input_id = current_input_id
            if last_classification is not None:
                state.cognition.last_classification = last_classification

        self._mutate_state(mutate)

    def set_speaking(self, is_speaking: bool, last_spoken_text: Optional[str] = None) -> None:
        def mutate(state: VexisState) -> None:
            state.speech.is_speaking = is_speaking
            if last_spoken_text is not None:
                state.speech.last_spoken_text = last_spoken_text

        self._mutate_state(mutate)

    def add_input(
        self,
        raw_text: str,
        source: str = "gui",
        input_type: str = "unknown",
        confidence: float = 0.0,
        metadata: Optional[dict] = None,
        timezone_name: Optional[str] = None,
        interaction_context: Optional[dict] = None,
    ) -> InputRecord:
        time_context = self.time_builder.now(timezone_name=timezone_name).to_dict()
        record: Optional[InputRecord] = None

        def mutate(state: VexisState) -> None:
            nonlocal record
            final_interaction_context = self._build_interaction_context(
                state=state,
                source=source,
                interaction_type=input_type,
                raw_text=raw_text,
                interaction_context=interaction_context,
                store_as_evidence=input_type in {"question", "claim", "command", "note"},
            )

            record = InputRecord(
                input_id=f"input_{uuid4().hex[:12]}",
                timestamp=time_context["utc_timestamp"],
                source=source,
                raw_text=raw_text,
                input_type=input_type,
                confidence=confidence,
                metadata=metadata or {},
                time_context=time_context,
                interaction_context=final_interaction_context,
            )

            state.recent_inputs.append(record)
            state.recent_inputs = state.recent_inputs[-50:]
            state.cognition.current_input_id = record.input_id
            state.cognition.last_classification = input_type

        self._mutate_state(mutate)
        if record is None:
            raise RuntimeError("failed to create input record")
        return record

    def add_memory(
        self,
        kind: str,
        content: str,
        source: str = "gui",
        related_input_id: Optional[str] = None,
        status: str = "active",
        metadata: Optional[dict] = None,
        timezone_name: Optional[str] = None,
        interaction_context: Optional[dict] = None,
    ) -> MemoryRecord:
        time_context = self.time_builder.now(timezone_name=timezone_name).to_dict()
        record: Optional[MemoryRecord] = None

        def mutate(state: VexisState) -> None:
            nonlocal record
            final_interaction_context = self._build_interaction_context(
                state=state,
                source=source,
                interaction_type=kind,
                raw_text=content,
                interaction_context=interaction_context,
                store_as_evidence=kind in {"question", "claim", "command", "note", "file", "fact", "resolved_question", "belief_candidate"},
            )

            record = MemoryRecord(
                memory_id=f"mem_{uuid4().hex[:12]}",
                timestamp=time_context["utc_timestamp"],
                kind=kind,
                content=content,
                source=source,
                related_input_id=related_input_id,
                status=status,
                metadata=metadata or {},
                time_context=time_context,
                interaction_context=final_interaction_context,
            )

            state.recent_memories.append(record)
            state.recent_memories = state.recent_memories[-100:]
            state.memory.total_items += 1
            state.memory.last_memory_id = record.memory_id

        self._mutate_state(mutate)
        if record is None:
            raise RuntimeError("failed to create memory record")
        return record

    def add_task(
        self,
        title: str,
        description: str,
        status: str = "active",
        priority: int = 0,
        related_input_id: Optional[str] = None,
        related_memory_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        timezone_name: Optional[str] = None,
        interaction_context: Optional[dict] = None,
    ) -> TaskRecord:
        time_context = self.time_builder.now(timezone_name=timezone_name).to_dict()
        record: Optional[TaskRecord] = None

        def mutate(state: VexisState) -> None:
            nonlocal record
            final_interaction_context = self._build_interaction_context(
                state=state,
                source="task_system",
                interaction_type="task",
                raw_text=description,
                interaction_context=interaction_context,
                store_as_evidence=True,
            )

            record = TaskRecord(
                task_id=f"task_{uuid4().hex[:12]}",
                timestamp=time_context["utc_timestamp"],
                title=title,
                description=description,
                status=status,
                priority=priority,
                related_input_id=related_input_id,
                related_memory_id=related_memory_id,
                metadata=metadata or {},
                time_context=time_context,
                interaction_context=final_interaction_context,
            )

            state.recent_tasks.append(record)
            state.recent_tasks = state.recent_tasks[-100:]

            if status == "active":
                state.tasks.active_task_ids.append(record.task_id)
            elif status == "unresolved":
                state.tasks.unresolved_task_ids.append(record.task_id)
            elif status == "completed":
                state.tasks.completed_task_ids.append(record.task_id)

        self._mutate_state(mutate)
        if record is None:
            raise RuntimeError("failed to create task record")
        return record

    def add_open_question(self, question: str) -> None:
        normalized_question = str(question).strip()
        if not normalized_question:
            return

        def mutate(state: VexisState) -> None:
            if normalized_question not in state.epistemic.open_questions:
                state.epistemic.open_questions.append(normalized_question)
                state.epistemic.open_questions = state.epistemic.open_questions[-100:]

        self._mutate_state(mutate)

    def remove_open_question(self, question: str) -> None:
        normalized_question = str(question).strip().lower()
        if not normalized_question:
            return

        def mutate(state: VexisState) -> None:
            state.epistemic.open_questions = [
                item
                for item in state.epistemic.open_questions
                if str(item).strip().lower() != normalized_question
            ]

        self._mutate_state(mutate)

    def add_open_claim(self, claim: str) -> None:
        def mutate(state: VexisState) -> None:
            state.epistemic.open_claims.append(claim)
            state.epistemic.open_claims = state.epistemic.open_claims[-100:]

        self._mutate_state(mutate)

    def add_unsupported_claim(self, claim: str) -> None:
        def mutate(state: VexisState) -> None:
            state.epistemic.unsupported_claims.append(claim)
            state.epistemic.unsupported_claims = state.epistemic.unsupported_claims[-100:]

        self._mutate_state(mutate)

    def add_contradiction(self, contradiction: str) -> None:
        def mutate(state: VexisState) -> None:
            state.epistemic.contradictions.append(contradiction)
            state.epistemic.contradictions = state.epistemic.contradictions[-100:]

        self._mutate_state(mutate)

    def queue_file_for_ingest(self, filepath: str) -> None:
        def mutate(state: VexisState) -> None:
            if filepath not in state.ingest.pending_files:
                state.ingest.pending_files.append(filepath)

        self._mutate_state(mutate)

    def mark_file_processing(self, filepath: Optional[str]) -> None:
        def mutate(state: VexisState) -> None:
            state.ingest.processing_file = filepath

        self._mutate_state(mutate)

    def mark_file_completed(self, filepath: str) -> None:
        def mutate(state: VexisState) -> None:
            state.ingest.processing_file = None
            if filepath in state.ingest.pending_files:
                state.ingest.pending_files.remove(filepath)
            if filepath not in state.ingest.completed_files:
                state.ingest.completed_files.append(filepath)

        self._mutate_state(mutate)

    def get_state(self) -> VexisState:
        return self.load_state()

    def _build_interaction_context(
        self,
        state: VexisState,
        source: str,
        interaction_type: str,
        raw_text: str,
        interaction_context: Optional[dict] = None,
        store_as_evidence: bool = True,
    ) -> dict:
        base = {
            "speaker_role": "user",
            "speaker_id": self.speaker_id,
            "source": source,
            "session_id": self.session_id,
            "interaction_type": interaction_type,
            "social_subtype": None,
            "device_label": self.device_label,
            "location_label": self.location_label,
            "presence_state_at_input": state.presence.visual_state,
            "cognitive_state_at_input": (
                "thinking" if state.cognition.is_thinking else state.presence.status_text
            ),
            "importance": "normal",
            "store_as_evidence": store_as_evidence,
            "expects_reply": interaction_type in {"social", "question", "claim", "command"},
            "expects_reasoning": interaction_type in {"question", "claim", "command"},
            "personality_allowed": interaction_type == "social",
            "raw_length": len(raw_text),
        }

        if interaction_context:
            base.update(interaction_context)

        return base
