from __future__ import annotations

import json
from pathlib import Path
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
        if not self.state_path.exists():
            state = VexisState()
            now = utc_now_iso()
            state.lifecycle.started_at = now
            state.lifecycle.last_active_at = now
            self.save_state(state)
            return state

        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return VexisState.from_dict(data)
        except Exception:
            state = VexisState()
            now = utc_now_iso()
            state.lifecycle.started_at = now
            state.lifecycle.last_active_at = now
            self.save_state(state)
            return state

    def save_state(self, state: Optional[VexisState] = None) -> None:
        if state is not None:
            self.state = state

        self.state.lifecycle.last_active_at = utc_now_iso()
        self.state_path.write_text(
            json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def increment_boot_count(self) -> int:
        self.state.lifecycle.boot_count += 1
        self.state.lifecycle.last_active_at = utc_now_iso()
        self.save_state()
        return self.state.lifecycle.boot_count

    def set_visual_state(self, visual_state: str, status_text: Optional[str] = None) -> None:
        self.state.presence.visual_state = visual_state
        if status_text is not None:
            self.state.presence.status_text = status_text
        self.save_state()

    def set_thinking(
        self,
        is_thinking: bool,
        current_focus: Optional[str] = None,
        current_input_id: Optional[str] = None,
        last_classification: Optional[str] = None,
    ) -> None:
        self.state.cognition.is_thinking = is_thinking
        if current_focus is not None:
            self.state.cognition.current_focus = current_focus
        if current_input_id is not None:
            self.state.cognition.current_input_id = current_input_id
        if last_classification is not None:
            self.state.cognition.last_classification = last_classification
        self.save_state()

    def set_speaking(self, is_speaking: bool, last_spoken_text: Optional[str] = None) -> None:
        self.state.speech.is_speaking = is_speaking
        if last_spoken_text is not None:
            self.state.speech.last_spoken_text = last_spoken_text
        self.save_state()

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
        final_interaction_context = self._build_interaction_context(
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

        self.state.recent_inputs.append(record)
        self.state.recent_inputs = self.state.recent_inputs[-50:]
        self.state.cognition.current_input_id = record.input_id
        self.state.cognition.last_classification = input_type
        self.save_state()
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
        final_interaction_context = self._build_interaction_context(
            source=source,
            interaction_type=kind,
            raw_text=content,
            interaction_context=interaction_context,
            store_as_evidence=kind in {"question", "claim", "command", "note", "file"},
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

        self.state.recent_memories.append(record)
        self.state.recent_memories = self.state.recent_memories[-100:]
        self.state.memory.total_items += 1
        self.state.memory.last_memory_id = record.memory_id
        self.save_state()
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
        final_interaction_context = self._build_interaction_context(
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

        self.state.recent_tasks.append(record)
        self.state.recent_tasks = self.state.recent_tasks[-100:]

        if status == "active":
            self.state.tasks.active_task_ids.append(record.task_id)
        elif status == "unresolved":
            self.state.tasks.unresolved_task_ids.append(record.task_id)
        elif status == "completed":
            self.state.tasks.completed_task_ids.append(record.task_id)

        self.save_state()
        return record

    def add_open_question(self, question: str) -> None:
        self.state.epistemic.open_questions.append(question)
        self.state.epistemic.open_questions = self.state.epistemic.open_questions[-100:]
        self.save_state()

    def add_open_claim(self, claim: str) -> None:
        self.state.epistemic.open_claims.append(claim)
        self.state.epistemic.open_claims = self.state.epistemic.open_claims[-100:]
        self.save_state()

    def add_unsupported_claim(self, claim: str) -> None:
        self.state.epistemic.unsupported_claims.append(claim)
        self.state.epistemic.unsupported_claims = self.state.epistemic.unsupported_claims[-100:]
        self.save_state()

    def add_contradiction(self, contradiction: str) -> None:
        self.state.epistemic.contradictions.append(contradiction)
        self.state.epistemic.contradictions = self.state.epistemic.contradictions[-100:]
        self.save_state()

    def queue_file_for_ingest(self, filepath: str) -> None:
        if filepath not in self.state.ingest.pending_files:
            self.state.ingest.pending_files.append(filepath)
        self.save_state()

    def mark_file_processing(self, filepath: Optional[str]) -> None:
        self.state.ingest.processing_file = filepath
        self.save_state()

    def mark_file_completed(self, filepath: str) -> None:
        self.state.ingest.processing_file = None
        if filepath in self.state.ingest.pending_files:
            self.state.ingest.pending_files.remove(filepath)
        if filepath not in self.state.ingest.completed_files:
            self.state.ingest.completed_files.append(filepath)
        self.save_state()

    def get_state(self) -> VexisState:
        return self.state

    def _build_interaction_context(
        self,
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
            "presence_state_at_input": self.state.presence.visual_state,
            "cognitive_state_at_input": (
                "thinking" if self.state.cognition.is_thinking else self.state.presence.status_text
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