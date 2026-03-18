from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.autonomy_engine import AutonomyAction, AutonomyEngine
from core.belief_engine import BeliefEngine, BeliefRecord
from core.evidence_engine import EvidenceAssessment, EvidenceEngine
from core.methodology_engine import MethodologyAssessment, MethodologyEngine
from core.schemas import MemoryRecord, VexisState
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


@dataclass
class CognitionCycleResult:
    cycle_timestamp: float
    open_question_count: int
    unsupported_claim_count: int
    active_task_count: int
    actions: list[AutonomyAction] = field(default_factory=list)
    belief_updates: list[BeliefRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_timestamp": self.cycle_timestamp,
            "open_question_count": self.open_question_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "active_task_count": self.active_task_count,
            "actions": [action.to_dict() for action in self.actions],
            "belief_updates": [belief.to_dict() for belief in self.belief_updates],
            "notes": self.notes,
            "metadata": self.metadata,
        }


class CognitionLoop:
    """
    Persistent deterministic cognition layer for VEX.

    This is not an output loop.
    This is a background review engine that keeps running whether or not the user is speaking.

    Responsibilities:
    - review open questions
    - review unsupported claims
    - review recent memories
    - create autonomous follow-up actions
    - promote evidence into belief candidates
    """

    def __init__(
        self,
        state_manager: StateManager,
        memory_store: MemoryStore,
        autonomy_engine: Optional[AutonomyEngine] = None,
        methodology_engine: Optional[MethodologyEngine] = None,
        evidence_engine: Optional[EvidenceEngine] = None,
        belief_engine: Optional[BeliefEngine] = None,
    ) -> None:
        self.state_manager = state_manager
        self.memory_store = memory_store
        self.autonomy_engine = autonomy_engine or AutonomyEngine()
        self.methodology_engine = methodology_engine or MethodologyEngine()
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self.belief_engine = belief_engine or BeliefEngine()

        self._cycle_index = 0
        self._last_cycle_at = 0.0

    def run_cycle(self) -> CognitionCycleResult:
        state = self.state_manager.get_state()
        now = time.time()
        self._cycle_index += 1
        self._last_cycle_at = now

        open_questions = list(state.epistemic.open_questions)
        unsupported_claims = list(state.epistemic.unsupported_claims)
        active_task_count = len(state.tasks.active_task_ids)

        notes: list[str] = []
        actions: list[AutonomyAction] = []
        belief_updates: list[BeliefRecord] = []

        # 1. Ask the autonomy engine what to do next.
        actions.extend(
            self.autonomy_engine.generate_actions(
                state=state,
                open_questions=open_questions,
                unsupported_claims=unsupported_claims,
                recent_memories=state.recent_memories,
                last_cycle_at=self._last_cycle_at,
            )
        )

        # 2. Review unresolved questions against memory.
        for question in open_questions[:10]:
            matches = self.memory_store.search_memories(question, limit=5)
            if matches:
                notes.append(
                    f'background review found {len(matches)} possible memory matches for open question "{question}"'
                )

        # 3. Review unsupported claims against memory.
        for claim in unsupported_claims[:10]:
            matches = self.memory_store.search_memories(claim, limit=5)
            if matches:
                notes.append(
                    f'background review found {len(matches)} possible memory matches for unsupported claim "{claim}"'
                )

        # 4. Try to form low-risk belief candidates from repeated note/file memories.
        belief_candidates = self._build_belief_candidates(state.recent_memories)
        belief_updates.extend(belief_candidates)

        return CognitionCycleResult(
            cycle_timestamp=now,
            open_question_count=len(open_questions),
            unsupported_claim_count=len(unsupported_claims),
            active_task_count=active_task_count,
            actions=actions,
            belief_updates=belief_updates,
            notes=notes,
            metadata={
                "cycle_index": self._cycle_index,
                "recent_memory_count": len(state.recent_memories),
            },
        )

    def _build_belief_candidates(self, recent_memories: list[MemoryRecord]) -> list[BeliefRecord]:
        """
        This is intentionally conservative.

        It only creates candidate beliefs from repeated, near-identical note/file memories.
        It does not try to do full semantic science extraction yet.
        """
        candidates: list[BeliefRecord] = []
        grouped: dict[str, list[MemoryRecord]] = {}

        for memory in recent_memories:
            if memory.kind not in {"note", "file"}:
                continue

            statement = self._normalize_statement(memory.content)
            if len(statement.split()) < 4:
                continue

            grouped.setdefault(statement, []).append(memory)

        for statement, items in grouped.items():
            if len(items) < 2:
                continue

            evidence_items: list[EvidenceAssessment] = []

            for item in items[:6]:
                source_type = str(item.metadata.get("source_type", "unknown"))
                methodology = self._methodology_for_memory(item)

                evidence = self.evidence_engine.assess_evidence(
                    statement=statement,
                    source_id=item.memory_id,
                    source_type=source_type,
                    methodology=methodology,
                    supports_statement=True,
                    independence_group=item.metadata.get("independence_group"),
                    contradiction_count=0,
                    metadata={
                        "memory_kind": item.kind,
                        "related_input_id": item.related_input_id,
                    },
                )
                evidence_items.append(evidence)

            belief = self.belief_engine.build_belief(
                belief_id=f"belief_{abs(hash(statement)) % 10_000_000}",
                statement=statement,
                evidence_items=evidence_items,
                metadata={
                    "origin": "background_cognition_loop",
                    "candidate_type": "repeated_memory_convergence",
                },
            )

            if belief.confidence_score >= 0.18:
                candidates.append(belief)

        return candidates

    def _methodology_for_memory(self, memory: MemoryRecord) -> Optional[MethodologyAssessment]:
        source_type = str(memory.metadata.get("source_type", "unknown"))
        if source_type in {"peer_reviewed_journal", "preprint", "media", "official_record", "technical_documentation"}:
            return self.methodology_engine.assess(
                text=memory.content,
                metadata={
                    "source_type": source_type,
                    "title": memory.metadata.get("title"),
                    "year": memory.metadata.get("year"),
                },
            )
        return None

    def _normalize_statement(self, text: str) -> str:
        statement = (text or "").strip().lower()
        statement = " ".join(statement.split())
        if len(statement) > 280:
            statement = statement[:280].rstrip()
        return statement