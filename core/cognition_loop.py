from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.autonomy_engine import AutonomyAction, AutonomyEngine
from core.belief_engine import BeliefEngine, BeliefRecord
from core.evidence_engine import EvidenceAssessment, EvidenceEngine
from core.fact_learning_engine import FactLearningEngine
from core.inquiry_engine import InquiryEngine
from core.methodology_engine import MethodologyAssessment, MethodologyEngine
from core.resolution_engine import ResolutionEngine
from core.schemas import MemoryRecord, VexisState
from core.self_model import SelfModel
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
    resolved_questions: list[str] = field(default_factory=list)
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
            "resolved_questions": self.resolved_questions,
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
        fact_learning_engine: Optional[FactLearningEngine] = None,
        inquiry_engine: Optional[InquiryEngine] = None,
        resolution_engine: Optional[ResolutionEngine] = None,
    ) -> None:
        self.state_manager = state_manager
        self.memory_store = memory_store
        self.autonomy_engine = autonomy_engine or AutonomyEngine()
        self.methodology_engine = methodology_engine or MethodologyEngine()
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self.belief_engine = belief_engine or BeliefEngine()
        self.fact_learning_engine = fact_learning_engine or FactLearningEngine(
            evidence_engine=self.evidence_engine,
            belief_engine=self.belief_engine,
            methodology_engine=self.methodology_engine,
        )
        self.inquiry_engine = inquiry_engine or InquiryEngine()
        self.resolution_engine = resolution_engine or ResolutionEngine(
            self_model=SelfModel(self.state_manager),
        )

        self._cycle_index = 0
        self._last_cycle_at = 0.0
        self._last_review_signals: dict[str, int] = {}

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
        resolved_questions: list[str] = []
        all_memories = self.memory_store.load_memories()
        recent_memories = all_memories[-250:]

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

        # 2. Review unresolved questions against memory and attempt low-risk background resolution.
        for question in open_questions[:10]:
            signal_key = f"open_question::{question}"

            if self._has_resolved_question_memory(question, all_memories):
                resolved_questions.append(question)
                self._last_review_signals.pop(signal_key, None)
                continue

            resolution = self.resolution_engine.resolve_question(
                question_text=question,
                state=state,
                recent_memories=recent_memories,
            )
            if resolution.resolved:
                resolved_questions.append(question)
                self._last_review_signals.pop(signal_key, None)
                if resolution.should_store_resolution and not self._has_resolved_question_memory(question, all_memories):
                    resolved_memory = self.state_manager.add_memory(
                        kind="resolved_question",
                        content=question,
                        source="cognition_loop",
                        status="resolved",
                        metadata={
                            "source_type": "background_resolution",
                            "answer_text": resolution.answer_text,
                            "question_type": resolution.question_type,
                            "reasons": list(resolution.reasons),
                        },
                        interaction_context={
                            "speaker_role": "system",
                            "speaker_id": "vex_cognition_loop",
                            "source": "cognition_loop",
                            "interaction_type": "resolved_question",
                            "importance": "high",
                            "store_as_evidence": True,
                            "expects_reply": False,
                            "expects_reasoning": True,
                            "personality_allowed": False,
                        },
                    )
                    self.memory_store.save_memory(resolved_memory)
                    all_memories.append(resolved_memory)
                    recent_memories.append(resolved_memory)
                notes.append(f'background review resolved open question "{question}"')
                continue

            matches = self.memory_store.search_memories(question, limit=5)
            if matches:
                match_count = len(matches)
                if self._last_review_signals.get(signal_key) != match_count:
                    notes.append(
                        f'background review found {match_count} possible memory matches for open question "{question}"'
                    )
                    self._last_review_signals[signal_key] = match_count
            else:
                self._last_review_signals.pop(signal_key, None)

        for question in resolved_questions:
            self.state_manager.remove_open_question(question)

        # 3. Review unsupported claims against memory.
        for claim in unsupported_claims[:10]:
            matches = self.memory_store.search_memories(claim, limit=5)
            if matches:
                signal_key = f"unsupported_claim::{claim}"
                match_count = len(matches)
                if self._last_review_signals.get(signal_key) != match_count:
                    notes.append(
                        f'background review found {match_count} possible memory matches for unsupported claim "{claim}"'
                    )
                    self._last_review_signals[signal_key] = match_count
            else:
                self._last_review_signals.pop(f"unsupported_claim::{claim}", None)

        # 4. Promote sourced structured facts into weighted belief candidates.
        fact_beliefs = self._build_fact_belief_candidates(state.recent_memories)
        if fact_beliefs:
            belief_updates.extend(fact_beliefs)
            inquiry_questions = self.inquiry_engine.generate_questions_from_beliefs(fact_beliefs, max_questions=4)
            for question in inquiry_questions:
                self.state_manager.add_open_question(question)
            if inquiry_questions:
                notes.append(f"generated {len(inquiry_questions)} follow-up questions from weakly supported beliefs")

        # 5. Try to form low-risk belief candidates from repeated note/file memories.
        belief_candidates = self._build_belief_candidates(state.recent_memories)
        belief_updates.extend(belief_candidates)

        current_state = self.state_manager.get_state()
        return CognitionCycleResult(
            cycle_timestamp=now,
            open_question_count=len(current_state.epistemic.open_questions),
            unsupported_claim_count=len(current_state.epistemic.unsupported_claims),
            active_task_count=active_task_count,
            actions=actions,
            belief_updates=belief_updates,
            resolved_questions=resolved_questions,
            notes=notes,
            metadata={
                "cycle_index": self._cycle_index,
                "recent_memory_count": len(state.recent_memories),
            },
        )

    def _has_resolved_question_memory(self, question: str, memories: list[MemoryRecord]) -> bool:
        normalized_question = self._normalize_statement(question)
        for memory in memories:
            if memory.kind != "resolved_question":
                continue
            if self._normalize_statement(memory.content) == normalized_question:
                return True
        return False

    def _build_fact_belief_candidates(self, recent_memories: list[MemoryRecord]) -> list[BeliefRecord]:
        existing_ids = {
            str(memory.metadata.get("belief_id"))
            for memory in recent_memories
            if memory.kind == "belief_candidate" and memory.metadata.get("belief_id")
        }
        fact_memories = [memory for memory in recent_memories if memory.kind == "fact"]
        return self.fact_learning_engine.build_beliefs_from_fact_memories(
            fact_memories,
            existing_belief_ids=existing_ids,
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