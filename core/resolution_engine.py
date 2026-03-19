from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.belief_engine import BeliefRecord
from core.fact_extractor import ExtractedFact, FactExtractionResult, FactExtractor
from core.self_model import SelfModel
from core.schemas import MemoryRecord, VexisState


@dataclass
class ResolutionResult:
    resolved: bool
    question_type: str
    answer_text: str
    confidence: float
    supporting_facts: list[dict[str, Any]] = field(default_factory=list)
    follow_up_question: Optional[str] = None
    should_store_resolution: bool = False
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "question_type": self.question_type,
            "answer_text": self.answer_text,
            "confidence": self.confidence,
            "supporting_facts": self.supporting_facts,
            "follow_up_question": self.follow_up_question,
            "should_store_resolution": self.should_store_resolution,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class ResolutionEngine:
    """
    Deterministic question-resolution engine for VEX.

    Responsibilities:
    - identify what missing fact would answer a question
    - search self model
    - search recent structured memory
    - search extracted facts from recent user teaching
    - decide whether question is resolved, provisional, or unresolved
    """

    def __init__(
        self,
        self_model: Optional[SelfModel] = None,
        fact_extractor: Optional[FactExtractor] = None,
    ) -> None:
        self.self_model = self_model
        self.fact_extractor = fact_extractor or FactExtractor()

    def resolve_question(
        self,
        question_text: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
        belief_records: Optional[list[BeliefRecord]] = None,
    ) -> ResolutionResult:
        normalized = (question_text or "").strip().lower()
        belief_records = belief_records or []

        question_type = self._question_type(normalized)

        if question_type == "self_location":
            return self._resolve_self_location(normalized, state, recent_memories)

        if question_type == "runtime_device":
            return self._resolve_runtime_device(normalized, state, recent_memories)

        if question_type == "self_identity":
            return self._resolve_self_identity(normalized)

        if question_type == "object_property":
            return self._resolve_object_property(normalized, recent_memories, belief_records)

        return ResolutionResult(
            resolved=False,
            question_type=question_type,
            answer_text="I do not yet know enough to answer that directly.",
            confidence=0.14,
            follow_up_question="What specific fact would resolve this question?",
            should_store_resolution=False,
            reasons=["no_resolution_path_available"],
            metadata={"normalized_question": normalized},
        )

    def learn_from_user_statement(
        self,
        statement_text: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
    ) -> FactExtractionResult:
        """
        Converts a user statement into structured facts for the deterministic core.
        """
        return self.fact_extractor.extract(statement_text)

    def _resolve_self_location(
        self,
        normalized_question: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
    ) -> ResolutionResult:
        supporting_facts: list[dict[str, Any]] = []
        reasons: list[str] = []

        # 1. Self-model / runtime state
        if self.self_model is not None:
            snapshot = self.self_model.snapshot()
            if snapshot.runtime.location_label and snapshot.runtime.location_label != "unknown_location":
                supporting_facts.append(
                    {
                        "source": "self_model",
                        "subject": "vex",
                        "relation": "location_label",
                        "value": snapshot.runtime.location_label,
                    }
                )
                reasons.append("resolved_from_self_model_runtime")

                answer = (
                    f'I am running in location context "{snapshot.runtime.location_label}" '
                    f'on device {snapshot.runtime.device_label}.'
                )
                return ResolutionResult(
                    resolved=True,
                    question_type="self_location",
                    answer_text=answer,
                    confidence=0.88,
                    supporting_facts=supporting_facts,
                    should_store_resolution=True,
                    reasons=reasons,
                    metadata={"runtime_snapshot": snapshot.to_dict()},
                )

        # 2. Recent user-taught facts
        extracted = self._extract_recent_facts(recent_memories)
        location_candidates = [
            fact for fact in extracted
            if fact.subject == "vex" and fact.relation == "location_label"
        ]
        if location_candidates:
            best = location_candidates[-1]
            supporting_facts.append(best.to_dict())
            reasons.append("resolved_from_recent_user_taught_fact")

            answer = f'I am currently treating "{best.value}" as my location context.'
            return ResolutionResult(
                resolved=True,
                question_type="self_location",
                answer_text=answer,
                confidence=min(best.confidence, 0.80),
                supporting_facts=supporting_facts,
                should_store_resolution=True,
                reasons=reasons,
                metadata={"fact_source": "recent_memory_extraction"},
            )

        return ResolutionResult(
            resolved=False,
            question_type="self_location",
            answer_text="I do not yet know what 'here' refers to in a grounded way.",
            confidence=0.18,
            follow_up_question="What location label should I use for 'here'?",
            should_store_resolution=False,
            reasons=["missing_location_label"],
        )

    def _resolve_runtime_device(
        self,
        normalized_question: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
    ) -> ResolutionResult:
        supporting_facts: list[dict[str, Any]] = []
        reasons: list[str] = []

        if self.self_model is not None:
            snapshot = self.self_model.snapshot()
            if snapshot.runtime.device_label and snapshot.runtime.device_label != "unknown_device":
                supporting_facts.append(
                    {
                        "source": "self_model",
                        "subject": "vex",
                        "relation": "running_on",
                        "value": snapshot.runtime.device_label,
                    }
                )
                reasons.append("resolved_from_self_model_runtime")

                return ResolutionResult(
                    resolved=True,
                    question_type="runtime_device",
                    answer_text=f"I am running on {snapshot.runtime.device_label}.",
                    confidence=0.90,
                    supporting_facts=supporting_facts,
                    should_store_resolution=True,
                    reasons=reasons,
                    metadata={"runtime_snapshot": snapshot.to_dict()},
                )

        extracted = self._extract_recent_facts(recent_memories)
        device_candidates = [
            fact for fact in extracted
            if fact.subject == "vex" and fact.relation == "running_on"
        ]
        if device_candidates:
            best = device_candidates[-1]
            supporting_facts.append(best.to_dict())
            reasons.append("resolved_from_recent_user_taught_fact")

            return ResolutionResult(
                resolved=True,
                question_type="runtime_device",
                answer_text=f"I am currently treating {best.value} as my runtime device.",
                confidence=min(best.confidence, 0.82),
                supporting_facts=supporting_facts,
                should_store_resolution=True,
                reasons=reasons,
                metadata={"fact_source": "recent_memory_extraction"},
            )

        return ResolutionResult(
            resolved=False,
            question_type="runtime_device",
            answer_text="I do not yet have a grounded runtime device label.",
            confidence=0.18,
            follow_up_question="What runtime device label should I use?",
            should_store_resolution=False,
            reasons=["missing_runtime_device_label"],
        )

    def _resolve_self_identity(self, normalized_question: str) -> ResolutionResult:
        if self.self_model is not None:
            snapshot = self.self_model.snapshot()
            return ResolutionResult(
                resolved=True,
                question_type="self_identity",
                answer_text=(
                    f"I am {snapshot.identity.system_label}, with the active identity name "
                    f"{snapshot.identity.name}."
                ),
                confidence=0.92,
                supporting_facts=[
                    {
                        "source": "self_model",
                        "subject": "vex",
                        "relation": "identity_name",
                        "value": snapshot.identity.name,
                    }
                ],
                should_store_resolution=False,
                reasons=["resolved_from_self_model_identity"],
                metadata={"runtime_snapshot": snapshot.to_dict()},
            )

        return ResolutionResult(
            resolved=False,
            question_type="self_identity",
            answer_text="I do not yet have a stable identity model available.",
            confidence=0.16,
            follow_up_question="What identity name should I use?",
            should_store_resolution=False,
            reasons=["self_model_unavailable"],
        )

    def _resolve_object_property(
        self,
        normalized_question: str,
        recent_memories: list[MemoryRecord],
        belief_records: list[BeliefRecord],
    ) -> ResolutionResult:
        object_name, relation = self._object_property_target(normalized_question)
        if object_name is None or relation is None:
            return ResolutionResult(
                resolved=False,
                question_type="object_property",
                answer_text="I could not determine the target object-property pair clearly enough.",
                confidence=0.14,
                follow_up_question="What object and property should I evaluate?",
                should_store_resolution=False,
                reasons=["object_property_target_ambiguous"],
            )

        supporting_facts: list[dict[str, Any]] = []
        reasons: list[str] = []

        extracted = self._extract_recent_facts(recent_memories)
        matches = [
            fact for fact in extracted
            if fact.subject == object_name and (fact.relation == relation or relation == "property")
        ]
        if matches:
            best = matches[-1]
            supporting_facts.append(best.to_dict())
            reasons.append("resolved_from_recent_extracted_fact")

            answer = f"The current {best.relation} of {best.subject} is {best.value}."
            return ResolutionResult(
                resolved=True,
                question_type="object_property",
                answer_text=answer,
                confidence=min(best.confidence, 0.78),
                supporting_facts=supporting_facts,
                should_store_resolution=True,
                reasons=reasons,
                metadata={"fact_source": "recent_memory_extraction"},
            )

        belief_matches = [
            belief for belief in belief_records
            if object_name in belief.statement.lower()
        ]
        if belief_matches:
            best_belief = sorted(
                belief_matches,
                key=lambda belief: belief.confidence_score,
                reverse=True,
            )[0]
            supporting_facts.append(
                {
                    "source": "belief_record",
                    "statement": best_belief.statement,
                    "confidence_score": best_belief.confidence_score,
                    "confidence_label": best_belief.confidence_label,
                }
            )
            reasons.append("resolved_from_belief_record")

            return ResolutionResult(
                resolved=True,
                question_type="object_property",
                answer_text=best_belief.statement,
                confidence=min(best_belief.confidence_score, 0.86),
                supporting_facts=supporting_facts,
                should_store_resolution=False,
                reasons=reasons,
                metadata={"belief_id": best_belief.belief_id},
            )

        return ResolutionResult(
            resolved=False,
            question_type="object_property",
            answer_text="I do not yet have enough grounded support to answer that object-property question.",
            confidence=0.16,
            follow_up_question=f"What is the {relation} of {object_name}?",
            should_store_resolution=False,
            reasons=["missing_object_property_support"],
        )

    def _extract_recent_facts(self, recent_memories: list[MemoryRecord]) -> list[ExtractedFact]:
        extracted: list[ExtractedFact] = []

        for memory in recent_memories[-20:]:
            result = self.fact_extractor.extract(memory.content)
            if result.success:
                extracted.extend(result.facts)

        return extracted

    def _question_type(self, normalized_question: str) -> str:
        if "where is here" in normalized_question or "where are you" in normalized_question:
            return "self_location"
        if "runtime device" in normalized_question or "what device are you on" in normalized_question:
            return "runtime_device"
        if "what is your name" in normalized_question or "who are you" in normalized_question or "what are you" in normalized_question:
            return "self_identity"
        if any(phrase in normalized_question for phrase in ("what color", "what is the color", "what property", "what is the")):
            return "object_property"
        return "generic"

    def _object_property_target(self, normalized_question: str) -> tuple[Optional[str], Optional[str]]:
        if "what color is the" in normalized_question:
            tail = normalized_question.split("what color is the", 1)[1].strip(" ?.")
            if tail:
                return tail, "color"

        if "what is the color of the" in normalized_question:
            tail = normalized_question.split("what is the color of the", 1)[1].strip(" ?.")
            if tail:
                return tail, "color"

        if "what is the" in normalized_question:
            tail = normalized_question.split("what is the", 1)[1].strip(" ?.")
            words = tail.split()
            if len(words) >= 2:
                relation = words[0]
                subject = " ".join(words[1:])
                return subject, relation

        return None, None