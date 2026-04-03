from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.belief_engine import BeliefRecord
from core.fact_extractor import ExtractedFact, FactExtractionResult, FactExtractor
from core.grounded_answer_engine import GroundedAnswerEngine
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
    - resolve direct arithmetic deterministically when possible
    - decide whether question is resolved, provisional, or unresolved
    """

    def __init__(
        self,
        self_model: Optional[SelfModel] = None,
        fact_extractor: Optional[FactExtractor] = None,
        grounded_answer_engine: Optional[GroundedAnswerEngine] = None,
    ) -> None:
        self.self_model = self_model
        self.fact_extractor = fact_extractor or FactExtractor()
        self.grounded_answer_engine = grounded_answer_engine or GroundedAnswerEngine()

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

        if question_type == "arithmetic":
            return self._resolve_arithmetic(normalized, state, recent_memories)

        if question_type == "self_location":
            return self._resolve_self_location(normalized, state, recent_memories)

        if question_type == "runtime_device":
            return self._resolve_runtime_device(normalized, state, recent_memories)

        if question_type == "repeat_capability":
            return self._resolve_repeat_capability()

        if question_type == "legacy_cube_presence":
            return self._resolve_legacy_cube_presence()

        if question_type == "self_identity":
            return self._resolve_self_identity(normalized)

        if question_type == "object_property":
            return self._resolve_object_property(normalized, recent_memories, belief_records)

        grounded_resolution = self._resolve_grounded_generic_question(question_text, recent_memories)
        if grounded_resolution is not None:
            return grounded_resolution

        return ResolutionResult(
            resolved=False,
            question_type=question_type,
            answer_text="I do not yet know enough to answer that directly.",
            confidence=0.14,
            follow_up_question=self._generic_follow_up_question(normalized),
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

    def _resolve_arithmetic(
        self,
        normalized_question: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
    ) -> ResolutionResult:
        parsed = self._parse_arithmetic_question(normalized_question)
        if parsed is None:
            return ResolutionResult(
                resolved=False,
                question_type="arithmetic",
                answer_text="I can tell this is arithmetic-shaped, but the expression is not specific enough yet.",
                confidence=0.18,
                follow_up_question="Which arithmetic expression should I evaluate?",
                should_store_resolution=False,
                reasons=["arithmetic_expression_ambiguous"],
            )

        left, operator, right = parsed
        computed_result = self._compute_arithmetic_result(left, operator, right)
        if computed_result is None:
            return ResolutionResult(
                resolved=False,
                question_type="arithmetic",
                answer_text="That arithmetic request is not resolvable in its current form.",
                confidence=0.12,
                follow_up_question="Is there a valid arithmetic operation here, or am I looking at division by zero?",
                should_store_resolution=False,
                reasons=["arithmetic_operation_invalid"],
                metadata={
                    "left_operand": left,
                    "right_operand": right,
                    "operator": operator,
                },
            )

        expression_key = f"{self._number_to_string(left)}{operator}{self._number_to_string(right)}"
        result_text = self._number_to_string(computed_result)
        supporting_facts: list[dict[str, Any]] = [
            {
                "source": "deterministic_arithmetic",
                "expression": expression_key,
                "value": result_text,
            }
        ]
        reasons = ["resolved_by_deterministic_arithmetic"]
        confidence = 0.95

        combined_memories = self._combine_recent_memories(state.recent_memories, recent_memories)
        extracted = self._extract_recent_facts(combined_memories)

        exact_matches: list[ExtractedFact] = []
        rule_matches: list[ExtractedFact] = []
        for fact in extracted:
            if self._matches_arithmetic_equation_fact(fact, left, operator, right, computed_result):
                exact_matches.append(fact)
            if operator == "+" and self._numbers_equal(right, 1) and fact.relation == "successor_rule":
                rule_matches.append(fact)

        if exact_matches:
            best = exact_matches[-1]
            supporting_facts.append(best.to_dict())
            reasons.append("supported_by_learned_arithmetic_fact")
            confidence = max(confidence, min(best.confidence, 0.99))
        elif rule_matches:
            best = rule_matches[-1]
            supporting_facts.append(best.to_dict())
            reasons.append("supported_by_learned_arithmetic_rule")
            confidence = max(confidence, min(best.confidence + 0.06, 0.96))

        operator_display = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
        }.get(operator, operator)

        return ResolutionResult(
            resolved=True,
            question_type="arithmetic",
            answer_text=(
                f"{self._number_to_string(left)} {operator_display} "
                f"{self._number_to_string(right)} = {result_text}."
            ),
            confidence=confidence,
            supporting_facts=supporting_facts,
            should_store_resolution=True,
            reasons=reasons,
            metadata={
                "expression": expression_key,
                "left_operand": left,
                "right_operand": right,
                "operator": operator,
                "result": computed_result,
            },
        )

    def _resolve_self_location(
        self,
        normalized_question: str,
        state: VexisState,
        recent_memories: list[MemoryRecord],
    ) -> ResolutionResult:
        supporting_facts: list[dict[str, Any]] = []
        reasons: list[str] = []

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

    def _resolve_repeat_capability(self) -> ResolutionResult:
        return ResolutionResult(
            resolved=True,
            question_type="repeat_capability",
            answer_text="Yes. I can repeat user-provided text back to you.",
            confidence=0.82,
            supporting_facts=[
                {
                    "source": "self_model_capability",
                    "subject": "vex",
                    "relation": "can_repeat_user_text",
                    "value": "true",
                }
            ],
            should_store_resolution=True,
            reasons=["resolved_from_active_conversation_capability"],
        )

    def _resolve_legacy_cube_presence(self) -> ResolutionResult:
        return ResolutionResult(
            resolved=True,
            question_type="legacy_cube_presence",
            answer_text="I do not currently use a cube presence model, so there is no cube color to report.",
            confidence=0.9,
            supporting_facts=[
                {
                    "source": "presence_configuration",
                    "subject": "vex",
                    "relation": "presence_model",
                    "value": "non_cube_avatar",
                }
            ],
            should_store_resolution=True,
            reasons=["resolved_from_current_presence_configuration"],
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

    def collect_recent_facts(self, recent_memories: list[MemoryRecord]) -> list[ExtractedFact]:
        return self._extract_recent_facts(recent_memories)

    def _resolve_grounded_generic_question(
        self,
        question_text: str,
        recent_memories: list[MemoryRecord],
    ) -> Optional[ResolutionResult]:
        grounded = self.grounded_answer_engine.resolve(question_text, recent_memories)
        if not grounded.resolved:
            return None

        supporting_facts = [
            {
                "source": "grounded_memory",
                "value": point,
            }
            for point in grounded.supporting_points[:3]
        ]
        supporting_facts.extend(
            {
                "source": "grounded_memory_id",
                "value": memory_id,
            }
            for memory_id in grounded.memory_ids[:3]
        )

        return ResolutionResult(
            resolved=True,
            question_type="generic_knowledge",
            answer_text=grounded.answer_text,
            confidence=grounded.confidence,
            supporting_facts=supporting_facts,
            should_store_resolution=True,
            reasons=list(grounded.reasons),
            metadata=grounded.to_dict(),
        )

    def _generic_follow_up_question(self, normalized_question: str) -> str:
        if normalized_question.startswith(("what ", "which ")):
            return "What source, definition, or observation would establish the answer?"
        if normalized_question.startswith("why "):
            return "What mechanism or evidence would explain why that is true?"
        if normalized_question.startswith("how "):
            return "What process or step-by-step evidence would answer that?"
        if normalized_question.startswith("where "):
            return "What location evidence would identify that clearly?"
        if normalized_question.startswith("who "):
            return "What identifying source would determine who that is?"
        return "What specific fact would resolve this question?"

    def _extract_recent_facts(self, recent_memories: list[MemoryRecord]) -> list[ExtractedFact]:
        extracted: list[ExtractedFact] = []

        for memory in recent_memories[-50:]:
            metadata_fact = self._fact_from_memory_metadata(memory)
            if metadata_fact is not None:
                extracted.append(metadata_fact)

            result = self.fact_extractor.extract(memory.content)
            if result.success:
                extracted.extend(result.facts)

        return extracted

    def _question_type(self, normalized_question: str) -> str:
        if self._parse_arithmetic_question(normalized_question) is not None:
            return "arithmetic"
        if "where is here" in normalized_question or "where are you" in normalized_question:
            return "self_location"
        if "runtime device" in normalized_question or "what device are you on" in normalized_question:
            return "runtime_device"
        if "repeat what i tell" in normalized_question or "repeat what i say" in normalized_question:
            return "repeat_capability"
        if "what color is your cube" in normalized_question or "what is the color of your cube" in normalized_question:
            return "legacy_cube_presence"
        if (
            "what is your name" in normalized_question
            or "who are you" in normalized_question
            or "what are you" in normalized_question
        ):
            return "self_identity"
        object_name, relation = self._object_property_target(normalized_question)
        if object_name is not None and relation is not None:
            return "object_property"
        return "generic"

    def _object_property_target(self, normalized_question: str) -> tuple[Optional[str], Optional[str]]:
        color_patterns = (
            r"what color is (?:the )?(.+)",
            r"what is the color of (?:the )?(.+)",
        )
        for pattern in color_patterns:
            match = re.search(pattern, normalized_question)
            if match:
                subject = match.group(1).strip(" ?.")
                if subject:
                    return subject, "color"

        relation_patterns = (
            r"what is the ([a-z0-9_\-]+) of (?:the )?(.+)",
            r"what is ([a-z0-9_\-]+) of (?:the )?(.+)",
            r"what is (?:the )?(.+?)'s ([a-z0-9_\-]+)",
        )
        for pattern in relation_patterns:
            match = re.search(pattern, normalized_question)
            if not match:
                continue
            if "'s" in pattern:
                subject = match.group(1).strip(" ?.")
                relation = match.group(2).strip(" ?.")
            else:
                relation = match.group(1).strip(" ?.")
                subject = match.group(2).strip(" ?.")
            if subject and relation:
                return subject, relation

        return None, None

    def _parse_arithmetic_question(self, normalized_question: str) -> Optional[tuple[float, str, float]]:
        text = normalized_question.strip().rstrip("?.!")
        patterns = [
            r"(?:what is|calculate|compute|evaluate)?\s*(-?\d+(?:\.\d+)?)\s*(\+|\-|\*|x|/)\s*(-?\d+(?:\.\d+)?)",
            r"(?:what is|calculate|compute|evaluate)?\s*(-?\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by)\s+(-?\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue

            left = self._parse_number(match.group(1))
            operator = self._normalize_operator(match.group(2))
            right = self._parse_number(match.group(3))
            if left is None or operator is None or right is None:
                continue
            return left, operator, right

        return None

    def _matches_arithmetic_equation_fact(
        self,
        fact: ExtractedFact,
        left: float,
        operator: str,
        right: float,
        computed_result: float,
    ) -> bool:
        if fact.relation != "equals":
            return False

        metadata = fact.metadata or {}
        left_meta = self._parse_number(metadata.get("left_operand"))
        right_meta = self._parse_number(metadata.get("right_operand"))
        operator_meta = self._normalize_operator(str(metadata.get("operator", ""))) if metadata.get("operator") is not None else None
        result_meta = self._parse_number(metadata.get("result"))

        if (
            left_meta is not None
            and right_meta is not None
            and operator_meta is not None
            and result_meta is not None
        ):
            return (
                self._numbers_equal(left_meta, left)
                and self._numbers_equal(right_meta, right)
                and operator_meta == operator
                and self._numbers_equal(result_meta, computed_result)
            )

        expression_key = f"{self._number_to_string(left)}{operator}{self._number_to_string(right)}"
        fact_subject = str(fact.subject).replace(" ", "")
        fact_value = self._parse_number(fact.value)
        return fact_subject == expression_key and fact_value is not None and self._numbers_equal(fact_value, computed_result)

    def _combine_recent_memories(
        self,
        state_memories: list[MemoryRecord],
        context_memories: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        combined: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for memory in list(state_memories[-50:]) + list(context_memories[-20:]):
            if memory.memory_id in seen_ids:
                continue
            seen_ids.add(memory.memory_id)
            combined.append(memory)

        return combined

    def _fact_from_memory_metadata(self, memory: MemoryRecord) -> Optional[ExtractedFact]:
        metadata = memory.metadata or {}
        subject = metadata.get("subject")
        relation = metadata.get("relation")
        value = metadata.get("value")
        if not subject or not relation or value is None:
            return None

        nested_metadata = dict(metadata.get("metadata", {}) or {})
        nested_metadata.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in {"subject", "relation", "value", "source_type", "confidence", "fact_type", "reasons", "metadata"}
            }
        )

        return ExtractedFact(
            subject=str(subject),
            relation=str(relation),
            value=str(value),
            confidence=float(metadata.get("confidence", 0.74)),
            fact_type=str(metadata.get("fact_type", memory.kind)),
            source_text=memory.content,
            reasons=list(metadata.get("reasons", [])),
            metadata=nested_metadata,
        )

    def _normalize_operator(self, operator: str) -> Optional[str]:
        mapping = {
            "+": "+",
            "-": "-",
            "*": "*",
            "x": "*",
            "/": "/",
            "plus": "+",
            "minus": "-",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
        }
        return mapping.get(operator.strip().lower())

    def _compute_arithmetic_result(self, left: float, operator: str, right: float) -> Optional[float]:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            if right == 0:
                return None
            return left / right
        return None

    def _parse_number(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _numbers_equal(self, left: float, right: float, tolerance: float = 1e-9) -> bool:
        return abs(float(left) - float(right)) <= tolerance

    def _number_to_string(self, value: float) -> str:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(round(numeric, 8)).rstrip("0").rstrip(".")
