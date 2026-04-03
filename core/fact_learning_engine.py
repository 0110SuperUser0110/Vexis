from __future__ import annotations

from typing import Optional

from core.belief_engine import BeliefEngine, BeliefRecord
from core.evidence_engine import EvidenceEngine
from core.methodology_engine import MethodologyAssessment, MethodologyEngine
from core.schemas import MemoryRecord


class FactLearningEngine:
    """
    Converts structured fact memories into reusable belief candidates.

    This is the step where VEX stops merely storing extracted text and starts
    carrying forward provisional learned beliefs with confidence, provenance,
    contradiction awareness, and scope for revision.
    """

    def __init__(
        self,
        evidence_engine: Optional[EvidenceEngine] = None,
        belief_engine: Optional[BeliefEngine] = None,
        methodology_engine: Optional[MethodologyEngine] = None,
        minimum_confidence: float = 0.08,
    ) -> None:
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self.belief_engine = belief_engine or BeliefEngine()
        self.methodology_engine = methodology_engine or MethodologyEngine()
        self.minimum_confidence = minimum_confidence

    def build_beliefs_from_fact_memories(
        self,
        fact_memories: list[MemoryRecord],
        existing_belief_ids: Optional[set[str]] = None,
    ) -> list[BeliefRecord]:
        existing_belief_ids = existing_belief_ids or set()
        grouped: dict[tuple[str, str, str], list[MemoryRecord]] = {}
        pair_groups: dict[tuple[str, str], dict[str, list[MemoryRecord]]] = {}

        for memory in fact_memories:
            key = self._fact_key(memory)
            if key is None:
                continue
            grouped.setdefault(key, []).append(memory)
            pair_groups.setdefault((key[0], key[1]), {}).setdefault(key[2], []).append(memory)

        beliefs: list[BeliefRecord] = []
        for group_key, items in grouped.items():
            subject, relation, value = self._display_fact(items[0])
            statement = self._render_statement(subject, relation, value)
            belief_id = f"belief_{abs(hash(statement)) % 10_000_000}"
            if belief_id in existing_belief_ids:
                continue

            representatives = self._representative_memories(items)
            conflicting_values = self._conflicting_values(group_key, pair_groups)
            contradiction_count = self._conflicting_source_count(group_key, pair_groups)
            evidence_items = []
            source_paths: list[str] = []
            source_titles: list[str] = []
            supporting_memory_ids: list[str] = []

            for memory in representatives[:6]:
                source_type = str(memory.metadata.get("source_type", "unknown"))
                methodology = self._methodology_for_memory(memory)
                source_id = self._evidence_source_id(memory)
                evidence_items.append(
                    self.evidence_engine.assess_evidence(
                        statement=statement,
                        source_id=source_id,
                        source_type=source_type,
                        methodology=methodology,
                        supports_statement=True,
                        independence_group=self._independence_group(memory),
                        contradiction_count=contradiction_count,
                        metadata={
                            "memory_id": memory.memory_id,
                            "source_path": memory.metadata.get("source_path"),
                            "title": memory.metadata.get("title"),
                            "fact_relation": relation,
                            "fact_value": value,
                            "conflicting_values": conflicting_values,
                        },
                    )
                )
                supporting_memory_ids.append(memory.memory_id)
                source_path = memory.metadata.get("source_path")
                title = memory.metadata.get("title")
                if source_path:
                    source_paths.append(str(source_path))
                if title:
                    source_titles.append(str(title))

            if not evidence_items:
                continue

            belief = self.belief_engine.build_belief(
                belief_id=belief_id,
                statement=statement,
                evidence_items=evidence_items,
                metadata={
                    "origin": "fact_learning_engine",
                    "subject": subject,
                    "relation": relation,
                    "value": value,
                    "render_text": statement,
                    "source_paths": list(dict.fromkeys(source_paths)),
                    "source_titles": list(dict.fromkeys(source_titles)),
                    "supporting_memory_ids": supporting_memory_ids,
                    "source_type": str(items[0].metadata.get("source_type", "unknown")),
                    "fact_type": str(items[0].metadata.get("fact_type", items[0].kind)),
                    "independence_group": self._independence_group(items[0]),
                    "fact_group_key": "|".join(group_key),
                    "conflicting_values": conflicting_values,
                    "conflict_count": contradiction_count,
                    "independent_source_count": len(representatives),
                },
            )

            self._apply_learning_floor(belief, items[0], relation, contradiction_count)
            if belief.confidence_score >= self.minimum_confidence:
                beliefs.append(belief)

        return beliefs

    def _fact_key(self, memory: MemoryRecord) -> Optional[tuple[str, str, str]]:
        metadata = memory.metadata or {}
        subject = str(metadata.get("subject", "")).strip()
        relation = str(metadata.get("relation", "")).strip()
        value = str(metadata.get("value", "")).strip()
        if not subject or not relation or not value:
            return None
        return (subject.lower(), relation.lower(), value.lower())

    def _display_fact(self, memory: MemoryRecord) -> tuple[str, str, str]:
        metadata = memory.metadata or {}
        subject = str(metadata.get("subject", "")).strip() or memory.content
        relation = str(metadata.get("relation", "")).strip() or "states"
        value = str(metadata.get("value", "")).strip()
        return subject, relation, value

    def _render_statement(self, subject: str, relation: str, value: str) -> str:
        relation_key = relation.lower()
        if relation_key == "defined_as":
            return f"{subject} is {value}."
        if relation_key == "equals":
            return f"{subject} equals {value}."
        if relation_key == "equals_expression":
            return f"{subject} equals {value}."
        if relation_key == "successor_rule":
            return f"For integers, {value}."
        if relation_key == "color":
            return f"The {subject} is {value}."
        if relation_key == "depends_on":
            return f"{subject} depends on {value}."
        if relation_key == "proportional_to":
            return f"{subject} is proportional to {value}."
        if relation_key == "inversely_proportional_to":
            return f"{subject} is inversely proportional to {value}."
        if relation_key == "increases_with":
            return f"{subject} increases as {value} increases."
        if relation_key == "decreases_with":
            return f"{subject} decreases as {value} increases."
        if relation_key == "conserved_in":
            return f"{subject} is conserved in {value}."
        if relation_key == "holds_when":
            return f"{subject} holds when {value}."
        if relation_key == "cannot_exceed":
            return f"{subject} cannot exceed {value}."
        relation_text = relation_key.replace("_", " ")
        return f"{subject} {relation_text} {value}."

    def _representative_memories(self, memories: list[MemoryRecord]) -> list[MemoryRecord]:
        grouped: dict[str, MemoryRecord] = {}
        for memory in memories:
            key = self._evidence_source_id(memory)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = memory
                continue
            current_confidence = float(memory.metadata.get("confidence", 0.0))
            existing_confidence = float(existing.metadata.get("confidence", 0.0))
            if current_confidence > existing_confidence:
                grouped[key] = memory
        return list(grouped.values())

    def _conflicting_values(
        self,
        group_key: tuple[str, str, str],
        pair_groups: dict[tuple[str, str], dict[str, list[MemoryRecord]]],
    ) -> list[str]:
        subject_key, relation_key, value_key = group_key
        grouped_values = pair_groups.get((subject_key, relation_key), {})
        conflicts: list[str] = []
        for other_value, memories in grouped_values.items():
            if other_value == value_key:
                continue
            if not memories:
                continue
            conflicts.append(self._display_fact(memories[0])[2])
        return list(dict.fromkeys(conflicts))[:4]

    def _conflicting_source_count(
        self,
        group_key: tuple[str, str, str],
        pair_groups: dict[tuple[str, str], dict[str, list[MemoryRecord]]],
    ) -> int:
        subject_key, relation_key, value_key = group_key
        grouped_values = pair_groups.get((subject_key, relation_key), {})
        count = 0
        for other_value, memories in grouped_values.items():
            if other_value == value_key:
                continue
            count += len(self._representative_memories(memories))
        return count

    def _methodology_for_memory(self, memory: MemoryRecord) -> Optional[MethodologyAssessment]:
        source_type = str(memory.metadata.get("source_type", "unknown")).lower()
        if source_type == "axiomatic_methodology":
            return MethodologyAssessment(
                study_type="axiomatic_methodology",
                rigor_score=0.95,
                rigor_label="high",
                sample_strength="strong",
                control_strength="strong",
                statistics_strength="strong",
                overreach_detected=False,
                limitations_acknowledged=True,
                replication_signal="positive",
                usable_for_belief_update=True,
                max_support_contribution=0.72,
                reasons=["bootstrapped_axiomatic_methodology"],
                metadata={"source_type": source_type, "title": memory.metadata.get("title")},
            )
        if source_type in {
            "peer_reviewed_journal",
            "preprint",
            "review_article",
            "official_record",
            "technical_documentation",
            "book",
        }:
            return self.methodology_engine.assess(
                text=memory.content,
                metadata={
                    "source_type": source_type,
                    "title": memory.metadata.get("title"),
                    "year": memory.metadata.get("year"),
                },
            )
        return None

    def _evidence_source_id(self, memory: MemoryRecord) -> str:
        independence_group = memory.metadata.get("independence_group")
        source_path = memory.metadata.get("source_path")
        title = memory.metadata.get("title")
        if independence_group:
            return str(independence_group)
        if source_path:
            return str(source_path)
        if title:
            return str(title)
        return memory.memory_id

    def _independence_group(self, memory: MemoryRecord) -> Optional[str]:
        value = memory.metadata.get("independence_group") or memory.metadata.get("source_path")
        return str(value) if value else None

    def _apply_learning_floor(
        self,
        belief: BeliefRecord,
        memory: MemoryRecord,
        relation: str,
        contradiction_count: int,
    ) -> None:
        source_type = str(memory.metadata.get("source_type", "unknown")).lower()
        source_floor = {
            "axiomatic_methodology": 0.90,
            "peer_reviewed_journal": 0.24,
            "official_record": 0.22,
            "technical_documentation": 0.20,
            "review_article": 0.20,
            "book": 0.18,
        }.get(source_type)
        if source_floor is None:
            return
        if contradiction_count > 0:
            source_floor = min(source_floor, 0.16)
        if belief.confidence_score >= source_floor:
            return

        if relation.lower() not in {
            "defined_as",
            "equals",
            "equals_expression",
            "successor_rule",
            "color",
            "property",
            "location_label",
            "identity_name",
            "supports",
            "requires",
            "contains",
            "uses",
            "causes",
            "depends_on",
            "proportional_to",
            "inversely_proportional_to",
            "increases",
            "decreases",
            "raises",
            "lowers",
            "reduces",
            "conserved_in",
            "holds_when",
            "cannot_exceed",
        }:
            source_floor = max(source_floor - 0.02, 0.14)

        belief.confidence_score = round(max(belief.confidence_score, source_floor), 3)
        belief.confidence_label = self._confidence_label(belief.confidence_score)
        belief.status = self._status_from_confidence(belief.confidence_score, belief.support_count)
        belief.reasons.append(f"source_weighted_learning_floor_applied:{source_floor:.2f}")
        if contradiction_count > 0:
            belief.reasons.append(f"contradiction_penalty_applied:{contradiction_count}")

    def _confidence_label(self, score: float) -> str:
        if score >= 0.85:
            return "established"
        if score >= 0.65:
            return "high"
        if score >= 0.40:
            return "moderate"
        if score >= 0.18:
            return "low"
        return "very_low"

    def _status_from_confidence(self, score: float, support_count: int) -> str:
        if score >= 0.85 and support_count >= 3:
            return "established"
        if score >= 0.65 and support_count >= 2:
            return "supported"
        if score >= 0.40:
            return "provisional"
        return "candidate"
