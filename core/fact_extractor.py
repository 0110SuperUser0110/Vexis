from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractedFact:
    subject: str
    relation: str
    value: str
    confidence: float
    fact_type: str
    source_text: str
    should_store: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "value": self.value,
            "confidence": self.confidence,
            "fact_type": self.fact_type,
            "source_text": self.source_text,
            "should_store": self.should_store,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class FactExtractionResult:
    success: bool
    facts: list[ExtractedFact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "facts": [fact.to_dict() for fact in self.facts],
            "notes": self.notes,
            "error": self.error,
            "metadata": self.metadata,
        }


class FactExtractor:
    """
    Deterministic fact extractor for VEX.

    Goal:
    - convert user teaching into structured facts
    - identify runtime/self-model facts
    - identify object-property facts
    - avoid pretending every sentence is a fact
    """

    def extract(self, text: str) -> FactExtractionResult:
        raw = (text or "").strip()
        normalized = self._normalize(raw)

        if not normalized:
            return FactExtractionResult(
                success=False,
                error="empty_input",
            )

        facts: list[ExtractedFact] = []
        notes: list[str] = []

        runtime_fact = self._extract_runtime_fact(raw, normalized)
        if runtime_fact is not None:
            facts.append(runtime_fact)
            notes.append("runtime fact extracted")

        location_fact = self._extract_location_fact(raw, normalized)
        if location_fact is not None:
            facts.append(location_fact)
            notes.append("location fact extracted")

        identity_fact = self._extract_identity_fact(raw, normalized)
        if identity_fact is not None:
            facts.append(identity_fact)
            notes.append("identity fact extracted")

        object_property_fact = self._extract_object_property_fact(raw, normalized)
        if object_property_fact is not None:
            facts.append(object_property_fact)
            notes.append("object property fact extracted")

        relation_fact = self._extract_general_relation_fact(raw, normalized)
        if relation_fact is not None:
            facts.append(relation_fact)
            notes.append("general relation fact extracted")

        deduped = self._dedupe_facts(facts)

        if not deduped:
            return FactExtractionResult(
                success=False,
                facts=[],
                notes=["no structured fact extracted"],
                metadata={"normalized": normalized},
            )

        return FactExtractionResult(
            success=True,
            facts=deduped,
            notes=notes,
            metadata={"normalized": normalized},
        )

    def _extract_runtime_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (
                r"(?:your|you are on|youre on|you are running on|youre running on|runtime device label is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "running_on",
                "runtime_device",
            ),
            (
                r"(?:device name is|system name is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "running_on",
                "runtime_device",
            ),
        ]

        for pattern, subject, relation, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                value = match.group(1).strip().upper()
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.86,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["runtime_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_location_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (
                r"(?:location label is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "location_label",
                "runtime_location",
            ),
            (
                r"(?:you are at|youre at|you are in|youre in|here is)\s+([a-zA-Z0-9_\-]+)",
                "vex",
                "location_label",
                "runtime_location",
            ),
            (
                r"(?:this is)\s+([a-zA-Z0-9_\-]+)\s*(?:for you|to you)?",
                "vex",
                "location_label",
                "runtime_location",
            ),
        ]

        for pattern, subject, relation, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                value = match.group(1).strip().lower()
                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.78,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["location_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_identity_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        match = re.search(r"(?:your name is|you are|youre)\s+([a-zA-Z0-9_\-]+)", normalized)
        if match and "running on" not in normalized and "at " not in normalized and "in " not in normalized:
            value = match.group(1).strip()
            if value not in {"on", "at", "in"}:
                return ExtractedFact(
                    subject="vex",
                    relation="identity_name",
                    value=value,
                    confidence=0.60,
                    fact_type="identity",
                    source_text=raw,
                    reasons=["identity_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_object_property_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        patterns = [
            (
                r"(?:the )?([a-zA-Z0-9_\-]+)\s+(?:color is|is)\s+([a-zA-Z0-9_\-]+)",
                "attribute",
            ),
            (
                r"(?:the )?([a-zA-Z0-9_\-]+)\s+(?:uses|has)\s+([a-zA-Z0-9_\-]+)",
                "attribute",
            ),
        ]

        for pattern, fact_type in patterns:
            match = re.search(pattern, normalized)
            if match:
                subject = match.group(1).strip().lower()
                value = match.group(2).strip().lower()

                # Avoid swallowing clearly self-model runtime statements here.
                if subject in {"you", "your", "vex"}:
                    return None

                relation = "property"
                if "color" in normalized:
                    relation = "color"

                return ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=value,
                    confidence=0.72,
                    fact_type=fact_type,
                    source_text=raw,
                    reasons=["object_property_pattern_match"],
                    metadata={"normalized_value": value},
                )
        return None

    def _extract_general_relation_fact(self, raw: str, normalized: str) -> Optional[ExtractedFact]:
        match = re.search(
            r"(?:the )?([a-zA-Z0-9_\-]+)\s+(orbits|contains|supports|contradicts|uses|requires)\s+(?:the )?([a-zA-Z0-9_\-]+)",
            normalized,
        )
        if match:
            subject = match.group(1).strip().lower()
            relation = match.group(2).strip().lower()
            value = match.group(3).strip().lower()

            return ExtractedFact(
                subject=subject,
                relation=relation,
                value=value,
                confidence=0.68,
                fact_type="relation",
                source_text=raw,
                reasons=["general_relation_pattern_match"],
                metadata={"normalized_value": value},
            )
        return None

    def _dedupe_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[ExtractedFact] = []

        for fact in facts:
            key = (fact.subject, fact.relation, fact.value)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)

        return deduped

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text