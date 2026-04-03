from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.methodology_engine import MethodologyAssessment


@dataclass
class EvidenceAssessment:
    statement: str
    source_id: str
    source_type: str
    supports_statement: bool
    source_weight: float
    methodology_weight: float
    independence_weight: float
    contradiction_penalty: float
    final_support_value: float
    confidence_cap: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "supports_statement": self.supports_statement,
            "source_weight": self.source_weight,
            "methodology_weight": self.methodology_weight,
            "independence_weight": self.independence_weight,
            "contradiction_penalty": self.contradiction_penalty,
            "final_support_value": self.final_support_value,
            "confidence_cap": self.confidence_cap,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class EvidenceEngine:
    """
    Converts a source plus methodology assessment into weighted support.

    Key rule:
    - single sources do not become high confidence on their own
    - source type affects contribution
    - methodology affects contribution
    - convergence across independent sources creates strong belief
    """

    def assess_evidence(
        self,
        statement: str,
        source_id: str,
        source_type: str,
        methodology: Optional[MethodologyAssessment],
        supports_statement: bool = True,
        independence_group: Optional[str] = None,
        contradiction_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EvidenceAssessment:
        metadata = metadata or {}
        reasons: list[str] = []

        source_weight = self._source_weight(source_type)
        reasons.append(f"source_weight:{source_weight:.2f}")

        methodology_weight = methodology.max_support_contribution if methodology else 0.08
        reasons.append(f"methodology_weight:{methodology_weight:.2f}")

        independence_weight = self._independence_weight(independence_group)
        reasons.append(f"independence_weight:{independence_weight:.2f}")

        contradiction_penalty = min(contradiction_count * 0.05, 0.20)
        if contradiction_penalty > 0:
            reasons.append(f"contradiction_penalty:{contradiction_penalty:.2f}")

        raw_support = source_weight * methodology_weight * independence_weight

        if not supports_statement:
            raw_support *= -1.0
            reasons.append("marked_as_contradictory")

        final_support = raw_support
        if final_support > 0:
            final_support = max(0.0, final_support - contradiction_penalty)
        else:
            final_support = min(0.0, final_support + contradiction_penalty)

        confidence_cap = self._single_source_confidence_cap(source_type)
        reasons.append(f"single_source_confidence_cap:{confidence_cap:.2f}")

        return EvidenceAssessment(
            statement=statement,
            source_id=source_id,
            source_type=source_type,
            supports_statement=supports_statement,
            source_weight=round(source_weight, 3),
            methodology_weight=round(methodology_weight, 3),
            independence_weight=round(independence_weight, 3),
            contradiction_penalty=round(contradiction_penalty, 3),
            final_support_value=round(final_support, 3),
            confidence_cap=round(confidence_cap, 3),
            reasons=reasons,
            metadata={
                "independence_group": independence_group,
                "methodology": methodology.to_dict() if methodology else None,
                **metadata,
            },
        )

    def _source_weight(self, source_type: str) -> float:
        source_type = (source_type or "unknown").lower()

        weights = {
            "axiomatic_methodology": 0.82,
            "peer_reviewed_journal": 0.62,
            "preprint": 0.42,
            "official_record": 0.55,
            "technical_documentation": 0.52,
            "sensor_data": 0.58,
            "system_observation": 0.58,
            "review_article": 0.45,
            "book": 0.35,
            "media": 0.18,
            "blog": 0.12,
            "social": 0.08,
            "unknown": 0.15,
        }
        return weights.get(source_type, 0.15)

    def _independence_weight(self, independence_group: Optional[str]) -> float:
        if not independence_group:
            return 1.0
        return 1.0

    def _single_source_confidence_cap(self, source_type: str) -> float:
        source_type = (source_type or "unknown").lower()
        caps = {
            "axiomatic_methodology": 0.88,
            "peer_reviewed_journal": 0.58,
            "preprint": 0.42,
            "official_record": 0.52,
            "technical_documentation": 0.50,
            "sensor_data": 0.58,
            "system_observation": 0.58,
            "review_article": 0.46,
            "book": 0.36,
            "media": 0.22,
            "blog": 0.16,
            "social": 0.10,
            "unknown": 0.20,
        }
        return caps.get(source_type, 0.20)