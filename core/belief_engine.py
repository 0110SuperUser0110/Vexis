from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.evidence_engine import EvidenceAssessment


@dataclass
class BeliefRecord:
    belief_id: str
    statement: str
    confidence_score: float
    confidence_label: str
    status: str
    support_count: int
    contradiction_count: int
    supporting_sources: list[str] = field(default_factory=list)
    contradicting_sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "statement": self.statement,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "status": self.status,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class BeliefEngine:
    """
    Aggregates evidence assessments into confidence-scored beliefs.

    Critical rule:
    - high confidence belongs to convergent conclusions
    - not to a single source
    """

    def build_belief(
        self,
        belief_id: str,
        statement: str,
        evidence_items: list[EvidenceAssessment],
        metadata: Optional[dict[str, Any]] = None,
    ) -> BeliefRecord:
        metadata = metadata or {}
        reasons: list[str] = []

        support_items = [item for item in evidence_items if item.final_support_value > 0]
        contradiction_items = [item for item in evidence_items if item.final_support_value < 0]

        support_sum = sum(item.final_support_value for item in support_items)
        contradiction_sum = abs(sum(item.final_support_value for item in contradiction_items))

        support_sources = [item.source_id for item in support_items]
        contradiction_sources = [item.source_id for item in contradiction_items]

        unique_support_sources = len(set(support_sources))
        unique_contradiction_sources = len(set(contradiction_sources))

        if unique_support_sources > 0:
            reasons.append(f"supporting_sources:{unique_support_sources}")
        if unique_contradiction_sources > 0:
            reasons.append(f"contradicting_sources:{unique_contradiction_sources}")

        convergence_bonus = self._convergence_bonus(unique_support_sources)
        contradiction_penalty = self._contradiction_penalty(unique_contradiction_sources, contradiction_sum)

        confidence_score = support_sum + convergence_bonus - contradiction_penalty

        # Apply single-source cap if only one supporting source exists.
        if unique_support_sources == 1 and support_items:
            single_cap = max(item.confidence_cap for item in support_items)
            confidence_score = min(confidence_score, single_cap)
            reasons.append(f"single_source_cap_applied:{single_cap:.2f}")

        confidence_score = max(0.0, min(confidence_score, 0.99))
        confidence_label = self._confidence_label(confidence_score)
        status = self._status_from_confidence(confidence_score, unique_support_sources, unique_contradiction_sources)

        reasons.append(f"support_sum:{support_sum:.3f}")
        reasons.append(f"contradiction_sum:{contradiction_sum:.3f}")
        reasons.append(f"convergence_bonus:{convergence_bonus:.3f}")
        reasons.append(f"final_confidence:{confidence_score:.3f}")

        return BeliefRecord(
            belief_id=belief_id,
            statement=statement,
            confidence_score=round(confidence_score, 3),
            confidence_label=confidence_label,
            status=status,
            support_count=unique_support_sources,
            contradiction_count=unique_contradiction_sources,
            supporting_sources=list(dict.fromkeys(support_sources)),
            contradicting_sources=list(dict.fromkeys(contradiction_sources)),
            reasons=reasons,
            metadata=metadata,
        )

    def _convergence_bonus(self, unique_support_sources: int) -> float:
        if unique_support_sources <= 1:
            return 0.0
        if unique_support_sources == 2:
            return 0.08
        if unique_support_sources == 3:
            return 0.16
        if unique_support_sources == 4:
            return 0.24
        return 0.30

    def _contradiction_penalty(self, contradiction_count: int, contradiction_sum: float) -> float:
        return min((contradiction_count * 0.06) + (contradiction_sum * 0.4), 0.45)

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

    def _status_from_confidence(
        self,
        score: float,
        support_count: int,
        contradiction_count: int,
    ) -> str:
        if score >= 0.85 and support_count >= 3 and contradiction_count == 0:
            return "established"
        if score >= 0.65 and support_count >= 2:
            return "supported"
        if score >= 0.40:
            return "provisional"
        return "candidate"