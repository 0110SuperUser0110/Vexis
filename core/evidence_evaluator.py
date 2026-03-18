from __future__ import annotations

from core.schemas import Evidence, EvidenceStrength


class EvidenceEvaluator:
    """
    Simple prototype evidence scoring.
    Later this will become much more rigorous.
    """

    def evaluate(self, evidence: Evidence) -> Evidence:
        source_type = evidence.provenance.source_type.lower()

        if source_type in {"peer_reviewed_journal", "primary_study", "dataset"}:
            evidence.rigor_score = 0.9
            evidence.strength = EvidenceStrength.STRONG
        elif source_type in {"news_article", "book", "secondary_review"}:
            evidence.rigor_score = 0.6
            evidence.strength = EvidenceStrength.MODERATE
        elif source_type in {"blog", "social_post", "forum", "unknown"}:
            evidence.rigor_score = 0.2
            evidence.strength = EvidenceStrength.WEAK
        else:
            evidence.rigor_score = 0.4
            evidence.strength = EvidenceStrength.WEAK

        return evidence