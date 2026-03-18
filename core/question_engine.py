from __future__ import annotations

from uuid import uuid4

from core.schemas import Claim, Question


class QuestionEngine:
    """
    Generates epistemic questions from claims and state gaps.
    """

    def generate_for_claim(self, claim: Claim) -> Question:
        if not claim.supporting_evidence_ids:
            content = f"What evidence supports the claim: '{claim.content}'?"
        elif claim.contradicting_evidence_ids:
            content = f"What resolves the contradiction around the claim: '{claim.content}'?"
        else:
            content = f"What additional evidence would increase confidence in the claim: '{claim.content}'?"

        return Question(
            question_id=f"q_{uuid4().hex[:8]}",
            content=content,
            created_from_claim_id=claim.claim_id,
            priority=0.85,
            resolved=False,
            resolution_notes=None,
        )