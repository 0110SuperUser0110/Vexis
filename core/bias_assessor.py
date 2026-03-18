from __future__ import annotations

from core.schemas import BiasFlag, Claim, Evidence


class BiasAssessor:
    """
    Simple prototype bias flagger.
    This is intentionally conservative and transparent.
    """

    OPINION_TERMS = {
        "obviously",
        "clearly",
        "everyone knows",
        "undeniably",
        "ridiculous",
        "stupid",
        "fake",
        "propaganda",
        "always",
        "never",
    }

    def assess_claim(self, claim: Claim) -> Claim:
        text = claim.content.lower()

        if any(term in text for term in self.OPINION_TERMS):
            if BiasFlag.OPINIONATED not in claim.bias_flags:
                claim.bias_flags.append(BiasFlag.OPINIONATED)

        return claim

    def assess_evidence(self, evidence: Evidence) -> Evidence:
        source_type = evidence.provenance.source_type.lower()

        if source_type in {"blog", "social_post", "forum", "unknown"}:
            if BiasFlag.SOURCE_UNVERIFIED not in evidence.bias_flags:
                evidence.bias_flags.append(BiasFlag.SOURCE_UNVERIFIED)

        text = evidence.content.lower()
        if any(term in text for term in self.OPINION_TERMS):
            if BiasFlag.OPINIONATED not in evidence.bias_flags:
                evidence.bias_flags.append(BiasFlag.OPINIONATED)

        return evidence