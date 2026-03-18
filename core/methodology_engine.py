from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MethodologyAssessment:
    study_type: str
    rigor_score: float
    rigor_label: str
    sample_strength: str
    control_strength: str
    statistics_strength: str
    overreach_detected: bool
    limitations_acknowledged: bool
    replication_signal: str
    usable_for_belief_update: bool
    max_support_contribution: float
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_type": self.study_type,
            "rigor_score": self.rigor_score,
            "rigor_label": self.rigor_label,
            "sample_strength": self.sample_strength,
            "control_strength": self.control_strength,
            "statistics_strength": self.statistics_strength,
            "overreach_detected": self.overreach_detected,
            "limitations_acknowledged": self.limitations_acknowledged,
            "replication_signal": self.replication_signal,
            "usable_for_belief_update": self.usable_for_belief_update,
            "max_support_contribution": self.max_support_contribution,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class MethodologyEngine:
    """
    Deterministic evaluator for scientific rigor.

    This does not decide truth.
    It decides how much epistemic weight a source deserves.
    """

    def assess(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MethodologyAssessment:
        metadata = metadata or {}
        normalized = self._normalize(text)
        reasons: list[str] = []

        study_type = self._detect_study_type(normalized, metadata)
        reasons.append(f"study_type:{study_type}")

        sample_strength = self._sample_strength(normalized)
        reasons.append(f"sample_strength:{sample_strength}")

        control_strength = self._control_strength(normalized)
        reasons.append(f"control_strength:{control_strength}")

        statistics_strength = self._statistics_strength(normalized)
        reasons.append(f"statistics_strength:{statistics_strength}")

        overreach = self._detect_overreach(normalized)
        if overreach:
            reasons.append("overreach_detected")

        limitations = self._limitations_acknowledged(normalized)
        if limitations:
            reasons.append("limitations_acknowledged")

        replication_signal = self._replication_signal(normalized)
        reasons.append(f"replication_signal:{replication_signal}")

        rigor_score = self._rigor_score(
            study_type=study_type,
            sample_strength=sample_strength,
            control_strength=control_strength,
            statistics_strength=statistics_strength,
            overreach_detected=overreach,
            limitations_acknowledged=limitations,
            replication_signal=replication_signal,
            metadata=metadata,
        )

        rigor_label = self._rigor_label(rigor_score)
        usable_for_belief_update = rigor_score >= 0.18

        max_support_contribution = min(rigor_score * 0.6, 0.45)
        if overreach:
            max_support_contribution *= 0.8

        return MethodologyAssessment(
            study_type=study_type,
            rigor_score=round(rigor_score, 3),
            rigor_label=rigor_label,
            sample_strength=sample_strength,
            control_strength=control_strength,
            statistics_strength=statistics_strength,
            overreach_detected=overreach,
            limitations_acknowledged=limitations,
            replication_signal=replication_signal,
            usable_for_belief_update=usable_for_belief_update,
            max_support_contribution=round(max_support_contribution, 3),
            reasons=reasons,
            metadata={
                "source_type": metadata.get("source_type", "unknown"),
                "title": metadata.get("title"),
                "year": metadata.get("year"),
            },
        )

    def _normalize(self, text: str) -> str:
        text = text or ""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _detect_study_type(self, text: str, metadata: dict[str, Any]) -> str:
        source_type = str(metadata.get("source_type", "")).lower()

        if "meta-analysis" in text or "systematic review" in text:
            return "meta_analysis"
        if "randomized controlled trial" in text or "randomised controlled trial" in text:
            return "randomized_trial"
        if "cohort study" in text:
            return "cohort_study"
        if "case-control" in text or "case control" in text:
            return "case_control"
        if "cross-sectional" in text or "cross sectional" in text:
            return "cross_sectional"
        if "case report" in text or "case series" in text:
            return "case_report"
        if "review" in text and "systematic" not in text:
            return "review"
        if "editorial" in text or "opinion" in text or source_type == "editorial":
            return "editorial"
        if source_type == "media":
            return "media_summary"
        if source_type == "preprint":
            return "preprint"
        if source_type == "peer_reviewed_journal":
            return "journal_article"
        return "unknown"

    def _sample_strength(self, text: str) -> str:
        match = re.search(r"\b(n|sample size)\s*[=:]?\s*(\d{1,6})\b", text)
        if match:
            try:
                n = int(match.group(2))
                if n >= 1000:
                    return "strong"
                if n >= 100:
                    return "moderate"
                return "weak"
            except Exception:
                pass

        if "small sample" in text:
            return "weak"
        if "large sample" in text:
            return "strong"
        return "unknown"

    def _control_strength(self, text: str) -> str:
        if "placebo-controlled" in text or "placebo controlled" in text:
            return "strong"
        if "control group" in text or "controlled" in text:
            return "moderate"
        if "uncontrolled" in text or "no control group" in text:
            return "weak"
        return "unknown"

    def _statistics_strength(self, text: str) -> str:
        indicators = 0
        if "confidence interval" in text:
            indicators += 1
        if re.search(r"\bp\s*[<=>]\s*0?\.\d+", text):
            indicators += 1
        if "effect size" in text:
            indicators += 1
        if "regression" in text:
            indicators += 1
        if "hazard ratio" in text or "odds ratio" in text or "relative risk" in text:
            indicators += 1

        if indicators >= 3:
            return "strong"
        if indicators >= 1:
            return "moderate"
        return "weak"

    def _detect_overreach(self, text: str) -> bool:
        overreach_markers = (
            "proves that",
            "proves",
            "definitively shows",
            "settles the question",
            "conclusively demonstrates",
            "without doubt",
        )
        caution_markers = (
            "may",
            "might",
            "suggests",
            "associated with",
            "consistent with",
            "further research",
        )

        has_overreach = any(marker in text for marker in overreach_markers)
        has_caution = any(marker in text for marker in caution_markers)

        return has_overreach and not has_caution

    def _limitations_acknowledged(self, text: str) -> bool:
        limitation_markers = (
            "limitations",
            "limitation",
            "we acknowledge",
            "our findings should be interpreted cautiously",
            "further research is needed",
            "future work",
        )
        return any(marker in text for marker in limitation_markers)

    def _replication_signal(self, text: str) -> str:
        if "replicated" in text or "replication" in text or "independent replication" in text:
            return "positive"
        if "first report" in text or "novel finding" in text:
            return "unknown"
        return "unknown"

    def _rigor_score(
        self,
        study_type: str,
        sample_strength: str,
        control_strength: str,
        statistics_strength: str,
        overreach_detected: bool,
        limitations_acknowledged: bool,
        replication_signal: str,
        metadata: dict[str, Any],
    ) -> float:
        score = 0.10

        study_weights = {
            "meta_analysis": 0.32,
            "randomized_trial": 0.28,
            "cohort_study": 0.22,
            "case_control": 0.18,
            "cross_sectional": 0.14,
            "journal_article": 0.16,
            "review": 0.12,
            "case_report": 0.08,
            "preprint": 0.10,
            "editorial": 0.03,
            "media_summary": 0.02,
            "unknown": 0.05,
        }
        score += study_weights.get(study_type, 0.05)

        sample_weights = {"strong": 0.12, "moderate": 0.07, "weak": 0.02, "unknown": 0.03}
        control_weights = {"strong": 0.12, "moderate": 0.07, "weak": 0.02, "unknown": 0.03}
        stats_weights = {"strong": 0.12, "moderate": 0.07, "weak": 0.02}

        score += sample_weights.get(sample_strength, 0.03)
        score += control_weights.get(control_strength, 0.03)
        score += stats_weights.get(statistics_strength, 0.02)

        if limitations_acknowledged:
            score += 0.03

        if replication_signal == "positive":
            score += 0.08

        if overreach_detected:
            score -= 0.10

        source_type = str(metadata.get("source_type", "")).lower()
        if source_type == "media":
            score = min(score, 0.22)
        elif source_type == "peer_reviewed_journal":
            score += 0.03
        elif source_type == "preprint":
            score -= 0.03

        return max(0.01, min(score, 0.85))

    def _rigor_label(self, score: float) -> str:
        if score >= 0.70:
            return "moderate_high"
        if score >= 0.50:
            return "moderate"
        if score >= 0.30:
            return "moderate_low"
        return "low"