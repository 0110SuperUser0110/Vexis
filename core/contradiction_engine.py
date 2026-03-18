from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ContradictionCandidate:
    left_statement: str
    right_statement: str
    contradiction_score: float
    contradiction_type: str
    should_flag: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_statement": self.left_statement,
            "right_statement": self.right_statement,
            "contradiction_score": self.contradiction_score,
            "contradiction_type": self.contradiction_type,
            "should_flag": self.should_flag,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class ContradictionBatchResult:
    candidates: list[ContradictionCandidate] = field(default_factory=list)
    flagged_count: int = 0
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "flagged_count": self.flagged_count,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class ContradictionEngine:
    """
    Deterministic contradiction detector for VEX.

    This is intentionally conservative.
    It flags likely contradictions for review rather than pretending perfect semantic understanding.
    """

    def __init__(self) -> None:
        self.negation_terms = {
            "not", "no", "never", "none", "cannot", "can't", "doesn't", "dont", "don't",
            "isn't", "isnt", "wasn't", "wasnt", "without",
        }

        self.antonym_pairs = [
            ("increase", "decrease"),
            ("increased", "decreased"),
            ("higher", "lower"),
            ("more", "less"),
            ("positive", "negative"),
            ("true", "false"),
            ("present", "absent"),
            ("enabled", "disabled"),
            ("active", "inactive"),
            ("supports", "contradicts"),
        ]

    def compare(self, left_statement: str, right_statement: str) -> ContradictionCandidate:
        left = self._normalize(left_statement)
        right = self._normalize(right_statement)

        reasons: list[str] = []
        score = 0.0
        contradiction_type = "none"

        if not left or not right:
            return ContradictionCandidate(
                left_statement=left_statement,
                right_statement=right_statement,
                contradiction_score=0.0,
                contradiction_type="none",
                should_flag=False,
                reasons=["empty_statement"],
            )

        overlap = self._token_overlap(left, right)
        if overlap >= 2:
            score += 0.20
            reasons.append(f"shared_topic_overlap:{overlap}")

        if self._negation_conflict(left, right):
            score += 0.40
            contradiction_type = "negation_conflict"
            reasons.append("negation_conflict_detected")

        antonym_hit = self._antonym_conflict(left, right)
        if antonym_hit is not None:
            score += 0.38
            contradiction_type = "antonym_conflict"
            reasons.append(f"antonym_conflict:{antonym_hit[0]}::{antonym_hit[1]}")

        subject_match = self._shared_subject_frame(left, right)
        if subject_match:
            score += 0.18
            reasons.append("shared_subject_frame_detected")

        if left == right:
            score = 0.0
            contradiction_type = "none"
            reasons.append("identical_statement")

        score = min(score, 0.98)
        should_flag = score >= 0.55

        if should_flag and contradiction_type == "none":
            contradiction_type = "possible_conflict"

        return ContradictionCandidate(
            left_statement=left_statement,
            right_statement=right_statement,
            contradiction_score=round(score, 3),
            contradiction_type=contradiction_type,
            should_flag=should_flag,
            reasons=reasons,
            metadata={
                "left_normalized": left,
                "right_normalized": right,
            },
        )

    def compare_many(self, statements: list[str]) -> ContradictionBatchResult:
        candidates: list[ContradictionCandidate] = []
        notes: list[str] = []

        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                candidate = self.compare(statements[i], statements[j])
                if candidate.should_flag:
                    candidates.append(candidate)

        if candidates:
            notes.append(f"flagged {len(candidates)} contradiction candidates")

        return ContradictionBatchResult(
            candidates=candidates,
            flagged_count=len(candidates),
            notes=notes,
            metadata={
                "statement_count": len(statements),
            },
        )

    def _normalize(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _token_overlap(self, left: str, right: str) -> int:
        left_tokens = {token for token in re.findall(r"[a-z0-9']+", left) if len(token) > 3}
        right_tokens = {token for token in re.findall(r"[a-z0-9']+", right) if len(token) > 3}
        return len(left_tokens.intersection(right_tokens))

    def _negation_conflict(self, left: str, right: str) -> bool:
        left_has_negation = any(term in left.split() for term in self.negation_terms)
        right_has_negation = any(term in right.split() for term in self.negation_terms)

        if left_has_negation == right_has_negation:
            return False

        overlap = self._token_overlap(left, right)
        return overlap >= 2

    def _antonym_conflict(self, left: str, right: str) -> Optional[tuple[str, str]]:
        for a, b in self.antonym_pairs:
            if a in left and b in right:
                return (a, b)
            if b in left and a in right:
                return (b, a)
        return None

    def _shared_subject_frame(self, left: str, right: str) -> bool:
        left_tokens = [token for token in re.findall(r"[a-z0-9']+", left) if len(token) > 3]
        right_tokens = [token for token in re.findall(r"[a-z0-9']+", right) if len(token) > 3]

        if len(left_tokens) < 2 or len(right_tokens) < 2:
            return False

        left_subject = left_tokens[:3]
        right_subject = right_tokens[:3]

        return len(set(left_subject).intersection(set(right_subject))) >= 1