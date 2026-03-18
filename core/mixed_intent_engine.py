from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.input_classifier import ClassificationResult, InputClassifier


@dataclass
class IntentSegment:
    text: str
    intent_type: str
    confidence: float
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent_type": self.intent_type,
            "confidence": self.confidence,
            "role": self.role,
            "metadata": self.metadata,
        }


@dataclass
class MixedIntentResult:
    is_mixed: bool
    dominant_intent: str
    segments: list[IntentSegment] = field(default_factory=list)
    social_preface: Optional[str] = None
    substantive_segment: Optional[str] = None
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_mixed": self.is_mixed,
            "dominant_intent": self.dominant_intent,
            "segments": [segment.to_dict() for segment in self.segments],
            "social_preface": self.social_preface,
            "substantive_segment": self.substantive_segment,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class MixedIntentEngine:
    """
    Detects and separates mixed-intent prompts.

    Examples:
    - "hey can you analyze this file"
    - "thanks, but why did that fail"
    - "hello, remind me what I said about the cube"
    - "I think this is broken, can you check"
    """

    def __init__(self, classifier: Optional[InputClassifier] = None) -> None:
        self.classifier = classifier or InputClassifier()

        self.clause_split_pattern = re.compile(r"\s*(?:,|;| and | but | then )\s*", re.IGNORECASE)

    def analyze(self, text: str) -> MixedIntentResult:
        raw = (text or "").strip()
        if not raw:
            return MixedIntentResult(
                is_mixed=False,
                dominant_intent="note",
                reasons=["empty_input"],
            )

        clauses = self._split_clauses(raw)
        if len(clauses) <= 1:
            classification = self.classifier.classify(raw)
            return MixedIntentResult(
                is_mixed=False,
                dominant_intent=classification.input_type,
                segments=[
                    IntentSegment(
                        text=raw,
                        intent_type=classification.input_type,
                        confidence=classification.confidence,
                        role="full_input",
                        metadata={
                            "classification_reasons": classification.reasons,
                            "features": classification.features,
                        },
                    )
                ],
                reasons=["single_clause_input"],
            )

        segments: list[IntentSegment] = []
        reasons: list[str] = []

        for index, clause in enumerate(clauses):
            classification = self.classifier.classify(clause)
            role = "substantive_candidate"

            if classification.input_type == "social" and index == 0:
                role = "social_preface"
            elif classification.input_type == "social":
                role = "social_clause"

            segments.append(
                IntentSegment(
                    text=clause,
                    intent_type=classification.input_type,
                    confidence=classification.confidence,
                    role=role,
                    metadata={
                        "classification_reasons": classification.reasons,
                        "features": classification.features,
                        "clause_index": index,
                    },
                )
            )

        social_preface = self._find_social_preface(segments)
        substantive_segment = self._find_substantive_segment(segments)
        dominant_intent = self._dominant_intent(segments)

        is_mixed = social_preface is not None and substantive_segment is not None
        if is_mixed:
            reasons.append("social_preface_plus_substantive_clause")
        elif len({segment.intent_type for segment in segments}) > 1:
            reasons.append("multi_intent_clauses_detected")
            is_mixed = True
        else:
            reasons.append("multi_clause_same_intent")

        return MixedIntentResult(
            is_mixed=is_mixed,
            dominant_intent=dominant_intent,
            segments=segments,
            social_preface=social_preface,
            substantive_segment=substantive_segment,
            reasons=reasons,
            metadata={
                "clause_count": len(clauses),
                "intent_set": sorted({segment.intent_type for segment in segments}),
            },
        )

    def rewrite_for_core(self, result: MixedIntentResult, original_text: str) -> str:
        """
        Returns the best substantive slice of the prompt for the deterministic core.
        """
        if result.substantive_segment:
            return result.substantive_segment

        substantive_types = {"question", "claim", "command", "note"}
        ranked = sorted(
            [segment for segment in result.segments if segment.intent_type in substantive_types],
            key=lambda segment: segment.confidence,
            reverse=True,
        )
        if ranked:
            return ranked[0].text

        return original_text.strip()

    def should_keep_social_tone(self, result: MixedIntentResult) -> bool:
        return result.social_preface is not None

    def _split_clauses(self, text: str) -> list[str]:
        parts = [part.strip() for part in self.clause_split_pattern.split(text) if part.strip()]
        return parts if parts else [text.strip()]

    def _find_social_preface(self, segments: list[IntentSegment]) -> Optional[str]:
        for segment in segments:
            if segment.role == "social_preface":
                return segment.text
        return None

    def _find_substantive_segment(self, segments: list[IntentSegment]) -> Optional[str]:
        substantive_priority = {"command": 4, "question": 3, "claim": 2, "note": 1}

        candidates = [
            segment
            for segment in segments
            if segment.intent_type in substantive_priority
        ]
        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda segment: (
                substantive_priority.get(segment.intent_type, 0),
                segment.confidence,
            ),
            reverse=True,
        )
        return ranked[0].text

    def _dominant_intent(self, segments: list[IntentSegment]) -> str:
        substantive_priority = {"command": 4, "question": 3, "claim": 2, "note": 1, "social": 0}
        ranked = sorted(
            segments,
            key=lambda segment: (
                substantive_priority.get(segment.intent_type, 0),
                segment.confidence,
            ),
            reverse=True,
        )
        return ranked[0].intent_type if ranked else "note"