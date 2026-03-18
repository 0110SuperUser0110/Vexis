from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassificationResult:
    input_type: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


class InputClassifier:
    """
    Weighted intent classifier for VEXIS.

    Categories:
    - social
    - question
    - claim
    - command
    - note

    Design:
    - no category is decided by a single exact trigger alone
    - multiple linguistic signals contribute to each score
    - exact phrases help, but do not control the whole classifier
    """

    def __init__(self) -> None:
        self.greeting_terms = {
            "hi", "hello", "hey", "heya", "yo", "howdy", "hiya",
            "morning", "afternoon", "evening",
        }
        self.gratitude_terms = {
            "thanks", "thank", "thx", "appreciate", "appreciated",
        }
        self.farewell_terms = {
            "bye", "goodbye", "later", "farewell",
        }
        self.social_openers = {
            "well", "so", "anyway", "alright", "okay", "ok",
        }
        self.question_starters = {
            "who", "what", "when", "where", "why", "how",
            "can", "could", "would", "will", "do", "does", "did", "is", "are",
        }
        self.command_verbs = {
            "open", "analyze", "review", "check", "compare", "summarize", "find",
            "search", "show", "write", "draft", "fix", "update", "remove", "add",
            "save", "load", "remember", "recall", "read", "ingest", "scan",
            "create", "generate", "list", "explain", "tell",
        }
        self.claim_markers = {
            "i think", "i believe", "i suspect", "i feel", "it seems", "it appears",
            "apparently", "probably", "likely",
        }
        self.state_verbs = {"is", "are", "was", "were", "seems", "appears", "looks"}
        self.direct_object_markers = {
            "file", "document", "folder", "memory", "task", "report", "data",
            "question", "claim", "system", "window", "model", "note",
        }
        self.social_patterns = [
            r"\bhow are you\b",
            r"\bhow are things\b",
            r"\bhow's it going\b",
            r"\bhows it going\b",
            r"\bwhat's up\b",
            r"\bwhats up\b",
            r"\byou doing alright\b",
            r"\byou there\b",
            r"\bare you there\b",
            r"\bgood morning\b",
            r"\bgood afternoon\b",
            r"\bgood evening\b",
            r"\bgood night\b",
            r"\bsee you\b",
            r"\bsee ya\b",
            r"\bthank you\b",
            r"\bappreciate it\b",
        ]

    def classify(self, text: str) -> ClassificationResult:
        raw = text or ""
        normalized = self._normalize(raw)
        tokens = self._tokens(normalized)
        features = self._extract_features(raw, normalized, tokens)

        if not normalized:
            return ClassificationResult(
                input_type="note",
                confidence=0.10,
                reasons=["empty_input_fallback"],
                features=features,
            )

        scores = {
            "social": self._score_social(normalized, tokens, features),
            "question": self._score_question(normalized, tokens, features),
            "command": self._score_command(normalized, tokens, features),
            "claim": self._score_claim(normalized, tokens, features),
            "note": self._score_note(normalized, tokens, features),
        }

        input_type = max(scores, key=scores.get)
        confidence = self._confidence_from_scores(scores, input_type)
        reasons = self._build_reasons(input_type, normalized, tokens, features, scores)

        features["scores"] = {k: round(v, 3) for k, v in scores.items()}

        return ClassificationResult(
            input_type=input_type,
            confidence=round(confidence, 2),
            reasons=reasons,
            features=features,
        )

    def _normalize(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokens(self, normalized: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", normalized)

    def _extract_features(self, raw: str, normalized: str, tokens: list[str]) -> dict[str, Any]:
        first = tokens[0] if tokens else ""
        return {
            "char_count": len(raw),
            "word_count": len(tokens),
            "token_count": len(tokens),
            "first_token": first,
            "ends_with_question": normalized.endswith("?"),
            "contains_question_mark": "?" in raw,
            "contains_exclamation": "!" in raw,
            "starts_with_question_word": first in self.question_starters,
            "starts_with_command_verb": first in self.command_verbs,
            "contains_first_person": bool(re.search(r"\b(i|i'm|im|my|me)\b", normalized)),
            "contains_second_person": bool(re.search(r"\b(you|your|you're|youre)\b", normalized)),
            "contains_social_pattern": any(re.search(pattern, normalized) for pattern in self.social_patterns),
            "contains_command_verb": any(token in self.command_verbs for token in tokens),
            "contains_direct_object_marker": any(token in self.direct_object_markers for token in tokens),
            "contains_claim_marker": any(marker in normalized for marker in self.claim_markers),
            "contains_state_verb": any(token in self.state_verbs for token in tokens),
            "contains_gratitude": any(token in self.gratitude_terms for token in tokens) or "thank you" in normalized,
            "contains_greeting": any(token in self.greeting_terms for token in tokens),
            "contains_farewell": any(token in self.farewell_terms for token in tokens) or "see you" in normalized or "see ya" in normalized,
            "contains_politeness": bool(re.search(r"\b(please|thanks|thank you|appreciate)\b", normalized)),
            "starts_with_social_opener": first in self.social_openers,
        }

    def _score_social(self, normalized: str, tokens: list[str], features: dict[str, Any]) -> float:
        score = 0.06

        if features["contains_social_pattern"]:
            score += 0.45

        if features["contains_greeting"]:
            score += 0.24

        if features["contains_gratitude"]:
            score += 0.28

        if features["contains_farewell"]:
            score += 0.26

        if features["starts_with_social_opener"] and features["contains_second_person"]:
            score += 0.10

        if features["contains_second_person"]:
            score += 0.08

        if features["word_count"] <= 6:
            score += 0.10

        if features["word_count"] <= 10 and not features["contains_direct_object_marker"]:
            score += 0.05

        if not features["contains_command_verb"]:
            score += 0.06

        if not features["contains_claim_marker"]:
            score += 0.04

        if features["starts_with_question_word"] and features["contains_social_pattern"]:
            score += 0.10

        if features["contains_question_mark"] and features["contains_social_pattern"]:
            score += 0.08

        if features["contains_direct_object_marker"]:
            score -= 0.16

        if features["contains_command_verb"] and features["contains_direct_object_marker"]:
            score -= 0.18

        if features["contains_claim_marker"]:
            score -= 0.10

        return max(0.0, min(score, 0.98))

    def _score_question(self, normalized: str, tokens: list[str], features: dict[str, Any]) -> float:
        score = 0.08

        if features["ends_with_question"]:
            score += 0.32

        if features["contains_question_mark"]:
            score += 0.12

        if features["starts_with_question_word"]:
            score += 0.24

        if normalized.startswith(("can you ", "could you ", "would you ", "do you know ")):
            score += 0.16

        if features["contains_social_pattern"]:
            score += 0.06

        if features["contains_direct_object_marker"]:
            score += 0.06

        if features["contains_command_verb"] and not features["ends_with_question"]:
            score -= 0.10

        return max(0.0, min(score, 0.97))

    def _score_command(self, normalized: str, tokens: list[str], features: dict[str, Any]) -> float:
        score = 0.07

        if features["starts_with_command_verb"]:
            score += 0.30

        if normalized.startswith(("please ", "can you ", "could you ", "would you ")):
            score += 0.12

        if features["contains_command_verb"]:
            score += 0.16

        if features["contains_direct_object_marker"]:
            score += 0.12

        if features["contains_question_mark"] and normalized.startswith(("can you ", "could you ", "would you ")):
            score += 0.10

        if features["contains_greeting"] and features["contains_command_verb"]:
            score += 0.06

        if features["contains_claim_marker"]:
            score -= 0.08

        return max(0.0, min(score, 0.95))

    def _score_claim(self, normalized: str, tokens: list[str], features: dict[str, Any]) -> float:
        score = 0.06

        if features["contains_claim_marker"]:
            score += 0.28

        if features["contains_first_person"] and features["contains_state_verb"]:
            score += 0.16

        if not features["contains_question_mark"] and features["contains_state_verb"]:
            score += 0.08

        if normalized.startswith(("the system is ", "the file is ", "it is ", "it's ", "there is ", "there are ")):
            score += 0.14

        if features["contains_command_verb"]:
            score -= 0.08

        if features["contains_social_pattern"]:
            score -= 0.14

        return max(0.0, min(score, 0.92))

    def _score_note(self, normalized: str, tokens: list[str], features: dict[str, Any]) -> float:
        score = 0.26

        if features["word_count"] > 4:
            score += 0.06

        if not features["contains_question_mark"]:
            score += 0.06

        if not features["contains_social_pattern"]:
            score += 0.04

        if not features["starts_with_command_verb"]:
            score += 0.04

        if not features["contains_claim_marker"]:
            score += 0.03

        if features["contains_direct_object_marker"]:
            score += 0.02

        return max(0.0, min(score, 0.78))

    def _confidence_from_scores(self, scores: dict[str, float], winner: str) -> float:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        second_score = ordered[1][1]
        margin = max(0.0, top_score - second_score)

        # Confidence comes from both absolute strength and separation from runner-up.
        confidence = (top_score * 0.72) + (margin * 0.55)
        return min(max(confidence, 0.10), 0.99)

    def _build_reasons(
        self,
        input_type: str,
        normalized: str,
        tokens: list[str],
        features: dict[str, Any],
        scores: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []

        if input_type == "social":
            if features["contains_social_pattern"]:
                reasons.append("social_pattern_detected")
            if features["contains_greeting"]:
                reasons.append("greeting_signal_detected")
            if features["contains_gratitude"]:
                reasons.append("gratitude_signal_detected")
            if features["contains_farewell"]:
                reasons.append("farewell_signal_detected")
            if features["contains_second_person"]:
                reasons.append("conversational_second_person_detected")
            if features["word_count"] <= 6:
                reasons.append("short_conversational_length")
            if not features["contains_direct_object_marker"]:
                reasons.append("low_task_density")

        elif input_type == "question":
            if features["contains_question_mark"]:
                reasons.append("question_mark_detected")
            if features["starts_with_question_word"]:
                reasons.append("interrogative_opening_detected")
            if normalized.startswith(("can you ", "could you ", "would you ", "do you know ")):
                reasons.append("question_phrase_detected")

        elif input_type == "command":
            if features["starts_with_command_verb"]:
                reasons.append("imperative_opening_detected")
            if features["contains_command_verb"]:
                reasons.append("task_verb_detected")
            if features["contains_direct_object_marker"]:
                reasons.append("task_object_detected")

        elif input_type == "claim":
            if features["contains_claim_marker"]:
                reasons.append("claim_marker_detected")
            if features["contains_first_person"]:
                reasons.append("first_person_assertion_detected")
            if features["contains_state_verb"]:
                reasons.append("state_language_detected")

        else:
            reasons.append("default_note_profile")

        winner_score = scores[input_type]
        runner_up = max(v for k, v in scores.items() if k != input_type)
        if winner_score - runner_up > 0.20:
            reasons.append("clear_margin_over_runner_up")
        elif winner_score - runner_up > 0.08:
            reasons.append("moderate_margin_over_runner_up")
        else:
            reasons.append("close_category_competition")

        return reasons