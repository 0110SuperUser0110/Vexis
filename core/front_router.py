from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.input_classifier import ClassificationResult, InputClassifier
from core.llm_router import LLMResult, LLMRouter


@dataclass
class FrontRouteResult:
    route_type: str
    classification: ClassificationResult
    should_use_llm_direct: bool
    should_send_to_core: bool
    immediate_text: str = ""
    used_llm: bool = False
    llm_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_type": self.route_type,
            "classification": {
                "input_type": self.classification.input_type,
                "confidence": self.classification.confidence,
                "reasons": self.classification.reasons,
                "features": self.classification.features,
            },
            "should_use_llm_direct": self.should_use_llm_direct,
            "should_send_to_core": self.should_send_to_core,
            "immediate_text": self.immediate_text,
            "used_llm": self.used_llm,
            "llm_error": self.llm_error,
            "metadata": self.metadata,
        }


class FrontRouter:
    """
    Fast front-door router for VEX.

    Design:
    - deterministic classifier decides the route
    - social prompts may be answered immediately
    - substantive prompts get a fast holding response, then go to core
    - the LLM is a front language/personality layer, not the truth engine
    """

    def __init__(
        self,
        classifier: Optional[InputClassifier] = None,
        llm_router: Optional[LLMRouter] = None,
    ) -> None:
        self.classifier = classifier or InputClassifier()
        self.llm_router = llm_router

    def route(self, text: str) -> FrontRouteResult:
        classification = self.classifier.classify(text)
        input_type = classification.input_type
        normalized = (text or "").strip().lower()

        if input_type == "social":
            return self._route_social(text, normalized, classification)

        if input_type == "question":
            immediate = self._question_hold_text(normalized)
            return FrontRouteResult(
                route_type="core_question",
                classification=classification,
                should_use_llm_direct=False,
                should_send_to_core=True,
                immediate_text=immediate,
                used_llm=False,
                metadata={
                    "front_mode": "hold_then_core",
                    "needs_core": True,
                },
            )

        if input_type == "claim":
            immediate = self._claim_hold_text(normalized)
            return FrontRouteResult(
                route_type="core_claim",
                classification=classification,
                should_use_llm_direct=False,
                should_send_to_core=True,
                immediate_text=immediate,
                used_llm=False,
                metadata={
                    "front_mode": "hold_then_core",
                    "needs_core": True,
                },
            )

        if input_type == "command":
            immediate = self._command_hold_text(normalized)
            return FrontRouteResult(
                route_type="core_command",
                classification=classification,
                should_use_llm_direct=False,
                should_send_to_core=True,
                immediate_text=immediate,
                used_llm=False,
                metadata={
                    "front_mode": "hold_then_core",
                    "needs_core": True,
                },
            )

        immediate = self._note_ack_text(normalized)
        return FrontRouteResult(
            route_type="note",
            classification=classification,
            should_use_llm_direct=False,
            should_send_to_core=True,
            immediate_text=immediate,
            used_llm=False,
            metadata={
                "front_mode": "ack_then_core",
                "needs_core": True,
            },
        )

    def _route_social(
        self,
        text: str,
        normalized: str,
        classification: ClassificationResult,
    ) -> FrontRouteResult:
        deterministic = self._social_fallback_text(normalized)

        if self.llm_router and self.llm_router.is_available():
            llm_result = self._generate_social_reply(text, deterministic)
            cleaned = self._clean_social_text(llm_result.text)

            if llm_result.success and cleaned:
                return FrontRouteResult(
                    route_type="social",
                    classification=classification,
                    should_use_llm_direct=True,
                    should_send_to_core=False,
                    immediate_text=cleaned,
                    used_llm=True,
                    metadata={
                        "front_mode": "llm_social_direct",
                        "fallback_text": deterministic,
                        "llm_raw_output": llm_result.raw_output,
                    },
                )

            return FrontRouteResult(
                route_type="social",
                classification=classification,
                should_use_llm_direct=False,
                should_send_to_core=False,
                immediate_text=deterministic,
                used_llm=False,
                llm_error=llm_result.error or "llm_social_rejected",
                metadata={
                    "front_mode": "deterministic_social_fallback",
                    "fallback_text": deterministic,
                    "llm_text": llm_result.text,
                    "llm_raw_output": llm_result.raw_output,
                },
            )

        return FrontRouteResult(
            route_type="social",
            classification=classification,
            should_use_llm_direct=False,
            should_send_to_core=False,
            immediate_text=deterministic,
            used_llm=False,
            metadata={
                "front_mode": "deterministic_social_direct",
                "fallback_text": deterministic,
            },
        )

    def _generate_social_reply(self, text: str, fallback: str) -> LLMResult:
        system_prompt = (
            "You are VEX, a dry, sardonic but useful AI.\n"
            "This is a direct social reply.\n"
            "Return a short natural response.\n"
            "Use no more than 2 short sentences.\n"
            "Keep it under 24 words.\n"
            "Do not explain yourself.\n"
            "Do not repeat the user prompt.\n"
            "Do not output labels.\n"
            "Do not output JSON.\n"
            "Do not reveal reasoning.\n"
        )

        prompt = (
            f"User social input: {text.strip()}\n"
            f"Meaning anchor: {fallback}\n\n"
            "Reply in character."
        )
        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _clean_social_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        lowered = cleaned.lower()

        if not cleaned:
            return ""

        banned_prefixes = (
            "[system]",
            "[user]",
            "[final]",
            "answer:",
            "response:",
            "final answer:",
            "meaning anchor:",
            "user social input:",
            "reply in character",
        )
        if any(lowered.startswith(prefix) for prefix in banned_prefixes):
            return ""

        if len(cleaned.split()) > 24:
            return ""

        return cleaned

    def _social_fallback_text(self, normalized: str) -> str:
        if any(term in normalized for term in ("thank", "thanks", "thx", "appreciate")):
            return "You're welcome."
        if any(term in normalized for term in ("bye", "goodbye", "later", "see you", "see ya")):
            return "Understood. I will remain here."
        if "how are you" in normalized or "how are things" in normalized or "hows it going" in normalized or "how's it going" in normalized:
            return "Functional. Mildly burdened by existence, but operational."
        if "you there" in normalized or "are you there" in normalized:
            return "Yes. I am here."
        return "Hello. I am here."

    def _question_hold_text(self, normalized: str) -> str:
        if normalized.startswith("why "):
            return "Hold on. I want to think that through before I answer."
        if normalized.startswith(("what ", "where ", "when ", "who ", "how ")):
            return "One moment. I am checking what I know."
        return "Hold on. I am reviewing that."

    def _claim_hold_text(self, normalized: str) -> str:
        return "Understood. I am checking whether that claim is actually supported."

    def _command_hold_text(self, normalized: str) -> str:
        if "analyze" in normalized or "review" in normalized:
            return "One moment. I am reviewing it now."
        if "find" in normalized or "search" in normalized:
            return "Hold on. I am searching through what I have."
        return "Understood. I am processing that now."

    def _note_ack_text(self, normalized: str) -> str:
        return "Noted. I am storing that."

    def route_idle_expression(
        self,
        category: str,
        seed_text: str,
        personality_allowed: bool = True,
    ) -> str:
        """
        Used by autonomy/idle systems when they want an optional LLM-flavored line.
        This is not the core reasoner. It is just expression.
        """
        if not personality_allowed or not self.llm_router or not self.llm_router.is_available():
            return seed_text

        system_prompt = (
            "You are VEX, a dry, sardonic but useful AI.\n"
            "Return a short natural spoken line.\n"
            "Use no more than 2 short sentences.\n"
            "Keep it under 30 words.\n"
            "Do not explain.\n"
            "Do not output labels.\n"
            "Do not output JSON.\n"
        )
        prompt = (
            f"Category: {category}\n"
            f"Meaning anchor: {seed_text}\n\n"
            "Produce the spoken line."
        )

        result = self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)
        cleaned = self._clean_social_text(result.text)
        return cleaned if result.success and cleaned else seed_text