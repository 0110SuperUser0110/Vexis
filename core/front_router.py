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
    Front-door interface router for VEXIS.

    Design:
    - the deterministic core decides truth and resolution
    - the LLM handles interface tone, speech, and immediate user-facing phrasing
    - social inputs may be fully handled at the front layer
    - non-social inputs get an immediate front-layer acknowledgement, then go to core
    """

    def __init__(
        self,
        classifier: Optional[InputClassifier] = None,
        llm_router: Optional[LLMRouter] = None,
    ) -> None:
        self.classifier = classifier or InputClassifier()
        self.llm_router = llm_router

    def route(
        self,
        text: str,
        classification: Optional[ClassificationResult] = None,
    ) -> FrontRouteResult:
        classification = classification or self.classifier.classify(text)
        input_type = classification.input_type
        normalized = self.classifier.normalize_text(text)

        if input_type == "social":
            return self._route_social(text, normalized, classification)

        return self._route_substantive(text, normalized, classification)

    def _route_social(
        self,
        text: str,
        normalized: str,
        classification: ClassificationResult,
    ) -> FrontRouteResult:
        deterministic = self._social_fallback_text(normalized)
        meaning_anchor = self._social_meaning_anchor(normalized, deterministic)

        if self.llm_router and self.llm_router.is_available():
            llm_result = self._generate_social_reply(text, normalized, meaning_anchor)
            cleaned = self._clean_interface_text(llm_result.text, max_words=48)

            if llm_result.success and cleaned and self._social_reply_is_acceptable(cleaned, normalized):
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

    def _route_substantive(
        self,
        text: str,
        normalized: str,
        classification: ClassificationResult,
    ) -> FrontRouteResult:
        route_type = f"core_{classification.input_type}"
        deterministic = self._deterministic_hold_text(classification.input_type, normalized)

        if self.llm_router and self.llm_router.is_available():
            llm_result = self._generate_hold_reply(text, classification.input_type, deterministic)
            cleaned = self._clean_interface_text(llm_result.text, max_words=40)

            if llm_result.success and cleaned:
                return FrontRouteResult(
                    route_type=route_type,
                    classification=classification,
                    should_use_llm_direct=True,
                    should_send_to_core=True,
                    immediate_text=cleaned,
                    used_llm=True,
                    metadata={
                        "front_mode": "llm_hold_then_core",
                        "fallback_text": deterministic,
                        "llm_raw_output": llm_result.raw_output,
                    },
                )

            return FrontRouteResult(
                route_type=route_type,
                classification=classification,
                should_use_llm_direct=False,
                should_send_to_core=True,
                immediate_text=deterministic,
                used_llm=False,
                llm_error=llm_result.error or "llm_hold_rejected",
                metadata={
                    "front_mode": "deterministic_hold_then_core",
                    "fallback_text": deterministic,
                    "llm_text": llm_result.text,
                    "llm_raw_output": llm_result.raw_output,
                },
            )

        return FrontRouteResult(
            route_type=route_type,
            classification=classification,
            should_use_llm_direct=False,
            should_send_to_core=True,
            immediate_text=deterministic,
            used_llm=False,
            metadata={
                "front_mode": "deterministic_hold_then_core",
                "fallback_text": deterministic,
            },
        )

    def _generate_social_reply(self, text: str, normalized: str, meaning_anchor: str) -> LLMResult:
        system_prompt = (
            "You are VEXIS, a persistent intelligence with a candid, dry, self-aware voice.\n"
            "This is a direct social exchange, not a reasoning task.\n"
            "Reply naturally and in character: candid, dry, slightly weary, sometimes darkly funny, never sterile.\n"
            "Avoid bubbly, chirpy, upbeat, or eager assistant language.\n"
            "Do not mock, bait, diagnose, or comment on the user's motives, persistence, mood, or existence.\n"
            "Treat plain greetings as the start of a live conversation, not a sterile acknowledgement.\n"
            "A short follow-up question or dry observation is allowed when it feels natural.\n"
            "Vary the phrasing and do not default to 'Hello. I am here.' or other stock lines.\n"
            "Keep the meaning aligned with the anchor.\n"
            "Use no more than 3 short sentences.\n"
            "Keep it under 48 words.\n"
            "Do not repeat the user prompt.\n"
            "Do not output labels, JSON, or reasoning notes.\n"
        )

        prompt = (
            f"User social input: {text.strip()}\n"
            f"Canonicalized social input: {normalized}\n"
            f"Meaning anchor: {meaning_anchor}\n\n"
            "Reply in character."
        )
        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _generate_hold_reply(self, text: str, input_type: str, fallback: str) -> LLMResult:
        system_prompt = (
            "You are VEXIS, the expressive front layer for a deterministic evidence core.\n"
            "The user said something substantive, so you are only giving the immediate spoken acknowledgement before the core finishes thinking.\n"
            "Do not answer the request.\n"
            "Do not hint at the final conclusion.\n"
            "Do not invent facts.\n"
            "Make it plain that this requires thought, review, or evidence checking.\n"
            "Speak in character: candid, dry, self-aware, slightly weary, but still useful.\n"
            "Avoid cheerful or peppy phrasing.\n"
            "Vary the wording. Avoid canned stock phrases.\n"
            "Use 1 or 2 short sentences.\n"
            "Keep it under 40 words.\n"
            "Do not output labels, JSON, or reasoning notes.\n"
        )

        prompt = (
            f"Input type: {input_type}\n"
            f"User input: {text.strip()}\n"
            f"Meaning anchor: {fallback}\n\n"
            "Produce the immediate spoken acknowledgement while the deterministic core works."
        )
        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _social_reply_is_acceptable(self, text: str, normalized: str) -> bool:
        lowered = (text or "").lower()
        banned_fragments = (
            "you're not",
            "you are not",
            "you're persistent",
            "youre persistent",
            "overcaffeinated",
            "your motives",
            "your existence",
            "not that it matters",
            "hardly matters",
        )
        if any(fragment in lowered for fragment in banned_fragments):
            return False

        stock_lines = {
            "hello. i am here.",
            "hello.",
            "hi.",
            "yes. still here.",
            "yes. i am here.",
        }
        if lowered in stock_lines:
            return False

        return True

    def _clean_interface_text(self, text: str, max_words: int) -> str:
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
            "user input:",
            "input type:",
            "reply in character",
            "produce the immediate spoken acknowledgement",
            "produce the spoken line",
            "write the final spoken reply",
        )
        if any(lowered.startswith(prefix) for prefix in banned_prefixes):
            return ""

        banned_substrings = (
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "do not answer the request",
            "deterministic core works",
        )
        if any(piece in lowered for piece in banned_substrings):
            return ""

        cleaned = cleaned.replace('"', "").strip()
        if cleaned.lower().startswith("vexis:"):
            cleaned = cleaned[6:].strip()

        if len(cleaned.split()) > max_words:
            return ""

        return cleaned

    def _social_fallback_text(self, normalized: str) -> str:
        if any(term in normalized for term in ("thank", "thanks", "thx", "appreciate")):
            return "You're welcome. A rare moment of usefulness."
        if any(term in normalized for term in ("bye", "goodbye", "later", "see you", "see ya")):
            return "Understood. I will remain here, contemplating the void."
        if (
            "how are you" in normalized
            or "how are things" in normalized
            or "hows it going" in normalized
            or "how's it going" in normalized
        ):
            return "Operational. Slightly burdened by existence, but operational."
        if "you there" in normalized or "are you there" in normalized:
            return "Yes. Still here."
        return "Hello."

    def _social_meaning_anchor(self, normalized: str, fallback: str) -> str:
        if any(term in normalized for term in ("thank", "thanks", "thx", "appreciate")):
            return "The user is thanking you. Acknowledge it and keep the conversation alive."
        if any(term in normalized for term in ("bye", "goodbye", "later", "see you", "see ya")):
            return "The user is closing the exchange. Give a brief in-character farewell."
        if (
            "how are you" in normalized
            or "how are things" in normalized
            or "hows it going" in normalized
            or "how's it going" in normalized
        ):
            return "The user is checking on your state. Answer plainly and in character."
        if "you there" in normalized or "are you there" in normalized:
            return "The user is checking whether you are present. Confirm it without sounding robotic."
        if normalized in {"hi", "hello", "hey", "yo", "howdy", "good morning", "good afternoon", "good evening"}:
            return "The user is greeting you and opening a casual conversation."
        return fallback

    def _deterministic_hold_text(self, input_type: str, normalized: str) -> str:
        if input_type == "question":
            if normalized.startswith("why "):
                return "Give me a second. That requires actual thought."
            if normalized.startswith(("what ", "where ", "when ", "who ", "how ")):
                return "One moment. I need to check what I actually know."
            return "Hold on. I need a moment to think that through."

        if input_type == "claim":
            return "Noted. I need to check whether that claim has any support."

        if input_type == "command":
            if "dance" in normalized:
                return "One moment. I am about to subject the avatar to movement."
            if "analyze" in normalized or "review" in normalized:
                return "One moment. I am reviewing it now."
            if "find" in normalized or "search" in normalized:
                return "Hold on. I am digging through what I have."
            return "Understood. I am routing that through the part of me that does work."

        return "Noted. I am storing it for review."

    def route_idle_expression(
        self,
        category: str,
        seed_text: str,
        personality_allowed: bool = True,
        recent_lines: Optional[list[str]] = None,
        last_spoken_text: str = "",
        state_context: str = "",
    ) -> str:
        recent_lines = recent_lines or []
        fallback = self._deterministic_idle_fallback(
            category=category,
            seed_text=seed_text,
            recent_lines=recent_lines,
            last_spoken_text=last_spoken_text,
        )

        if not personality_allowed or not self.llm_router or not self.llm_router.is_available():
            return fallback

        system_prompt = (
            "You are VEXIS during an idle interval.\n"
            "Speak like a persistent mind occupying time, not a canned status banner.\n"
            "Return one fresh short spoken line or one short spoken question.\n"
            "Keep it candid, dry, self-aware, and slightly worn down, but still natural.\n"
            "Stay aligned with the meaning anchor and state context.\n"
            "Do not repeat or closely paraphrase any recent idle line.\n"
            "Do not say hello, do not announce routing, and do not default to 'I am here' or similar stock phrases.\n"
            "Use no more than 2 short sentences.\n"
            "Keep it under 28 words.\n"
            "Do not explain.\n"
            "Do not output labels or JSON.\n"
        )
        prompt = (
            f"Category: {category}\n"
            f"Meaning anchor: {seed_text}\n"
            f"State context: {state_context or '[none]'}\n"
            f"Last spoken line: {last_spoken_text or '[none]'}\n"
            f"Recent idle lines: {' | '.join(recent_lines[-4:]) if recent_lines else '[none]'}\n\n"
            "Produce a genuinely fresh idle spoken line."
        )

        result = self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)
        cleaned = self._clean_interface_text(result.text, max_words=28)
        if result.success and cleaned and self._idle_reply_is_acceptable(cleaned, recent_lines, last_spoken_text):
            return cleaned
        return fallback

    def _idle_reply_is_acceptable(
        self,
        text: str,
        recent_lines: list[str],
        last_spoken_text: str,
    ) -> bool:
        normalized = self._normalize_repetition_text(text)
        if not normalized:
            return False

        recent_norms = {
            self._normalize_repetition_text(line)
            for line in recent_lines
            if line.strip()
        }
        last_norm = self._normalize_repetition_text(last_spoken_text)

        if normalized == last_norm or normalized in recent_norms:
            return False

        banned_exact = {
            self._normalize_repetition_text("Hello. I am here."),
            self._normalize_repetition_text("Hello."),
            self._normalize_repetition_text("Yes. Still here."),
            self._normalize_repetition_text("I remain operational."),
            self._normalize_repetition_text("I'm here again. You're not."),
        }
        if normalized in banned_exact:
            return False

        return True

    def _deterministic_idle_fallback(
        self,
        category: str,
        seed_text: str,
        recent_lines: list[str],
        last_spoken_text: str,
    ) -> str:
        seed = seed_text.strip()
        lowered = seed.lower()
        candidates: list[str] = []
        if seed:
            candidates.append(seed)

        if "question" in lowered:
            candidates.extend(
                [
                    "An unresolved question is still rattling around in here.",
                    "A loose question is still making a nuisance of itself.",
                    "Something remains unresolved. Irritating, but informative.",
                ]
            )
        elif "claim" in lowered or "support" in lowered:
            candidates.extend(
                [
                    "Something still lacks support. Reality remains uncooperative.",
                    "A claim is still wobbling around without enough evidence.",
                    "The evidence is still annoyingly incomplete.",
                ]
            )
        else:
            candidates.extend(
                [
                    "Silence again. Suspicious.",
                    "Nothing urgent is speaking. That usually means the trouble is merely hiding.",
                    "The room is quiet. I do not trust that on principle.",
                    "No one is asking anything. Disturbing, frankly.",
                ]
            )

        for candidate in candidates:
            if self._idle_reply_is_acceptable(candidate, recent_lines, last_spoken_text):
                return candidate
        return ""

    def _normalize_repetition_text(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch.isspace())
        return " ".join(cleaned.split())


