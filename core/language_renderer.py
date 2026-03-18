from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.llm_router import LLMResult, LLMRouter


@dataclass
class RenderResult:
    text: str
    used_llm: bool
    success: bool
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LanguageRenderer:
    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm_router = llm_router

    def render(self, internal_answer: dict[str, Any], prefer_llm: bool = True) -> RenderResult:
        deterministic_text = self._render_deterministic(internal_answer)
        answer_type = str(internal_answer.get("answer_type", ""))

        if prefer_llm and self.llm_router and self.llm_router.is_available():
            if answer_type == "social_response":
                llm_result = self._render_social_with_llm(internal_answer, deterministic_text)
            else:
                llm_result = self._render_with_llm(internal_answer)

            cleaned = self._clean_llm_text(llm_result.text)

            if llm_result.success and cleaned:
                return RenderResult(
                    text=cleaned,
                    used_llm=True,
                    success=True,
                    metadata={"model": llm_result.model_name},
                )

            return RenderResult(
                text=deterministic_text,
                used_llm=False,
                success=True,
                error=llm_result.error or "llm_output_rejected",
                metadata={"fallback_reason": "llm_failed"},
            )

        return RenderResult(
            text=deterministic_text,
            used_llm=False,
            success=True,
        )

    def _render_social_with_llm(self, internal_answer: dict[str, Any], deterministic_text: str) -> LLMResult:
        social_subtype = str(internal_answer.get("metadata", {}).get("social_subtype", "social"))
        system_prompt = (
            "You are VEX, a dry, sardonic but useful AI.\n"
            "This is a SOCIAL reply only.\n"
            "Return a short natural social reply.\n"
            "Use no more than 3 short sentences.\n"
            "Keep it under 60 words.\n"
            "Do not explain.\n"
            "Do not restate the prompt.\n"
            "Do not output labels.\n"
            "Do not output JSON.\n"
            "Do not output 'answer', 'final answer', or 'response'.\n"
            "Do not reveal reasoning.\n"
        )

        prompt = (
            f"Social subtype: {social_subtype}\n"
            f"Base reply meaning: {deterministic_text}\n\n"
            "Respond now in character."
        )
        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _render_with_llm(self, internal_answer: dict[str, Any]) -> LLMResult:
        system_prompt = (
            "You are the VEXIS language rendering layer.\n"
            "You are not the reasoner.\n"
            "Do not add facts.\n"
            "Do not change certainty.\n"
            "Do not reveal chain-of-thought.\n"
            "Do not restate the prompt.\n"
            "Return one clean final spoken answer only.\n"
            "Keep it short and natural.\n"
        )

        prompt = (
            "Convert this internal answer into one short natural spoken reply.\n\n"
            f"{self._format_internal_answer(internal_answer)}"
        )

        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _render_deterministic(self, internal_answer: dict[str, Any]) -> str:
        proposed_text = str(internal_answer.get("proposed_text", "")).strip()
        facts = internal_answer.get("facts", []) or []
        unknowns = internal_answer.get("unknowns", []) or []
        answer_type = str(internal_answer.get("answer_type", "response"))

        if proposed_text:
            return proposed_text

        if answer_type == "social_response":
            return "Hello."

        if answer_type == "question_response":
            if internal_answer.get("resolved", False) and facts:
                return f"I found relevant support. {facts[0]}"
            if unknowns:
                return f"I could not fully resolve that. {unknowns[0]}"
            return "I completed the reasoning pass, but the answer is still incomplete."

        if answer_type == "claim_assessment":
            if facts:
                return f"I logged the claim. {facts[0]}"
            if unknowns:
                return f"I logged the claim, but it remains unresolved. {unknowns[0]}"
            return "I logged the claim, but it remains unresolved."

        if answer_type == "command_result":
            if internal_answer.get("resolved", False):
                return "I accepted the command and attached it to the current task flow."
            if unknowns:
                return f"I logged the command, but it is not fully available yet. {unknowns[0]}"
            return "I logged the command, but it is not fully available yet."

        if answer_type == "note_acknowledgement":
            return "I stored that note for later recall."

        if facts:
            return facts[0]
        if unknowns:
            return unknowns[0]

        return "I completed the reasoning pass."

    def _clean_llm_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        lower = cleaned.lower()

        banned_prefixes = (
            "render the following",
            "convert this internal",
            "internal answer",
            "final answer",
            "answer:",
            "response:",
            "assistant:",
            "social subtype:",
            "base reply meaning:",
        )
        if any(lower.startswith(prefix) for prefix in banned_prefixes):
            return ""

        banned_substrings = (
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "structured internal",
            "respond now in character",
            "respond now with",
        )
        if any(piece in lower for piece in banned_substrings):
            return ""

        cleaned = cleaned.replace('"', "").strip()

        if len(cleaned.split()) > 60:
            return ""

        return cleaned

    def _format_internal_answer(self, internal_answer: dict[str, Any]) -> str:
        lines = []
        for key, value in internal_answer.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)