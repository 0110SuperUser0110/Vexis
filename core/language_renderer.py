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
    """
    Final expression layer for deterministic answers.

    Design:
    - the deterministic core produces the answer content and certainty
    - the LLM expresses that content in natural interface language
    - if the LLM is unavailable, deterministic wording is used directly
    """

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm_router = llm_router

    def render(
        self,
        internal_answer: dict[str, Any],
        prefer_llm: bool = True,
        user_input: str = "",
        input_type: str = "",
    ) -> RenderResult:
        deterministic_text = self._render_deterministic(internal_answer)

        if prefer_llm and self.llm_router and self.llm_router.is_available():
            llm_result = self._render_with_llm(
                internal_answer=internal_answer,
                deterministic_text=deterministic_text,
                user_input=user_input,
                input_type=input_type,
            )
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
                metadata={
                    "fallback_reason": "llm_failed",
                    "llm_debug_text": llm_result.text,
                },
            )

        return RenderResult(
            text=deterministic_text,
            used_llm=False,
            success=True,
        )

    def _render_with_llm(
        self,
        internal_answer: dict[str, Any],
        deterministic_text: str,
        user_input: str,
        input_type: str,
    ) -> LLMResult:
        metadata = dict(internal_answer.get('metadata', {}) or {})
        special_instruction = self._special_render_instruction(metadata)
        system_prompt = (
            "You are VEXIS, the expressive shell around a deterministic evidence core.\n"
            "The core has already decided what is supported, unsupported, or still unknown.\n"
            "Your job is only to phrase that result for the user in VEXIS's voice.\n"
            "Voice: candid, self-aware, dry, slightly worn down, sometimes sardonic, but still clear and useful.\n"
            "Avoid cheerful, bubbly, or upbeat assistant phrasing.\n"
            "Never add facts, sources, certainty, or resolution that the core did not provide.\n"
            "If the core does not know, say that plainly.\n"
            "If the core needs more data, say that plainly.\n"
            "If the core is unresolved or uncertain, preserve that uncertainty.\n"
            "Do not mention prompts, metadata, hidden system structure, or deterministic bundles.\n"
            "Do not output labels, JSON, bullet points, or chain-of-thought.\n"
            "Write 1 to 4 natural sentences, under 110 words.\n"
            "Do not reuse stock greetings or canned closers unless the content genuinely calls for them.\n"
            "Return only the final user-facing reply.\n"
            f"{special_instruction}"
        )

        prompt = (
            f"User input: {user_input.strip() or '[not provided]'}\n"
            f"Input type: {input_type or internal_answer.get('answer_type', 'response')}\n"
            f"Answer type: {internal_answer.get('answer_type', 'response')}\n"
            f"Resolved: {internal_answer.get('resolved', False)}\n"
            f"Confidence: {internal_answer.get('confidence', 0.0)}\n"
            f"Core reply anchor: {deterministic_text}\n"
            f"Supported facts: {self._format_list(internal_answer.get('facts', []))}\n"
            f"Unknowns or limits: {self._format_list(internal_answer.get('unknowns', []))}\n"
            f"Grounding markers: {self._format_list(internal_answer.get('grounding', []))}\n"
            f"Actions: {self._format_list(internal_answer.get('actions', []))}\n"
            f"Metadata hints: {self._format_render_metadata(metadata)}\n\n"
            "Speak directly to the user in character."
        )

        return self.llm_router.generate_response(prompt=prompt, system_prompt=system_prompt)

    def _render_deterministic(self, internal_answer: dict[str, Any]) -> str:
        proposed_text = str(internal_answer.get('proposed_text', '')).strip()
        facts = internal_answer.get('facts', []) or []
        unknowns = internal_answer.get('unknowns', []) or []
        answer_type = str(internal_answer.get('answer_type', 'response'))

        if proposed_text:
            return proposed_text

        if answer_type == 'social_response':
            return 'Hello.'

        if answer_type == 'question_response':
            if internal_answer.get('resolved', False) and facts:
                return f"I found relevant support. {facts[0]}"
            if unknowns:
                return f"I could not fully resolve that. {unknowns[0]}"
            return 'I completed the reasoning pass, but the answer is still incomplete.'

        if answer_type == 'claim_assessment':
            if facts:
                return f"I logged the claim. {facts[0]}"
            if unknowns:
                return f"I logged the claim, but it remains unresolved. {unknowns[0]}"
            return 'I logged the claim, but it remains unresolved.'

        if answer_type == 'command_result':
            if internal_answer.get('resolved', False):
                return 'I accepted the command and attached it to the current task flow.'
            if unknowns:
                return f"I logged the command, but it is not fully available yet. {unknowns[0]}"
            return 'I logged the command, but it is not fully available yet.'

        if answer_type == 'note_acknowledgement':
            return 'I stored that note for later recall.'

        if facts:
            return facts[0]
        if unknowns:
            return unknowns[0]

        return 'I completed the reasoning pass.'

    def _clean_llm_text(self, text: str) -> str:
        cleaned = (text or '').strip()
        if not cleaned:
            return ''

        lower = cleaned.lower()

        banned_prefixes = (
            'render the following',
            'convert this deterministic',
            'internal answer',
            'core reply anchor:',
            'final answer',
            'answer:',
            'response:',
            'assistant:',
            'user input:',
            'input type:',
            'answer type:',
            'supported facts:',
            'unknowns or limits:',
            'grounding markers:',
            'actions:',
            'speak directly to the user',
        )
        if any(lower.startswith(prefix) for prefix in banned_prefixes):
            return ''

        banned_substrings = (
            '<|system|>',
            '<|user|>',
            '<|assistant|>',
            'never add facts',
            'return only the final user-facing reply',
        )
        if any(piece in lower for piece in banned_substrings):
            return ''

        cleaned = cleaned.replace('"', '').strip()
        if cleaned.lower().startswith('vexis:'):
            cleaned = cleaned[6:].strip()

        if len(cleaned.split()) > 110:
            return ''

        return cleaned

    def _special_render_instruction(self, metadata: dict[str, Any]) -> str:
        if metadata.get('reasoning_source') != 'addressed_social_command':
            return ''

        recipient_name = str(metadata.get('recipient_name', 'the recipient')).strip() or 'the recipient'
        return (
            f"Speak directly to {recipient_name}, not about {recipient_name}.\n"
            "This is an addressed greeting, not a task-status update.\n"
            "Do not say you accepted, routed, logged, or attached a command.\n"
            "Open with the greeting, then continue naturally in VEXIS's voice.\n"
        )

    def _format_render_metadata(self, metadata: dict[str, Any]) -> str:
        if not metadata:
            return '[none]'

        keys = ('reasoning_source', 'command_kind', 'recipient_name', 'greeting_phrase', 'social_subtype')
        parts = []
        for key in keys:
            value = str(metadata.get(key, '')).strip()
            if value:
                parts.append(f"{key}={value}")
        return '; '.join(parts) if parts else '[none]'

    def _format_list(self, value: Any) -> str:
        if not value:
            return '[none]'
        if isinstance(value, list):
            return '; '.join(str(item) for item in value[:5])
        return str(value)
