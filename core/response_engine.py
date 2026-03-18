from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.context_builder import ContextBundle
from core.front_router import FrontRouteResult
from core.input_classifier import ClassificationResult
from core.language_renderer import LanguageRenderer
from core.mixed_intent_engine import MixedIntentResult
from core.reasoning_engine import InternalAnswer, ReasoningEngine
from core.schemas import InputRecord, MemoryRecord, TaskRecord, VexisState


@dataclass
class ResponseBundle:
    response_text: str
    internal_answer: dict
    memory_record: Optional[MemoryRecord] = None
    task_record: Optional[TaskRecord] = None
    status_text: str = "ready"
    visual_state: str = "idle"
    epistemic_updates: dict[str, list[str]] = field(default_factory=dict)
    used_llm: bool = False
    render_error: Optional[str] = None
    immediate_front_text: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class ResponseEngine:
    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        language_renderer: Optional[LanguageRenderer] = None,
    ) -> None:
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        self.language_renderer = language_renderer or LanguageRenderer()

    def generate(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        memory_record: Optional[MemoryRecord] = None,
        task_record: Optional[TaskRecord] = None,
        context: Optional[ContextBundle] = None,
        prefer_llm: bool = True,
        front_route: Optional[FrontRouteResult] = None,
        mixed_intent: Optional[MixedIntentResult] = None,
        belief_records: Optional[list] = None,
    ) -> ResponseBundle:
        internal_answer = self.reasoning_engine.reason(
            state=state,
            input_record=input_record,
            classification=classification,
            context=context,
            task_record=task_record,
            belief_records=belief_records or [],
            mixed_intent=mixed_intent.to_dict() if mixed_intent else None,
        )

        llm_allowed = (
            prefer_llm
            and (
                internal_answer.answer_type == "social_response"
                or (
                    internal_answer.answer_type == "question_response"
                    and internal_answer.resolved
                    and len(internal_answer.facts) > 0
                )
            )
        )

        render_result = self.language_renderer.render(
            internal_answer=internal_answer.to_dict(),
            prefer_llm=llm_allowed,
        )

        internal_answer_dict = internal_answer.to_dict()
        internal_answer_dict.setdefault("metadata", {})
        internal_answer_dict["metadata"].update(render_result.metadata)

        status_text = self._derive_status_text(classification, internal_answer)
        visual_state = "speaking"
        epistemic_updates = self._derive_epistemic_updates(
            input_record=input_record,
            classification=classification,
            internal_answer=internal_answer,
        )

        return ResponseBundle(
            response_text=render_result.text,
            internal_answer=internal_answer_dict,
            memory_record=memory_record,
            task_record=task_record,
            status_text=status_text,
            visual_state=visual_state,
            epistemic_updates=epistemic_updates,
            used_llm=render_result.used_llm,
            render_error=render_result.error,
            immediate_front_text=front_route.immediate_text if front_route else "",
            metadata={
                "front_route": front_route.to_dict() if front_route else None,
                "mixed_intent": mixed_intent.to_dict() if mixed_intent else None,
            },
        )

    def _derive_status_text(
        self,
        classification: ClassificationResult,
        internal_answer: InternalAnswer,
    ) -> str:
        input_type = classification.input_type

        if input_type == "social":
            return "social response ready"
        if input_type == "question":
            return "question resolved" if internal_answer.resolved else "question logged"
        if input_type == "claim":
            return "claim assessed" if internal_answer.resolved else "claim logged"
        if input_type == "command":
            return "command resolved" if internal_answer.resolved else "command pending"
        if input_type == "note":
            return "note stored"

        return "ready"

    def _derive_epistemic_updates(
        self,
        input_record: InputRecord,
        classification: ClassificationResult,
        internal_answer: InternalAnswer,
    ) -> dict[str, list[str]]:
        updates: dict[str, list[str]] = {}

        if classification.input_type == "question" and not internal_answer.resolved:
            updates["open_questions"] = [input_record.raw_text]

        if classification.input_type == "claim":
            updates["open_claims"] = [input_record.raw_text]
            if not internal_answer.resolved:
                updates["unsupported_claims"] = [input_record.raw_text]

        if classification.input_type == "note" and "contradiction_engine" in internal_answer.grounding:
            updates["contradictions"] = [input_record.raw_text]

        return updates