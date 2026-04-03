from __future__ import annotations

import json

from dataclasses import dataclass, field
from typing import Any, Optional

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
        belief_records: Optional[list[Any]] = None,
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

        render_result = self.language_renderer.render(
            internal_answer=internal_answer.to_dict(),
            prefer_llm=prefer_llm,
            user_input=input_record.raw_text,
            input_type=classification.input_type,
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

        if internal_answer.metadata.get("reasoning_source") == "fact_extractor":
            extracted = internal_answer.metadata.get("fact_extraction", {})
            if extracted and extracted.get("success"):
                epistemic_updates.setdefault("extracted_facts", [])
                for fact in extracted.get("facts", []):
                    subject = fact.get("subject", "")
                    relation = fact.get("relation", "")
                    value = fact.get("value", "")
                    if subject and relation and value:
                        epistemic_updates["extracted_facts"].append(
                            json.dumps(
                                {
                                    "subject": subject,
                                    "relation": relation,
                                    "value": value,
                                    "confidence": fact.get("confidence", 0.72),
                                    "fact_type": fact.get("fact_type", "fact"),
                                    "reasons": fact.get("reasons", []),
                                    "metadata": fact.get("metadata", {}),
                                },
                                ensure_ascii=False,
                            )
                        )

        if internal_answer.metadata.get("reasoning_source") == "fact_extractor_pending_verification":
            extracted = internal_answer.metadata.get("fact_extraction", {})
            if extracted and extracted.get("success"):
                epistemic_updates.setdefault("open_questions", [])
                for fact in extracted.get("facts", [])[:3]:
                    subject = fact.get("subject", "")
                    relation = fact.get("relation", "")
                    value = fact.get("value", "")
                    if subject and relation and value:
                        epistemic_updates["open_questions"].append(
                            f"What source or proof supports the proposed fact: {subject} {relation} {value}?"
                        )

        if internal_answer.metadata.get("reasoning_source") == "resolution_engine":
            resolution = internal_answer.metadata.get("resolution", {})
            if resolution.get("resolved") and resolution.get("should_store_resolution"):
                epistemic_updates.setdefault("resolved_questions", [])
                epistemic_updates["resolved_questions"].append(input_record.raw_text)

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
        source = internal_answer.metadata.get("reasoning_source")

        if input_type == "social":
            return "social response ready"

        if input_type == "question":
            if source == "resolution_engine" and internal_answer.resolved:
                return "question resolved"
            return "question resolved" if internal_answer.resolved else "question logged"

        if input_type == "claim":
            return "claim assessed" if internal_answer.resolved else "claim logged"

        if input_type == "command":
            return "command resolved" if internal_answer.resolved else "command pending"

        if input_type == "note":
            if internal_answer.metadata.get("requires_fact_confirmation"):
                return "note stored pending fact verification"
            if source == "fact_extractor":
                return "note stored with extracted facts"
            return "note stored"

        return "ready"

    def _derive_epistemic_updates(
        self,
        input_record: InputRecord,
        classification: ClassificationResult,
        internal_answer: InternalAnswer,
    ) -> dict[str, list[str]]:
        updates: dict[str, list[str]] = {}

        if classification.input_type == "question":
            if internal_answer.resolved:
                updates["resolved_questions"] = [input_record.raw_text]
            else:
                updates["open_questions"] = [input_record.raw_text]

        if classification.input_type == "claim":
            updates["open_claims"] = [input_record.raw_text]
            if not internal_answer.resolved:
                updates["unsupported_claims"] = [input_record.raw_text]

        contradiction_result = internal_answer.metadata.get("contradiction_result")
        if contradiction_result and contradiction_result.get("flagged_count", 0) > 0:
            updates["contradictions"] = [input_record.raw_text]

        if internal_answer.metadata.get("requires_fact_confirmation"):
            extraction = internal_answer.metadata.get("fact_extraction", {})
            if extraction.get("success"):
                updates.setdefault("open_questions", [])
                for fact in extraction.get("facts", [])[:3]:
                    subject = fact.get("subject", "")
                    relation = fact.get("relation", "")
                    value = fact.get("value", "")
                    if subject and relation and value:
                        updates["open_questions"].append(
                            f"Should VEXIS store as a fact candidate: {subject} {relation} {value}?"
                        )

        return updates
