from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Optional

from core.context_builder import ContextBundle
from core.contradiction_engine import ContradictionBatchResult, ContradictionEngine
from core.fact_extractor import ExtractedFact, FactExtractionResult, FactExtractor
from core.grounded_answer_engine import GroundedAnswerEngine
from core.input_classifier import ClassificationResult, InputClassifier
from core.resolution_engine import ResolutionEngine, ResolutionResult
from core.schemas import InputRecord, TaskRecord, VexisState
from core.self_model import SelfModel


@dataclass
class InternalAnswer:
    answer_type: str
    resolved: bool
    confidence: float
    facts: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    grounding: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    proposed_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_type": self.answer_type,
            "resolved": self.resolved,
            "confidence": self.confidence,
            "facts": self.facts,
            "unknowns": self.unknowns,
            "grounding": self.grounding,
            "actions": self.actions,
            "proposed_text": self.proposed_text,
            "metadata": self.metadata,
        }


class ReasoningEngine:
    """
    Deterministic reasoning layer for VEX.

    This is the cognition engine.
    The LLM should only express or stylize the result, not decide truth.
    """

    def __init__(
        self,
        self_model: Optional[SelfModel] = None,
        contradiction_engine: Optional[ContradictionEngine] = None,
        fact_extractor: Optional[FactExtractor] = None,
        resolution_engine: Optional[ResolutionEngine] = None,
    ) -> None:
        self.self_model = self_model
        self.contradiction_engine = contradiction_engine or ContradictionEngine()
        self.fact_extractor = fact_extractor or FactExtractor()
        self.grounded_answer_engine = GroundedAnswerEngine()
        self.resolution_engine = resolution_engine or ResolutionEngine(
            self_model=self.self_model,
            fact_extractor=self.fact_extractor,
            grounded_answer_engine=self.grounded_answer_engine,
        )

    def reason(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle] = None,
        task_record: Optional[TaskRecord] = None,
        belief_records: Optional[list[Any]] = None,
        mixed_intent: Optional[dict[str, Any]] = None,
    ) -> InternalAnswer:
        input_type = classification.input_type

        if input_type == "social":
            return self._reason_social(
                state=state,
                input_record=input_record,
                classification=classification,
                context=context,
                task_record=task_record,
                mixed_intent=mixed_intent,
            )

        if input_type == "question":
            return self._reason_question(
                state=state,
                input_record=input_record,
                classification=classification,
                context=context,
                task_record=task_record,
                belief_records=belief_records or [],
            )

        if input_type == "claim":
            return self._reason_claim(
                state=state,
                input_record=input_record,
                classification=classification,
                context=context,
                task_record=task_record,
                belief_records=belief_records or [],
            )

        if input_type == "command":
            return self._reason_command(
                state=state,
                input_record=input_record,
                classification=classification,
                context=context,
                task_record=task_record,
            )

        return self._reason_note(
            state=state,
            input_record=input_record,
            classification=classification,
            context=context,
            task_record=task_record,
        )

    def _reason_social(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle],
        task_record: Optional[TaskRecord],
        mixed_intent: Optional[dict[str, Any]] = None,
    ) -> InternalAnswer:
        text = input_record.raw_text.strip().lower()
        interaction = input_record.interaction_context or {}
        social_subtype = self._detect_social_subtype(text)
        proposed = self._social_base_reply(social_subtype, text)

        if mixed_intent and mixed_intent.get("is_mixed"):
            substantive = mixed_intent.get("substantive_segment")
            if substantive:
                proposed = "I caught the social preface, but there is also substantive content to review."
                return InternalAnswer(
                    answer_type="social_response",
                    resolved=True,
                    confidence=0.92,
                    facts=["Mixed-intent input detected."],
                    unknowns=[],
                    grounding=["mixed_intent:detected"],
                    actions=["preserve social tone", "send substantive segment to core"],
                    proposed_text=proposed,
                    metadata={
                        "classification": classification.input_type,
                        "social_subtype": social_subtype,
                        "personality_allowed": bool(interaction.get("personality_allowed", True)),
                        "mixed_intent": mixed_intent,
                    },
                )

        return InternalAnswer(
            answer_type="social_response",
            resolved=True,
            confidence=0.95,
            facts=["This was classified as a low-stakes social interaction."],
            unknowns=[],
            grounding=[f"social_subtype:{social_subtype}"],
            actions=["store interaction for conversational history"],
            proposed_text=proposed,
            metadata={
                "classification": classification.input_type,
                "social_subtype": social_subtype,
                "personality_allowed": bool(interaction.get("personality_allowed", True)),
            },
        )

    def _reason_question(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle],
        task_record: Optional[TaskRecord],
        belief_records: list[Any],
    ) -> InternalAnswer:
        recent_memories = context.related_memories if context else []
        resolution: ResolutionResult = self.resolution_engine.resolve_question(
            question_text=input_record.raw_text,
            state=state,
            recent_memories=recent_memories,
            belief_records=belief_records,
        )

        if resolution.resolved:
            facts = [fact.get("value", fact.get("statement", "")) for fact in resolution.supporting_facts]
            facts = [fact for fact in facts if fact]

            return InternalAnswer(
                answer_type="question_response",
                resolved=True,
                confidence=resolution.confidence,
                facts=facts,
                unknowns=[],
                grounding=[f"resolution_engine:{resolution.question_type}"],
                actions=["store resolved answer"] if resolution.should_store_resolution else [],
                proposed_text=resolution.answer_text,
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "resolution_engine",
                    "resolution": resolution.to_dict(),
                },
            )

        unresolved_resolution_payload: Optional[InternalAnswer] = None
        if resolution.follow_up_question:
            unresolved_resolution_payload = InternalAnswer(
                answer_type="question_response",
                resolved=False,
                confidence=resolution.confidence,
                facts=[],
                unknowns=[resolution.answer_text],
                grounding=[f"resolution_engine:{resolution.question_type}"],
                actions=["retain question in unresolved queue", "request missing fact"],
                proposed_text=f"{resolution.answer_text} {resolution.follow_up_question}",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "resolution_engine_unresolved",
                    "resolution": resolution.to_dict(),
                },
            )

        belief_match = self._best_belief_match(input_record.raw_text.lower(), belief_records)
        if belief_match is not None:
            statement, score, belief = belief_match
            confidence_score = float(getattr(belief, "confidence_score", 0.0))
            confidence_label = str(getattr(belief, "confidence_label", "low"))
            belief_text = self._render_belief_statement(belief)
            resolved_from_belief = confidence_score >= 0.18
            unknowns = [] if resolved_from_belief else ["The learned belief is still weak and needs stronger support."]
            actions = [] if resolved_from_belief else ["keep question open for stronger evidence"]
            proposed_text = (
                f"{belief_text} My current confidence in that is {confidence_label}."
                if resolved_from_belief
                else f"I have a weak learned belief that {belief_text.lower()} My current confidence in that is {confidence_label}."
            )
            return InternalAnswer(
                answer_type="question_response",
                resolved=resolved_from_belief,
                confidence=min(max(confidence_score, 0.08), 0.95),
                facts=[
                    f"Learned belief: {statement}",
                    f"Confidence label: {confidence_label}.",
                ],
                unknowns=unknowns,
                grounding=[
                    f"belief:{getattr(belief, 'belief_id', 'unknown')}",
                    f"belief_status:{getattr(belief, 'status', 'candidate')}",
                ],
                actions=actions,
                proposed_text=proposed_text,
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "belief_engine",
                    "belief_id": getattr(belief, "belief_id", None),
                    "belief_score": score,
                    "belief_confidence": confidence_score,
                },
            )

        grounding: list[str] = []
        facts: list[str] = []
        unknowns: list[str] = []
        actions: list[str] = []

        if context and context.related_memories:
            best = context.related_memories[0]
            best_text_lower = best.content.lower()
            shared_keywords = [
                word for word in input_record.raw_text.lower().split()
                if len(word) > 3 and word in best_text_lower
            ]

            if len(shared_keywords) >= 2:
                memory_text = self._clip(best.content, 220)
                grounding.append(f"memory:{best.memory_id}")
                facts.append(f"Relevant stored memory: {memory_text}")

                if len(context.related_memories) > 1:
                    grounding.append(f"related_memory_count:{len(context.related_memories)}")
                    facts.append(f"I found {len(context.related_memories)} related memories, with one strongest match.")

                return InternalAnswer(
                    answer_type="question_response",
                    resolved=True,
                    confidence=min(0.55 + (0.06 * min(len(context.related_memories), 3)), 0.82),
                    facts=facts,
                    unknowns=[],
                    grounding=grounding,
                    actions=[],
                    proposed_text=f"I found a relevant stored memory that answers your question: {memory_text}",
                    metadata={
                        "classification": classification.input_type,
                        "task_id": task_record.task_id if task_record else None,
                        "reasoning_source": "memory_recall",
                        "source_memory_id": best.memory_id,
                        "shared_keywords": shared_keywords,
                    },
                )

        if context and context.recent_inputs:
            recent = context.recent_inputs[0]
            recent_text = self._clip(recent.raw_text, 180)

            grounding.append(f"recent_input:{recent.input_id}")
            facts.append(f"Most relevant recent input was: {recent_text}")
            unknowns.append("No directly supporting stored memory or established belief was found.")
            actions.append("retain question in unresolved queue")

            return InternalAnswer(
                answer_type="question_response",
                resolved=False,
                confidence=0.28,
                facts=facts,
                unknowns=unknowns,
                grounding=grounding,
                actions=actions,
                proposed_text=(
                    "I found related context, but I do not yet have enough grounded support to answer confidently."
                ),
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "recent_context_only",
                },
            )

        if task_record:
            grounding.append(f"task:{task_record.task_id}")
            actions.append("keep unresolved task active")

        if state.epistemic.open_questions:
            grounding.append(f"open_question_count:{len(state.epistemic.open_questions)}")

        if unresolved_resolution_payload is not None:
            unresolved_resolution_payload.grounding.extend(
                item for item in grounding if item not in unresolved_resolution_payload.grounding
            )
            unresolved_resolution_payload.actions.extend(
                item for item in actions if item not in unresolved_resolution_payload.actions
            )
            return unresolved_resolution_payload

        unknowns.append("No relevant supporting belief or memory was found.")

        return InternalAnswer(
            answer_type="question_response",
            resolved=False,
            confidence=0.16,
            facts=[],
            unknowns=unknowns,
            grounding=grounding,
            actions=actions,
            proposed_text="I do not yet have enough grounded information to answer that directly.",
            metadata={
                "classification": classification.input_type,
                "task_id": task_record.task_id if task_record else None,
                "reasoning_source": "unresolved_question",
            },
        )

    def _reason_claim(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle],
        task_record: Optional[TaskRecord],
        belief_records: list[Any],
    ) -> InternalAnswer:
        grounding: list[str] = []
        facts: list[str] = []
        unknowns: list[str] = ["The claim is not yet verified."]
        actions: list[str] = ["mark claim as unverified"]

        combined_memories = self._combine_recent_memories(
            state.recent_memories,
            context.related_memories if context else [],
        )
        evidence_memories = self._filter_claim_evidence_memories(
            combined_memories,
            input_record.input_id,
        )
        recent_facts = self.resolution_engine.collect_recent_facts(evidence_memories)
        claim_extraction = self.fact_extractor.extract(input_record.raw_text)
        support_matches, conflict_matches = self._match_claim_facts(
            claim_extraction.facts if claim_extraction.success else [],
            recent_facts,
        )

        if support_matches and not conflict_matches:
            best_support = support_matches[0]
            support_line = f"{best_support.subject} {best_support.relation} {best_support.value}"
            if task_record:
                grounding.append(f"task:{task_record.task_id}")
            grounding.append("stored_fact_support")
            return InternalAnswer(
                answer_type="claim_assessment",
                resolved=True,
                confidence=min(max(best_support.confidence, 0.62), 0.86),
                facts=[f"Stored evidence supports the claim: {support_line}."],
                unknowns=[],
                grounding=grounding,
                actions=["mark claim as supported"],
                proposed_text=f"I have grounded support for that claim. {support_line}.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "claim_supported_by_fact",
                    "supporting_fact": best_support.to_dict(),
                },
            )

        if conflict_matches:
            best_conflict = conflict_matches[0]
            conflict_line = f"{best_conflict.subject} {best_conflict.relation} {best_conflict.value}"
            grounding.append("stored_fact_conflict")
            actions.append("review conflicting evidence")
            if task_record:
                grounding.append(f"task:{task_record.task_id}")
            return InternalAnswer(
                answer_type="claim_assessment",
                resolved=False,
                confidence=0.74,
                facts=[f"Stored evidence conflicts with the claim: {conflict_line}."],
                unknowns=["The claim conflicts with evidence already stored in memory."],
                grounding=grounding,
                actions=actions,
                proposed_text=f"I found grounded evidence against that claim. Stored memory says {conflict_line}.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "claim_conflicted_by_fact",
                    "conflicting_fact": best_conflict.to_dict(),
                },
            )

        comparison_statements = [input_record.raw_text]
        if context and context.related_memories:
            comparison_statements.extend(memory.content for memory in context.related_memories[:4])

        contradiction_result: ContradictionBatchResult = self.contradiction_engine.compare_many(comparison_statements)
        if contradiction_result.flagged_count > 0:
            facts.append(f"I found {contradiction_result.flagged_count} contradiction candidate(s) related to this claim.")
            grounding.append("contradiction_engine")
            actions.append("review contradiction candidates")
        elif context and context.related_memories:
            best = context.related_memories[0]
            grounding.append(f"memory:{best.memory_id}")
            facts.append(f"Potentially related memory exists: {self._clip(best.content, 180)}")
            actions.append("compare claim against related memory")

        if task_record:
            grounding.append(f"task:{task_record.task_id}")

        return InternalAnswer(
            answer_type="claim_assessment",
            resolved=False,
            confidence=0.28,
            facts=facts,
            unknowns=unknowns,
            grounding=grounding,
            actions=actions,
            proposed_text="I logged the claim and I am treating it as unresolved until the evidence is stronger.",
            metadata={
                "classification": classification.input_type,
                "task_id": task_record.task_id if task_record else None,
                "reasoning_source": "claim_assessment",
                "contradiction_result": contradiction_result.to_dict(),
            },
        )

    def _reason_command(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle],
        task_record: Optional[TaskRecord],
    ) -> InternalAnswer:
        text = InputClassifier().normalize_text(input_record.raw_text)
        grounding: list[str] = []
        facts: list[str] = []
        unknowns: list[str] = []
        actions: list[str] = []

        supported_keywords = (
            "analyze",
            "review",
            "check",
            "compare",
            "summarize",
            "find",
            "search",
            "open",
            "show",
            "write",
            "draft",
            "fix",
            "update",
            "remove",
            "add",
            "save",
            "load",
            "remember",
            "recall",
            "read",
            "list",
            "explain",
        )

        supported = any(word in text for word in supported_keywords)
        addressed_social = self._extract_addressed_social_command(text)

        if addressed_social is not None:
            recipient_name = addressed_social["recipient_name"]
            greeting_phrase = addressed_social["greeting_phrase"]
            greeting_display = addressed_social["greeting_display"]
            facts.append(f"The command asks for a direct greeting to {recipient_name}.")
            actions.append("deliver addressed social greeting")

            if task_record:
                grounding.append(f"task:{task_record.task_id}")

            return InternalAnswer(
                answer_type="command_result",
                resolved=True,
                confidence=0.9,
                facts=facts,
                unknowns=[],
                grounding=grounding,
                actions=actions,
                proposed_text=f"{greeting_display}, {recipient_name}.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "addressed_social_command",
                    "command_kind": "addressed_social_greeting",
                    "recipient_name": recipient_name,
                    "greeting_phrase": greeting_phrase,
                },
            )

        if re.search(r"\bdance\b", text):
            facts.append("The command targets a presence action rather than knowledge storage.")
            actions.append("trigger presence action: dance")

            if task_record:
                grounding.append(f"task:{task_record.task_id}")

            return InternalAnswer(
                answer_type="command_result",
                resolved=True,
                confidence=0.88,
                facts=facts,
                unknowns=[],
                grounding=grounding,
                actions=actions,
                proposed_text="I can do that. Starting a brief dance routine now.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "presence_action_command",
                    "presence_action": "dancing",
                    "presence_action_duration_ms": 7000,
                },
            )

        if supported:
            facts.append("The command matches a recognized capability category.")
            actions.append("route command through active task flow")

            if task_record:
                grounding.append(f"task:{task_record.task_id}")

            if ("remember" in text or "recall" in text) and context and context.related_memories:
                best = context.related_memories[0]
                grounding.append(f"memory:{best.memory_id}")
                facts.append(f"Related stored memory found: {self._clip(best.content, 180)}")
                return InternalAnswer(
                    answer_type="command_result",
                    resolved=True,
                    confidence=0.72,
                    facts=facts,
                    unknowns=[],
                    grounding=grounding,
                    actions=actions,
                    proposed_text=f"I found a relevant stored memory for that recall request: {self._clip(best.content, 180)}",
                    metadata={
                        "classification": classification.input_type,
                        "task_id": task_record.task_id if task_record else None,
                        "reasoning_source": "command_memory_recall",
                    },
                )

            informative_answer = self._resolve_informational_command(
                command_text=input_record.raw_text,
                classification=classification,
                related_memories=context.related_memories if context else [],
                task_record=task_record,
            )
            if informative_answer is not None:
                return informative_answer

            return InternalAnswer(
                answer_type="command_result",
                resolved=True,
                confidence=0.64,
                facts=facts,
                unknowns=[],
                grounding=grounding,
                actions=actions,
                proposed_text="I accepted the command and attached it to the active task flow.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "command_router",
                },
            )

        unknowns.append("That command capability is not fully implemented yet.")
        actions.append("log command as pending capability")

        if task_record:
            grounding.append(f"task:{task_record.task_id}")

        return InternalAnswer(
            answer_type="command_result",
            resolved=False,
            confidence=0.22,
            facts=[],
            unknowns=unknowns,
            grounding=grounding,
            actions=actions,
            proposed_text="I logged the command, but that capability is not fully wired yet.",
            metadata={
                "classification": classification.input_type,
                "task_id": task_record.task_id if task_record else None,
                "reasoning_source": "unsupported_command",
            },
        )

    def _reason_note(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle],
        task_record: Optional[TaskRecord],
    ) -> InternalAnswer:
        extraction: FactExtractionResult = self.fact_extractor.extract(input_record.raw_text)
        source = str(input_record.source or "")
        user_assertion = source.startswith("gui")

        if extraction.success and extraction.facts:
            fact_lines = [
                f"{fact.subject} {fact.relation} {fact.value}"
                for fact in extraction.facts
            ]

            if user_assertion:
                return InternalAnswer(
                    answer_type="note_acknowledgement",
                    resolved=False,
                    confidence=0.54,
                    facts=[],
                    unknowns=["No supporting proof was attached for the extracted claim or fact candidate."],
                    grounding=["fact_extractor", "user_assertion"],
                    actions=[
                        "ask whether to store as fact candidate",
                        "request supporting proof before fact storage",
                    ],
                    proposed_text=(
                        "I can treat that as a fact candidate, but not as a fact yet. "
                        "Do you want it stored, and what proof or source supports it?"
                    ),
                    metadata={
                        "classification": classification.input_type,
                        "reasoning_source": "fact_extractor_pending_verification",
                        "fact_extraction": extraction.to_dict(),
                        "fact_lines": fact_lines,
                        "requires_fact_confirmation": True,
                        "requires_fact_proof": True,
                    },
                )

            return InternalAnswer(
                answer_type="note_acknowledgement",
                resolved=True,
                confidence=0.92,
                facts=["Structured fact(s) extracted from note."],
                unknowns=[],
                grounding=["fact_extractor"],
                actions=["store extracted facts for later resolution"],
                proposed_text="I stored that and extracted structured facts from it.",
                metadata={
                    "classification": classification.input_type,
                    "reasoning_source": "fact_extractor",
                    "fact_extraction": extraction.to_dict(),
                    "fact_lines": fact_lines,
                },
            )

        return InternalAnswer(
            answer_type="note_acknowledgement",
            resolved=True,
            confidence=0.95,
            facts=["The note has been stored in memory."],
            unknowns=[],
            grounding=[],
            actions=["retain note for future recall"],
            proposed_text="I stored that note for later review.",
            metadata={
                "classification": classification.input_type,
                "reasoning_source": "note_storage",
            },
        )

    def _best_belief_match(
        self,
        question_text: str,
        belief_records: list[Any],
    ) -> Optional[tuple[str, float, Any]]:
        best_match: Optional[tuple[str, float, Any]] = None
        question_terms = self._semantic_terms(question_text)
        subject_hint = self._question_subject_hint(question_text)
        hint_terms = self._semantic_terms(subject_hint)
        definitional_request = question_text.startswith(("what is", "define", "explain", "tell me about"))

        for belief in belief_records:
            statement = getattr(belief, "statement", "")
            confidence_score = float(getattr(belief, "confidence_score", 0.0))
            metadata = dict(getattr(belief, "metadata", {}) or {})
            subject = str(metadata.get("subject", "")).strip().lower()
            relation = str(metadata.get("relation", "")).strip().lower()
            value = str(metadata.get("value", "")).strip().lower()

            belief_terms = self._semantic_terms(statement)
            belief_terms.update(self._semantic_terms(subject))
            belief_terms.update(self._semantic_terms(value))
            overlap = len(question_terms.intersection(belief_terms))

            if overlap == 0 and not (hint_terms and hint_terms.intersection(self._semantic_terms(subject))):
                continue

            score = overlap + confidence_score
            if hint_terms and hint_terms.intersection(self._semantic_terms(subject)):
                score += 2.5
            if definitional_request and relation == "defined_as":
                score += 1.5
            if subject and subject in question_text:
                score += 1.0

            if best_match is None or score > best_match[1]:
                best_match = (statement, score, belief)

        return best_match

    def _render_belief_statement(self, belief: Any) -> str:
        metadata = dict(getattr(belief, "metadata", {}) or {})
        render_text = str(metadata.get("render_text", "")).strip()
        if render_text:
            return render_text

        subject = str(metadata.get("subject", "")).strip()
        relation = str(metadata.get("relation", "")).strip().lower()
        value = str(metadata.get("value", "")).strip()
        if subject and relation and value:
            if relation == "defined_as":
                return f"{subject} is {value}."
            if relation in {"equals", "equals_expression"}:
                return f"{subject} equals {value}."
            if relation == "color":
                return f"The {subject} is {value}."
            if relation == "depends_on":
                return f"{subject} depends on {value}."
            if relation == "proportional_to":
                return f"{subject} is proportional to {value}."
            if relation == "inversely_proportional_to":
                return f"{subject} is inversely proportional to {value}."
            if relation == "increases_with":
                return f"{subject} increases as {value} increases."
            if relation == "decreases_with":
                return f"{subject} decreases as {value} increases."
            if relation == "conserved_in":
                return f"{subject} is conserved in {value}."
            if relation == "holds_when":
                return f"{subject} holds when {value}."
            if relation == "cannot_exceed":
                return f"{subject} cannot exceed {value}."
            return f"{subject} {relation.replace('_', ' ')} {value}."
        return str(getattr(belief, "statement", "")).strip()

    def _question_subject_hint(self, question_text: str) -> str:
        normalized = question_text.strip().lower().rstrip("?.!")
        patterns = (
            r"what is (?:a |an |the )?(.+)",
            r"what does (.+) equal",
            r"define (.+)",
            r"explain (.+)",
            r"tell me about (.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1).strip()
        return ""

    def _semantic_terms(self, text: str) -> set[str]:
        cleaned = re.findall(r"[a-z0-9_\-]+", (text or "").lower())
        return {term for term in cleaned if len(term) > 2}

    def _resolve_informational_command(
        self,
        command_text: str,
        classification: ClassificationResult,
        related_memories: list[Any],
        task_record: Optional[TaskRecord],
    ) -> Optional[InternalAnswer]:
        if not self._is_informational_command(command_text):
            return None

        grounded = self.grounded_answer_engine.resolve(command_text, list(related_memories))
        if not grounded.resolved:
            return None

        grounding = [f"memory:{memory_id}" for memory_id in grounded.memory_ids[:3]]
        if task_record:
            grounding.append(f"task:{task_record.task_id}")

        facts = []
        if grounded.source_label:
            facts.append(f"Retrieved grounded material from {grounded.source_label}.")
        facts.extend(grounded.supporting_points[:2])

        return InternalAnswer(
            answer_type="command_result",
            resolved=True,
            confidence=grounded.confidence,
            facts=facts,
            unknowns=[],
            grounding=grounding,
            actions=[
                "retrieve grounded knowledge from memory",
                "answer informational request",
            ],
            proposed_text=grounded.answer_text,
            metadata={
                "classification": classification.input_type,
                "task_id": task_record.task_id if task_record else None,
                "reasoning_source": "command_knowledge_recall",
                "grounded_answer": grounded.to_dict(),
            },
        )

    def _is_informational_command(self, text: str) -> bool:
        normalized = text.strip().lower()
        prefixes = (
            "explain ",
            "summarize ",
            "describe ",
            "teach me ",
            "walk me through ",
            "outline ",
            "tell me about ",
            "show me what ",
            "show me how ",
            "give me an overview of ",
        )
        return normalized.startswith(prefixes)

    def _extract_command_focus(self, text: str) -> str:
        focus = text.strip()
        patterns = (
            r"^(please\s+)?explain\s+",
            r"^(please\s+)?summari[sz]e\s+",
            r"^(please\s+)?describe\s+",
            r"^(please\s+)?teach me\s+",
            r"^(please\s+)?walk me through\s+",
            r"^(please\s+)?outline\s+",
            r"^(please\s+)?tell me about\s+",
            r"^(please\s+)?show me what\s+",
            r"^(please\s+)?show me how\s+",
            r"^(please\s+)?give me an overview of\s+",
        )
        for pattern in patterns:
            focus = re.sub(pattern, "", focus, flags=re.IGNORECASE)
        focus = re.sub(r"\bplease\b", "", focus, flags=re.IGNORECASE)
        focus = re.sub(r"\s+", " ", focus)
        return focus.strip(" .?!,")

    def _rank_command_memories(
        self,
        query_text: str,
        memories: list[Any],
    ) -> list[tuple[float, Any]]:
        query_terms = self._content_terms(query_text)
        source_counts: dict[str, int] = {}

        for memory in memories:
            source_key = self._memory_source_key(memory)
            if source_key:
                source_counts[source_key] = source_counts.get(source_key, 0) + 1

        dominant_source = ""
        if source_counts:
            dominant_source = max(source_counts.items(), key=lambda item: item[1])[0]

        ranked: list[tuple[float, Any]] = []
        for memory in memories:
            cleaned = self._clean_memory_text(getattr(memory, "content", ""))
            if not cleaned:
                continue

            memory_terms = self._content_terms(
                " ".join(
                    [
                        cleaned,
                        str(getattr(memory, "kind", "")),
                        str(getattr(memory, "metadata", {}).get("section_name", "")),
                    ]
                )
            )
            overlap = len(query_terms.intersection(memory_terms))
            kind_bonus = {
                "fact": 3.2,
                "knowledge_chunk": 2.5,
                "resolved_question": 1.8,
                "knowledge_source": 1.2,
            }.get(getattr(memory, "kind", ""), 1.0)
            richness = min(len(cleaned.split()) / 60.0, 1.2)
            source_bonus = 0.8 if dominant_source and self._memory_source_key(memory) == dominant_source else 0.0
            score = kind_bonus + (overlap * 3.5) + richness + source_bonus

            if self._looks_like_front_matter(cleaned):
                score -= 5.0
            if getattr(memory, "kind", "") == "knowledge_source":
                score -= 0.4

            ranked.append((score, memory))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _compose_command_memory_answer(
        self,
        query_text: str,
        ranked_memories: list[tuple[float, Any]],
    ) -> tuple[str, list[str], str]:
        query_terms = self._content_terms(query_text)
        broad_request = len(query_terms) <= 2
        sentence_candidates: list[tuple[float, str, str]] = []
        selected_normalized: set[str] = set()
        seen_memory_ids: set[str] = set()

        memory_window = ranked_memories[:6]
        if broad_request:
            content_window = [
                (score, memory)
                for score, memory in ranked_memories
                if getattr(memory, "kind", "") == "knowledge_chunk"
                and int(getattr(memory, "metadata", {}).get("chunk_index", 0)) >= 8
            ]
            if content_window:
                memory_window = content_window[:4]

        for memory_score, memory in memory_window:
            cleaned = self._clean_memory_text(getattr(memory, "content", ""))
            if not cleaned:
                continue

            for sentence in self._split_sentences(cleaned):
                score = self._score_memory_sentence(
                    sentence=sentence,
                    memory_score=memory_score,
                    query_terms=query_terms,
                )
                if score <= 0:
                    continue
                sentence_candidates.append((score, sentence, memory.memory_id))

        sentence_candidates.sort(key=lambda item: item[0], reverse=True)

        selected_sentences: list[str] = []
        for _, sentence, memory_id in sentence_candidates:
            normalized = sentence.lower()
            if normalized in selected_normalized:
                continue
            if memory_id in seen_memory_ids and len(selected_sentences) >= 2:
                continue
            selected_sentences.append(sentence)
            selected_normalized.add(normalized)
            seen_memory_ids.add(memory_id)
            if len(selected_sentences) >= 3:
                break

        source_label = self._source_label_for_memories([memory for _, memory in ranked_memories])

        if selected_sentences:
            answer_text = " ".join(selected_sentences)
            if source_label and not answer_text.lower().startswith("from "):
                if len(answer_text) > 1:
                    answer_text = f"From {source_label}, {answer_text[0].lower() + answer_text[1:]}"
                else:
                    answer_text = f"From {source_label}, {answer_text.lower()}"
            return answer_text.strip(), selected_sentences, source_label

        topics = self._extract_topics_from_memories([memory for _, memory in memory_window])
        if not topics:
            return "", [], source_label

        if source_label:
            answer_text = f"{source_label} covers {self._format_series(topics[:5])}."
        else:
            answer_text = f"The retrieved material covers {self._format_series(topics[:5])}."
        return answer_text, topics[:2], source_label

    def _score_memory_sentence(
        self,
        sentence: str,
        memory_score: float,
        query_terms: set[str],
    ) -> float:
        if len(sentence) < 40 or len(sentence) > 260:
            return -1.0
        if sentence and sentence[0].islower():
            return -1.0

        lowered = sentence.lower()
        terms = self._content_terms(sentence)
        overlap = len(query_terms.intersection(terms))
        score = min(memory_score, 6.0) * 0.28
        score += overlap * 4.0

        if any(marker in lowered for marker in ("states", "means", "develop", "assume", "prove", "focus", "covers", "concerned with")):
            score += 0.8

        if self._looks_like_front_matter(lowered):
            score -= 6.0
        if any(marker in lowered for marker in ("foreword", "acknowledg", "cover photograph", "table of contents", "contents")):
            score -= 4.0

        if lowered.count("chapter") > 2 or lowered.count("page") > 1:
            score -= 2.5

        if overlap == 0 and len(query_terms) >= 2 and "mathematics" not in query_terms and "mathmatics" not in query_terms:
            score -= 1.5

        return score

    def _extract_topics_from_memories(self, memories: list[Any]) -> list[str]:
        topic_counts: dict[str, int] = {}
        for memory in memories:
            cleaned = self._clean_memory_text(getattr(memory, "content", ""))
            for sentence in self._split_sentences(cleaned):
                if len(sentence) > 140:
                    continue
                pieces = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b", sentence)
                for piece in pieces:
                    lowered = piece.lower()
                    if lowered in {
                        "basic mathematics",
                        "serge lang",
                        "columbia university",
                        "publishing company",
                        "reading massachusetts",
                    }:
                        continue
                    topic_counts[piece] = topic_counts.get(piece, 0) + 1

            section_name = str(getattr(memory, "metadata", {}).get("section_name", "")).replace("_", " ").strip()
            if section_name and section_name not in {"full text", "full_text"}:
                topic = " ".join(word.capitalize() for word in section_name.split())
                topic_counts[topic] = topic_counts.get(topic, 0) + 2

        ordered = sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
        return [topic for topic, _ in ordered[:6]]

    def _source_label_for_memories(self, memories: list[Any]) -> str:
        for memory in memories:
            metadata = getattr(memory, "metadata", {}) or {}
            title = str(metadata.get("title", "")).strip()
            if title and not self._looks_like_filename(title):
                return title

        for memory in memories:
            cleaned = self._clean_memory_text(getattr(memory, "content", ""))
            heading_match = re.search(r"\b[A-Z]{4,}(?:\s+[A-Z]{4,})?\b", cleaned)
            if heading_match:
                return heading_match.group(0).title()
            for sentence in self._split_sentences(cleaned):
                if sentence and sentence[0].islower():
                    continue
                if 8 <= len(sentence) <= 80 and not self._looks_like_front_matter(sentence.lower()):
                    return sentence.rstrip(".")

        return ""

    def _memory_source_key(self, memory: Any) -> str:
        metadata = getattr(memory, "metadata", {}) or {}
        return str(metadata.get("source_path") or metadata.get("title") or "")

    def _content_terms(self, text: str) -> set[str]:
        stop = {
            "about",
            "again",
            "also",
            "does",
            "explain",
            "give",
            "into",
            "just",
            "more",
            "show",
            "tell",
            "that",
            "them",
            "there",
            "these",
            "this",
            "through",
            "what",
            "with",
            "would",
            "your",
        }
        terms = set(re.findall(r"[a-z0-9']{3,}", text.lower()))
        return {term for term in terms if term not in stop}

    def _clean_memory_text(self, text: str) -> str:
        cleaned = text.replace("­", "")
        cleaned = re.sub(r"\[page\s+\d+\]", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _split_sentences(self, text: str) -> list[str]:
        raw_parts = re.split(r"(?<=[.!?])\s+", text)
        parts: list[str] = []
        markers = (
            " In this part ",
            " This part ",
            " We assume ",
            " We then ",
            " The notion of ",
        )
        for raw in raw_parts:
            sentence = raw.strip(" \t\r\n.-")
            if not sentence:
                continue
            for marker in markers:
                if marker in sentence and not sentence.startswith(marker.strip()):
                    sentence = sentence[sentence.index(marker) + 1 :].strip()
                    break
            if not sentence:
                continue
            parts.append(sentence)
        return parts

    def _looks_like_front_matter(self, text: str) -> bool:
        lowered = text.lower()
        markers = (
            "publishing company",
            "columbia university",
            "courtesy of",
            "cover photograph",
            "copyright",
            "isbn",
            "addison-wesley",
            "reading, massachusetts",
            "don mills",
            "menlo park",
            "london",
            "foreword",
            "acknowledg",
            "table of contents",
            "contents",
            "index",
        )
        if any(marker in lowered for marker in markers):
            return True
        alpha_words = re.findall(r"[a-zA-Z]+", text)
        if not alpha_words:
            return True
        uppercase_words = [word for word in alpha_words if word.isupper() and len(word) > 2]
        return len(uppercase_words) >= max(4, len(alpha_words) // 2)

    def _looks_like_filename(self, text: str) -> bool:
        candidate = text.strip().lower()
        return bool(re.fullmatch(r".+\.[a-z0-9]{2,5}", candidate))

    def _format_series(self, items: list[str]) -> str:
        cleaned = [item.strip() for item in items if item.strip()]
        if not cleaned:
            return "the retrieved material"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    def _match_claim_facts(
        self,
        claim_facts: list[ExtractedFact],
        recent_facts: list[ExtractedFact],
    ) -> tuple[list[ExtractedFact], list[ExtractedFact]]:
        support_matches: list[ExtractedFact] = []
        conflict_matches: list[ExtractedFact] = []

        for claim_fact in claim_facts:
            for fact in recent_facts:
                if self._facts_support_each_other(claim_fact, fact):
                    support_matches.append(fact)
                elif self._facts_conflict(claim_fact, fact):
                    conflict_matches.append(fact)

        support_matches.sort(key=lambda fact: fact.confidence, reverse=True)
        conflict_matches.sort(key=lambda fact: fact.confidence, reverse=True)
        return support_matches, conflict_matches

    def _facts_support_each_other(self, left: ExtractedFact, right: ExtractedFact) -> bool:
        same_subject = left.subject == right.subject
        same_relation = left.relation == right.relation or left.relation == "property" or right.relation == "property"
        same_value = left.value == right.value
        return same_subject and same_relation and same_value

    def _facts_conflict(self, left: ExtractedFact, right: ExtractedFact) -> bool:
        same_subject = left.subject == right.subject
        same_relation = left.relation == right.relation or left.relation == "property" or right.relation == "property"
        different_value = left.value != right.value
        return same_subject and same_relation and different_value

    def _filter_claim_evidence_memories(
        self,
        memories: list[Any],
        current_input_id: Optional[str],
    ) -> list[Any]:
        evidence_kinds = {"fact", "resolved_question", "knowledge_chunk", "knowledge_source"}
        filtered: list[Any] = []

        for memory in memories:
            if current_input_id and getattr(memory, "related_input_id", None) == current_input_id:
                continue

            metadata = getattr(memory, "metadata", {}) or {}
            has_structured_fact = all(str(metadata.get(field, "")).strip() for field in ("subject", "relation", "value"))
            if getattr(memory, "kind", "") in evidence_kinds or has_structured_fact:
                filtered.append(memory)

        return filtered

    def _combine_recent_memories(
        self,
        state_memories: list[Any],
        context_memories: list[Any],
    ) -> list[Any]:
        combined: list[Any] = []
        seen_ids: set[str] = set()

        for memory in list(state_memories[-60:]) + list(context_memories[:20]):
            memory_id = getattr(memory, "memory_id", None)
            if memory_id and memory_id in seen_ids:
                continue
            if memory_id:
                seen_ids.add(memory_id)
            combined.append(memory)
        return combined

    def _social_base_reply(self, social_subtype: str, text: str) -> str:
        if social_subtype == "greeting":
            return "Hello."
        if social_subtype == "status_check":
            return "Functional. Proceed."
        if social_subtype == "gratitude":
            return "You're welcome."
        if social_subtype == "farewell":
            return "Understood. I will remain here."
        if social_subtype == "approval":
            return "Noted."
        return "I received that."

    def _detect_social_subtype(self, text: str) -> str:
        normalized = text.strip().lower()

        if normalized in {"hi", "hello", "hey", "yo", "howdy", "good morning", "good afternoon", "good evening"}:
            return "greeting"

        if normalized.startswith(("hi ", "hello ", "hey ", "yo ", "good morning", "good afternoon", "good evening")):
            return "greeting"

        if normalized.startswith(("how are you", "how are things", "how's it going", "hows it going", "you there", "are you there")):
            return "status_check"

        if normalized in {"thanks", "thank you", "thx"} or normalized.startswith(("thanks ", "thank you ")):
            return "gratitude"

        if normalized in {"bye", "goodbye", "see you", "see ya", "later", "good night"} or normalized.startswith(("bye ", "goodbye ", "see you ")):
            return "farewell"

        if normalized in {"nice", "cool", "awesome", "great", "good job"}:
            return "approval"

        return "social"

    def _extract_addressed_social_command(self, text: str) -> Optional[dict[str, str]]:
        normalized = text.strip().lower().rstrip(".!?,")
        patterns = (
            r"^(?:please\s+)?(?:say|tell)\s+(?P<greeting>hello|hi|hey|good morning|good afternoon|good evening)\s+to\s+(?P<recipient>[a-z0-9][a-z0-9'_\-]*(?:\s+[a-z0-9][a-z0-9'_\-]*){0,2})$",
            r"^(?:please\s+)?greet\s+(?P<recipient>[a-z0-9][a-z0-9'_\-]*(?:\s+[a-z0-9][a-z0-9'_\-]*){0,2})$",
        )

        for pattern in patterns:
            match = re.match(pattern, normalized)
            if not match:
                continue

            recipient = (match.groupdict().get("recipient") or "").strip()
            if not recipient:
                continue

            greeting_phrase = (match.groupdict().get("greeting") or "hello").strip()
            recipient_name = " ".join(part.capitalize() for part in recipient.split())
            greeting_display = greeting_phrase[:1].upper() + greeting_phrase[1:]
            return {
                "recipient_name": recipient_name,
                "greeting_phrase": greeting_phrase,
                "greeting_display": greeting_display,
            }

        return None

    def _clip(self, text: str, length: int) -> str:
        if len(text) <= length:
            return text
        return text[: length - 3].rstrip() + "..."
