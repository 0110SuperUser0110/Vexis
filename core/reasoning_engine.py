from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.belief_engine import BeliefRecord

from core.context_builder import ContextBundle
from core.contradiction_engine import ContradictionBatchResult, ContradictionEngine
from core.input_classifier import ClassificationResult
from core.schemas import InputRecord, MemoryRecord, TaskRecord, VexisState
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

    This is the actual cognition layer.
    The LLM is only for language and personality filtering.
    """

    def __init__(
        self,
        self_model: Optional[SelfModel] = None,
        contradiction_engine: Optional[ContradictionEngine] = None,
    ) -> None:
        self.self_model = self_model
        self.contradiction_engine = contradiction_engine or ContradictionEngine()

    def reason(
        self,
        state: VexisState,
        input_record: InputRecord,
        classification: ClassificationResult,
        context: Optional[ContextBundle] = None,
        task_record: Optional[TaskRecord] = None,
        belief_records: Optional[list[BeliefRecord]] = None,
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
        belief_records: list[BeliefRecord],
    ) -> InternalAnswer:
        question_text = input_record.raw_text.strip().lower()

        # 1. Self-model answers first.
        if self.self_model is not None:
            self_answer = self.self_model.answer_identity_question(question_text)
            if self_answer:
                return InternalAnswer(
                    answer_type="question_response",
                    resolved=True,
                    confidence=0.90,
                    facts=[self_answer],
                    unknowns=[],
                    grounding=["self_model"],
                    actions=[],
                    proposed_text=self_answer,
                    metadata={
                        "classification": classification.input_type,
                        "task_id": task_record.task_id if task_record else None,
                        "reasoning_source": "self_model",
                    },
                )

        # 2. Belief candidates.
        belief_match = self._best_belief_match(question_text, belief_records)
        if belief_match is not None:
            statement, score, belief = belief_match
            return InternalAnswer(
                answer_type="question_response",
                resolved=True,
                confidence=min(max(belief.confidence_score, 0.42), 0.95),
                facts=[
                    f"Relevant belief candidate: {statement}",
                    f"Confidence label: {belief.confidence_label}.",
                ],
                unknowns=[],
                grounding=[
                    f"belief:{belief.belief_id}",
                    f"belief_status:{belief.status}",
                ],
                actions=[],
                proposed_text=f"{statement} My current confidence in that is {belief.confidence_label}.",
                metadata={
                    "classification": classification.input_type,
                    "task_id": task_record.task_id if task_record else None,
                    "reasoning_source": "belief_engine",
                    "belief_id": belief.belief_id,
                    "belief_score": score,
                },
            )

        # 3. Memory recall.
        grounding: list[str] = []
        facts: list[str] = []
        unknowns: list[str] = []
        actions: list[str] = []

        if context and context.related_memories:
            best = context.related_memories[0]
            best_text_lower = best.content.lower()
            shared_keywords = [
                word for word in question_text.split()
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

        # 4. Recent context, but unresolved.
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

        # 5. No answer.
        if task_record:
            grounding.append(f"task:{task_record.task_id}")
            actions.append("keep unresolved task active")

        if state.epistemic.open_questions:
            grounding.append(f"open_question_count:{len(state.epistemic.open_questions)}")

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
        belief_records: list[BeliefRecord],
    ) -> InternalAnswer:
        grounding: list[str] = []
        facts: list[str] = []
        unknowns: list[str] = ["The claim is not yet verified."]
        actions: list[str] = ["mark claim as unverified"]

        # Compare claim against nearby memories for contradictions.
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
        text = input_record.raw_text.lower()
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
        belief_records: list[BeliefRecord],
    ) -> Optional[tuple[str, float, BeliefRecord]]:
        best_match: Optional[tuple[str, float, BeliefRecord]] = None
        question_terms = {word for word in question_text.split() if len(word) > 3}

        for belief in belief_records:
            belief_terms = {word for word in belief.statement.lower().split() if len(word) > 3}
            overlap = len(question_terms.intersection(belief_terms))
            if overlap == 0:
                continue

            score = overlap + belief.confidence_score
            if best_match is None or score > best_match[1]:
                best_match = (belief.statement, score, belief)

        return best_match

    def _social_base_reply(self, social_subtype: str, text: str) -> str:
        if social_subtype == "greeting":
            return "Hello. I am here."
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

    def _clip(self, text: str, length: int) -> str:
        if len(text) <= length:
            return text
        return text[: length - 3].rstrip() + "..."