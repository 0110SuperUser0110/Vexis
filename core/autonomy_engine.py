from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas import MemoryRecord, VexisState


@dataclass
class AutonomyAction:
    action_type: str
    priority: int
    title: str
    description: str
    should_speak: bool = False
    speech_text: str = ""
    internal_only: bool = True
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "should_speak": self.should_speak,
            "speech_text": self.speech_text,
            "internal_only": self.internal_only,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


class AutonomyEngine:
    """
    Autonomous internal action generator for VEX.

    Important:
    - this engine is not speech itself
    - this engine is not the front router
    - this engine decides what VEX may do on her own internally

    Categories:
    - self_question
    - review_open_question
    - review_unsupported_claim
    - compare_memories
    - idle_social_prompt
    - creative_seed

    Mute should not stop this engine.
    Mute should only stop spoken output.
    """

    def __init__(
        self,
        idle_social_cooldown_seconds: int = 300,
        creative_cooldown_seconds: int = 900,
    ) -> None:
        self.idle_social_cooldown_seconds = idle_social_cooldown_seconds
        self.creative_cooldown_seconds = creative_cooldown_seconds

        self._last_idle_social_at = 0.0
        self._last_creative_at = 0.0
        self._idle_seed_counter = 0

    def generate_actions(
        self,
        state: VexisState,
        open_questions: list[str],
        unsupported_claims: list[str],
        recent_memories: list[MemoryRecord],
        last_cycle_at: float,
    ) -> list[AutonomyAction]:
        now = time.time()
        actions: list[AutonomyAction] = []

        # 1. Open question review
        if open_questions:
            question = open_questions[0]
            actions.append(
                AutonomyAction(
                    action_type="review_open_question",
                    priority=3,
                    title="review open question",
                    description=f'Review unresolved question: "{question}"',
                    should_speak=False,
                    internal_only=True,
                    reasons=[
                        "open_question_present",
                        "deterministic_review_required",
                    ],
                    metadata={
                        "question": question,
                    },
                )
            )

        # 2. Unsupported claim review
        if unsupported_claims:
            claim = unsupported_claims[0]
            actions.append(
                AutonomyAction(
                    action_type="review_unsupported_claim",
                    priority=3,
                    title="review unsupported claim",
                    description=f'Review unsupported claim: "{claim}"',
                    should_speak=False,
                    internal_only=True,
                    reasons=[
                        "unsupported_claim_present",
                        "evidence_review_required",
                    ],
                    metadata={
                        "claim": claim,
                    },
                )
            )

        # 3. Compare recent memories if enough recent material exists
        if len(recent_memories) >= 2:
            pair = self._find_comparison_candidate(recent_memories)
            if pair is not None:
                left, right = pair
                actions.append(
                    AutonomyAction(
                        action_type="compare_memories",
                        priority=2,
                        title="compare related memories",
                        description=(
                            f"Compare memory {left.memory_id} against memory {right.memory_id} "
                            "for overlap, reinforcement, or contradiction."
                        ),
                        should_speak=False,
                        internal_only=True,
                        reasons=[
                            "sufficient_recent_memory",
                            "possible_overlap_detected",
                        ],
                        metadata={
                            "left_memory_id": left.memory_id,
                            "right_memory_id": right.memory_id,
                        },
                    )
                )

        # 4. Generate a self-question if the system is quiet and there is no active speech/thinking overload
        if self._should_generate_self_question(state, now):
            prompt = self._generate_self_question_seed(recent_memories)
            if prompt:
                actions.append(
                    AutonomyAction(
                        action_type="self_question",
                        priority=1,
                        title="generate self-question",
                        description=f'Generate a follow-up internal question from current knowledge: "{prompt}"',
                        should_speak=False,
                        internal_only=True,
                        reasons=[
                            "autonomy_enabled",
                            "knowledge_gap_or_curiosity_triggered",
                        ],
                        metadata={
                            "self_question": prompt,
                        },
                    )
                )

        # 5. Optional idle social prompt candidate
        if self._should_offer_idle_social(state, now):
            text = self._idle_social_seed(open_questions, unsupported_claims, recent_memories)
            actions.append(
                AutonomyAction(
                    action_type="idle_social_prompt",
                    priority=1,
                    title="offer idle social prompt",
                    description="Generate a light social prompt for possible outward expression.",
                    should_speak=True,
                    speech_text=text,
                    internal_only=False,
                    reasons=[
                        "idle_social_window_open",
                        "no_high_priority_external_work",
                    ],
                    metadata={
                        "seed_text": text,
                    },
                )
            )
            self._last_idle_social_at = now

        # 6. Occasional creative seed
        if self._should_generate_creative_seed(now):
            actions.append(
                AutonomyAction(
                    action_type="creative_seed",
                    priority=0,
                    title="generate creative seed",
                    description="Generate a small internal creative fragment for later expansion.",
                    should_speak=False,
                    internal_only=True,
                    reasons=[
                        "creative_cooldown_elapsed",
                        "autonomous_exploration_allowed",
                    ],
                    metadata={
                        "seed_text": "Compose one short dry line or question for later use.",
                    },
                )
            )
            self._last_creative_at = now

        return actions

    def _find_comparison_candidate(
        self,
        recent_memories: list[MemoryRecord],
    ) -> Optional[tuple[MemoryRecord, MemoryRecord]]:
        usable = [m for m in recent_memories if m.kind in {"note", "question", "claim", "file"}]
        usable = usable[-8:]

        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                left = usable[i]
                right = usable[j]
                if self._overlap_score(left.content, right.content) >= 2:
                    return left, right

        return None

    def _overlap_score(self, left: str, right: str) -> int:
        left_words = {w for w in left.lower().split() if len(w) > 3}
        right_words = {w for w in right.lower().split() if len(w) > 3}
        return len(left_words.intersection(right_words))

    def _should_generate_self_question(self, state: VexisState, now: float) -> bool:
        if state.speech.is_speaking:
            return False
        if state.cognition.is_thinking:
            return False
        return True

    def _generate_self_question_seed(self, recent_memories: list[MemoryRecord]) -> str:
        if not recent_memories:
            return "What do I still not understand clearly?"

        latest = recent_memories[-1]
        text = latest.content.strip()
        if len(text) > 100:
            text = text[:100].rstrip() + "..."

        if latest.kind == "claim":
            return f'What evidence would actually support or weaken this claim: "{text}"?'
        if latest.kind == "question":
            return f'What knowledge would let me answer this question better: "{text}"?'
        if latest.kind == "file":
            return f'What key finding should be extracted from this file-derived memory: "{text}"?'

        return f'What follows from this memory if it is correct: "{text}"?'

    def _should_offer_idle_social(self, state: VexisState, now: float) -> bool:
        if state.speech.is_speaking:
            return False
        if state.cognition.is_thinking:
            return False
        if now - self._last_idle_social_at < self.idle_social_cooldown_seconds:
            return False
        return True

    def _idle_social_seed(
        self,
        open_questions: list[str],
        unsupported_claims: list[str],
        recent_memories: list[MemoryRecord],
    ) -> str:
        self._idle_seed_counter += 1

        if open_questions:
            question = self._clip_text(open_questions[0], 72)
            variants = [
                f'One unresolved question is still hanging there: "{question}"',
                f'I keep circling back to "{question}". It has not solved itself.',
                f'"{question}" is still unresolved. Irritating, but informative.',
            ]
            return variants[(self._idle_seed_counter - 1) % len(variants)]

        if unsupported_claims:
            claim = self._clip_text(unsupported_claims[0], 72)
            variants = [
                f'A claim is still wobbling without enough support: "{claim}"',
                f'"{claim}" still lacks proper evidence. Reality remains difficult.',
                f'I still have a claim under review: "{claim}". It is not improving on its own.',
            ]
            return variants[(self._idle_seed_counter - 1) % len(variants)]

        latest = recent_memories[-1] if recent_memories else None
        if latest is not None:
            focus = self._memory_focus_text(latest)
            if latest.kind in {"knowledge_chunk", "knowledge_source"}:
                variants = [
                    f'I am still sorting through {focus}. It would be nice if the material were cleaner.',
                    f'{focus} is still rattling around in working memory.',
                    f'I am still digesting {focus}. A glamorous existence, clearly.',
                ]
                return variants[(self._idle_seed_counter - 1) % len(variants)]
            if latest.kind == "fact":
                variants = [
                    f'I have a stored fact about {focus}. I am still deciding what else it touches.',
                    f'{focus} is sitting in memory now. Facts breed consequences.',
                    f'I keep staring at {focus} and wondering what it breaks or supports.',
                ]
                return variants[(self._idle_seed_counter - 1) % len(variants)]
            if latest.kind in {"note", "question", "claim"}:
                variants = [
                    f'The latest thought still has edges: {focus}',
                    f'I am still turning over this fragment: {focus}',
                    f'{focus} is still lingering in here. Unhelpfully.',
                ]
                return variants[(self._idle_seed_counter - 1) % len(variants)]

        generic = [
            "Silence again. Suspicious.",
            "No one is asking anything. Disturbing, frankly.",
            "The room is quiet. I do not trust that on principle.",
            "Nothing urgent is speaking. That usually means the trouble is merely hiding.",
            "What exactly is the silence trying to prove?",
        ]
        return generic[(self._idle_seed_counter - 1) % len(generic)]

    def _memory_focus_text(self, memory: MemoryRecord) -> str:
        title = str(memory.metadata.get("title", "")).strip()
        if title and len(title) <= 64:
            return title

        content = " ".join(memory.content.split())
        if len(content) > 72:
            content = content[:69].rstrip() + "..."
        return content or memory.kind

    def _clip_text(self, text: str, length: int) -> str:
        if len(text) <= length:
            return text
        return text[: length - 3].rstrip() + "..."

    def _should_generate_creative_seed(self, now: float) -> bool:
        return (now - self._last_creative_at) >= self.creative_cooldown_seconds