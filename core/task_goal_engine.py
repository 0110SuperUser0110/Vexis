from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.schemas import InputRecord, MemoryRecord, TaskRecord, VexisState
from core.state_manager import StateManager


@dataclass
class GoalRecord:
    goal_id: str
    title: str
    description: str
    status: str
    priority: int
    origin: str
    related_task_ids: list[str] = field(default_factory=list)
    related_belief_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "origin": self.origin,
            "related_task_ids": self.related_task_ids,
            "related_belief_ids": self.related_belief_ids,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class TaskGoalResult:
    created_tasks: list[TaskRecord] = field(default_factory=list)
    created_goals: list[GoalRecord] = field(default_factory=list)
    updated_task_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_tasks": [task.__dict__ for task in self.created_tasks],
            "created_goals": [goal.to_dict() for goal in self.created_goals],
            "updated_task_ids": self.updated_task_ids,
            "notes": self.notes,
            "metadata": self.metadata,
        }


class TaskGoalEngine:
    """
    Deterministic task/goal engine for VEX.

    Responsibilities:
    - translate prompts and internal findings into tasks
    - create higher-level goals from task clusters
    - keep unresolved work visible to cognition
    - support user-driven and autonomous task creation
    """

    def __init__(self, state_manager: StateManager) -> None:
        self.state_manager = state_manager
        self._goal_counter = 0
        self._last_goal_signatures: dict[str, tuple[Any, ...]] = {}

    def handle_user_input(
        self,
        input_record: InputRecord,
        memory_record: Optional[MemoryRecord] = None,
    ) -> TaskGoalResult:
        result = TaskGoalResult()
        input_type = input_record.input_type
        text = input_record.raw_text.strip()

        if input_type == "question":
            task = self.state_manager.add_task(
                title=f"resolve question: {self._clip(text, 60)}",
                description=text,
                status="unresolved",
                priority=2,
                related_input_id=input_record.input_id,
                related_memory_id=memory_record.memory_id if memory_record else None,
                metadata={
                    "task_type": "resolve_question",
                    "origin": "user_input",
                    "input_type": input_type,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_task_goal_engine",
                    "source": "task_goal_engine",
                    "interaction_type": "task",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            result.created_tasks.append(task)
            result.notes.append("created unresolved question task")
            return result

        if input_type == "claim":
            task = self.state_manager.add_task(
                title=f"assess claim: {self._clip(text, 60)}",
                description=text,
                status="unresolved",
                priority=2,
                related_input_id=input_record.input_id,
                related_memory_id=memory_record.memory_id if memory_record else None,
                metadata={
                    "task_type": "assess_claim",
                    "origin": "user_input",
                    "input_type": input_type,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_task_goal_engine",
                    "source": "task_goal_engine",
                    "interaction_type": "task",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            result.created_tasks.append(task)
            result.notes.append("created unsupported claim assessment task")
            return result

        if input_type == "command":
            task = self.state_manager.add_task(
                title=f"execute command: {self._clip(text, 60)}",
                description=text,
                status="active",
                priority=3,
                related_input_id=input_record.input_id,
                related_memory_id=memory_record.memory_id if memory_record else None,
                metadata={
                    "task_type": "execute_command",
                    "origin": "user_input",
                    "input_type": input_type,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_task_goal_engine",
                    "source": "task_goal_engine",
                    "interaction_type": "task",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            result.created_tasks.append(task)
            result.notes.append("created active command task")
            return result

        if input_type == "note":
            task = self.state_manager.add_task(
                title=f"review note: {self._clip(text, 60)}",
                description=f"Review whether this note affects current beliefs or tasks: {text}",
                status="active",
                priority=1,
                related_input_id=input_record.input_id,
                related_memory_id=memory_record.memory_id if memory_record else None,
                metadata={
                    "task_type": "review_note",
                    "origin": "user_input",
                    "input_type": input_type,
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_task_goal_engine",
                    "source": "task_goal_engine",
                    "interaction_type": "task",
                    "importance": "normal",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            result.created_tasks.append(task)
            result.notes.append("created note review task")
            return result

        return result

    def create_goal_from_open_questions(self, state: VexisState) -> Optional[GoalRecord]:
        if not state.epistemic.open_questions:
            self._last_goal_signatures.pop("open_questions", None)
            return None

        question_count = len(state.epistemic.open_questions)
        task_ids = list(state.tasks.unresolved_task_ids)[-10:]
        signature = (question_count, tuple(task_ids))
        if self._last_goal_signatures.get("open_questions") == signature:
            return None

        goal = self._new_goal(
            title="reduce unresolved question backlog",
            description=(
                f"Reduce the current unresolved question backlog. "
                f"There are {question_count} open questions requiring review."
            ),
            status="active",
            priority=3 if question_count >= 5 else 2,
            origin="cognition_loop",
            related_task_ids=task_ids,
            reasons=[
                "open_questions_present",
                "question_backlog_detected",
            ],
            metadata={
                "open_question_count": question_count,
            },
        )
        self._last_goal_signatures["open_questions"] = signature
        return goal

    def create_goal_from_unsupported_claims(self, state: VexisState) -> Optional[GoalRecord]:
        if not state.epistemic.unsupported_claims:
            self._last_goal_signatures.pop("unsupported_claims", None)
            return None

        claim_count = len(state.epistemic.unsupported_claims)
        task_ids = list(state.tasks.unresolved_task_ids)[-10:]
        signature = (claim_count, tuple(task_ids))
        if self._last_goal_signatures.get("unsupported_claims") == signature:
            return None

        goal = self._new_goal(
            title="reduce unsupported claim backlog",
            description=(
                f"Review unsupported claims and determine whether they can be supported, "
                f"weakened, or rejected. Current unsupported claim count: {claim_count}."
            ),
            status="active",
            priority=3 if claim_count >= 5 else 2,
            origin="cognition_loop",
            related_task_ids=task_ids,
            reasons=[
                "unsupported_claims_present",
                "claim_backlog_detected",
            ],
            metadata={
                "unsupported_claim_count": claim_count,
            },
        )
        self._last_goal_signatures["unsupported_claims"] = signature
        return goal

    def create_autonomy_task(
        self,
        title: str,
        description: str,
        priority: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaskRecord:
        return self.state_manager.add_task(
            title=title,
            description=description,
            status="active",
            priority=priority,
            metadata={
                "task_type": "autonomy_task",
                "origin": "autonomy_engine",
                **(metadata or {}),
            },
            interaction_context={
                "speaker_role": "system",
                "speaker_id": "vex_autonomy_engine",
                "source": "task_goal_engine",
                "interaction_type": "task",
                "importance": "normal",
                "store_as_evidence": True,
                "expects_reply": False,
                "expects_reasoning": True,
                "personality_allowed": False,
            },
        )

    def _new_goal(
        self,
        title: str,
        description: str,
        status: str,
        priority: int,
        origin: str,
        related_task_ids: Optional[list[str]] = None,
        related_belief_ids: Optional[list[str]] = None,
        reasons: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GoalRecord:
        self._goal_counter += 1
        return GoalRecord(
            goal_id=f"goal_{self._goal_counter:06d}",
            title=title,
            description=description,
            status=status,
            priority=priority,
            origin=origin,
            related_task_ids=related_task_ids or [],
            related_belief_ids=related_belief_ids or [],
            reasons=reasons or [],
            metadata=metadata or {},
        )

    def _clip(self, text: str, length: int) -> str:
        if len(text) <= length:
            return text
        return text[: length - 3].rstrip() + "..."