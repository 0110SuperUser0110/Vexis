from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.event_bus import EventBus
from core.state_manager import StateManager
from core.schemas import BlackboardEntry, Claim, Evidence, LifecycleState, Provenance
from core.temporal_engine import TemporalEngine
from core.working_memory import WorkingMemory, WorkingMemoryItem
from core.blackboard import Blackboard
from core.question_engine import QuestionEngine
from core.evidence_evaluator import EvidenceEvaluator
from core.bias_assessor import BiasAssessor
from interface.speech_router import SpeechRouter
from speech.speech_governor import SpeechGovernor, SpeechPolicy
from speech.tts_engine import TTSEngine
from speech.voice_queue import VoiceQueue


def main() -> None:
    state_manager = StateManager()
    event_bus = EventBus()
    working_memory = WorkingMemory()
    blackboard = Blackboard()
    temporal_engine = TemporalEngine()
    question_engine = QuestionEngine()
    evidence_evaluator = EvidenceEvaluator()
    bias_assessor = BiasAssessor()

    tts_engine = TTSEngine("config/voices.json")
    speech_governor = SpeechGovernor(
        SpeechPolicy(
            user_present=True,
            muted=False,
            allow_reflective=False,
            allow_creative=False,
            unsolicited_cooldown_seconds=300,
        )
    )
    voice_queue = VoiceQueue()
    speech_router = SpeechRouter(tts_engine, speech_governor, voice_queue)

    state_manager.load_state()
    state_manager.set_lifecycle_state(LifecycleState.ACTIVE)

    current_time = temporal_engine.current_time_context()
    state_manager.update_time_context(current_time)

    event_bus.publish(
        "system_boot",
        {
            "message": "VEXIS boot sequence initialized.",
            "date": current_time.date_label,
            "time": current_time.time_label,
            "weekday": current_time.weekday_name,
            "day_part": current_time.day_part,
        },
    )

    working_memory.add_item(
        WorkingMemoryItem(
            item_id="wm_boot_001",
            content="Evaluate initial system startup condition.",
            item_type="startup_review",
            priority=0.20,
        )
    )

    blackboard.post(
        BlackboardEntry(
            entry_id="bb_boot_001",
            source_module="boot",
            entry_type="system_status",
            content="VEXIS entered active lifecycle state.",
            priority=0.20,
            metadata={
                "date": current_time.date_label,
                "time": current_time.time_label,
                "weekday": current_time.weekday_name,
                "day_part": current_time.day_part,
            },
        )
    )

    claim = Claim(
        claim_id="claim_001",
        content="A blog post clearly proves that a new alloy is stronger than steel in all conditions.",
        provenance=Provenance(
            source_id="local_claim_input",
            source_type="user_input",
            author="prototype",
            title="Initial test claim",
        ),
    )

    evidence = Evidence(
        evidence_id="evidence_001",
        content="This blog says the alloy is obviously better and everyone knows steel is outdated.",
        provenance=Provenance(
            source_id="blog_test_source",
            source_type="blog",
            author="unknown",
            title="Unverified alloy article",
        ),
    )

    claim = bias_assessor.assess_claim(claim)
    evidence = evidence_evaluator.evaluate(evidence)
    evidence = bias_assessor.assess_evidence(evidence)

    generated_question = question_engine.generate_for_claim(claim)

    working_memory.add_item(
        WorkingMemoryItem(
            item_id="wm_question_001",
            content=generated_question.content,
            item_type="epistemic_question",
            priority=0.95,
        )
    )

    blackboard.post(
        BlackboardEntry(
            entry_id="bb_question_001",
            source_module="question_engine",
            entry_type="uncertainty",
            content=generated_question.content,
            priority=0.95,
            metadata={
                "claim_id": claim.claim_id,
                "claim_bias_flags": [flag.value for flag in claim.bias_flags],
                "evidence_strength": evidence.strength.value,
                "evidence_rigor_score": evidence.rigor_score,
                "evidence_bias_flags": [flag.value for flag in evidence.bias_flags],
            },
        )
    )

    highest_item = working_memory.highest_priority_item()
    top_blackboard = blackboard.highest_priority()
    state = state_manager.get_state()

    print("VEXIS core initialized.")
    print(f"Pending events: {event_bus.size()}")
    print(f"Lifecycle state: {state.self_model.lifecycle_state.value}")
    print(f"Working memory items: {working_memory.size()}")
    print(f"Blackboard entries: {blackboard.size()}")
    print(f"Current date: {current_time.date_label}")
    print(f"Current time: {current_time.time_label}")
    print(f"Weekday: {current_time.weekday_name}")
    print(f"Part of day: {current_time.day_part}")
    print(f"Boot count: {state.boot_count}")
    print(f"Crossed day boundary since last run: {state.crossed_day_boundary}")

    if state.previous_time_context is not None:
        print(
            "Previous observed time: "
            f"{state.previous_time_context.date_label} "
            f"{state.previous_time_context.time_label} "
            f"{state.previous_time_context.weekday_name}"
        )

    print(f"Generated question: {generated_question.content}")
    print(f"Claim bias flags: {[flag.value for flag in claim.bias_flags]}")
    print(f"Evidence strength: {evidence.strength.value}")
    print(f"Evidence rigor score: {evidence.rigor_score}")
    print(f"Evidence bias flags: {[flag.value for flag in evidence.bias_flags]}")

    if highest_item:
        print(f"Top working item: {highest_item.content}")

    if top_blackboard:
        print(f"Top blackboard note: {top_blackboard.content}")

    print(f"Temporal label for now: {temporal_engine.relative_day_label(temporal_engine.now())}")

    spoken, speech_reason = speech_router.say(
        text=(
            f"It is {current_time.time_label} on {current_time.weekday_name}. "
            f"I have formed an epistemic question for review."
        ),
        speech_type="response",
        user_present=True,
    )
    print(f"Speech result: {spoken} ({speech_reason})")

    state_manager.save_state()


if __name__ == "__main__":
    main()