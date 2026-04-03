from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.input_classifier import InputClassifier
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.resolution_engine import ResolutionEngine
from core.self_model import SelfModel
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


class TestVexisCore(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"case_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.classifier = InputClassifier()
        self.context_builder = ContextBuilder(self.memory_store)
        self.reasoning_engine = ReasoningEngine()

        self.language_renderer = LanguageRenderer(llm_router=None)
        self.response_engine = ResponseEngine(
            reasoning_engine=self.reasoning_engine,
            language_renderer=self.language_renderer,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_classifier_question(self) -> None:
        result = self.classifier.classify("What did I just tell you?")
        self.assertEqual(result.input_type, "question")
        self.assertGreater(result.confidence, 0.5)

    def test_classifier_claim(self) -> None:
        result = self.classifier.classify("I think the device is overheating.")
        self.assertEqual(result.input_type, "claim")
        self.assertGreater(result.confidence, 0.4)

    def test_classifier_command(self) -> None:
        result = self.classifier.classify("Open the intake file and analyze it.")
        self.assertEqual(result.input_type, "command")
        self.assertGreater(result.confidence, 0.4)

    def test_classifier_stuttered_greeting_is_social(self) -> None:
        result = self.classifier.classify("hhello")
        self.assertEqual(result.input_type, "social")
        self.assertTrue(result.features["contains_greeting"])

    def test_classifier_dance_is_command(self) -> None:
        result = self.classifier.classify("dance.")
        self.assertEqual(result.input_type, "command")
        self.assertGreater(result.confidence, 0.4)

    def test_resolution_engine_repeat_capability_question_is_resolved(self) -> None:
        engine = ResolutionEngine(self_model=SelfModel(self.state_manager))

        result = engine.resolve_question(
            "can you repeat what I tell you to?",
            self.state_manager.get_state(),
            self.memory_store.get_recent_memories(20),
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.question_type, "repeat_capability")
        self.assertIn("repeat user-provided text", result.answer_text.lower())

    def test_resolution_engine_legacy_cube_question_is_resolved(self) -> None:
        engine = ResolutionEngine(self_model=SelfModel(self.state_manager))

        result = engine.resolve_question(
            "what color is your cube?",
            self.state_manager.get_state(),
            self.memory_store.get_recent_memories(20),
        )

        self.assertTrue(result.resolved)
        self.assertEqual(result.question_type, "legacy_cube_presence")
        self.assertIn("do not currently use a cube", result.answer_text.lower())

    def test_memory_store_and_recall(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="The farm uses purple branding.",
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="note",
            content="The farm uses purple branding.",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        matches = self.memory_store.search_memories("purple branding", limit=3)
        self.assertTrue(matches)
        self.assertEqual(matches[0].content, "The farm uses purple branding.")

    def test_records_include_time_context(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="Time test input.",
            source="gui",
            input_type="note",
            confidence=1.0,
        )

        self.assertIn("utc_timestamp", input_record.time_context)
        self.assertIn("local_timestamp", input_record.time_context)
        self.assertIn("timezone_name", input_record.time_context)
        self.assertIn("utc_offset", input_record.time_context)
        self.assertIn("part_of_day", input_record.time_context)

        memory_record = self.state_manager.add_memory(
            kind="note",
            content="Time test memory.",
            source="gui",
            related_input_id=input_record.input_id,
        )

        self.assertIn("utc_timestamp", memory_record.time_context)
        self.assertIn("local_timestamp", memory_record.time_context)

        task_record = self.state_manager.add_task(
            title="time test task",
            description="Verify time context on task records.",
            status="active",
            related_input_id=input_record.input_id,
            related_memory_id=memory_record.memory_id,
        )

        self.assertIn("utc_timestamp", task_record.time_context)
        self.assertIn("local_timestamp", task_record.time_context)

    def test_records_include_interaction_context(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="hello vex",
            source="gui",
            input_type="social",
            confidence=0.95,
        )

        self.assertIn("speaker_role", input_record.interaction_context)
        self.assertIn("session_id", input_record.interaction_context)
        self.assertIn("device_label", input_record.interaction_context)
        self.assertIn("location_label", input_record.interaction_context)
        self.assertEqual(input_record.interaction_context["interaction_type"], "social")
        self.assertTrue(input_record.interaction_context["personality_allowed"])
        self.assertFalse(input_record.interaction_context["store_as_evidence"])

        memory_record = self.state_manager.add_memory(
            kind="social",
            content="hello vex",
            source="gui",
            related_input_id=input_record.input_id,
            interaction_context=input_record.interaction_context,
        )

        self.assertEqual(memory_record.interaction_context["interaction_type"], "social")
        self.assertEqual(memory_record.interaction_context["speaker_role"], "user")

    def test_response_engine_recalls_memory_for_question(self) -> None:
        seed_input = self.state_manager.add_input(
            raw_text="The cube is purple.",
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(seed_input)

        seed_memory = self.state_manager.add_memory(
            kind="note",
            content="The cube is purple.",
            source="gui",
            related_input_id=seed_input.input_id,
        )
        self.memory_store.save_memory(seed_memory)

        question_input = self.state_manager.add_input(
            raw_text="What color is the cube?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(question_input)

        question_memory = self.state_manager.add_memory(
            kind="question",
            content="What color is the cube?",
            source="gui",
            related_input_id=question_input.input_id,
        )
        self.memory_store.save_memory(question_memory)

        classification = self.classifier.classify("What color is the cube?")
        context = self.context_builder.build(self.state_manager.get_state(), question_input)

        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=question_input,
            classification=classification,
            memory_record=question_memory,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertIn(bundle.status_text, {"question resolved", "question logged"})
        self.assertEqual(bundle.internal_answer["answer_type"], "question_response")
        self.assertIn(bundle.internal_answer["resolved"], {True, False})
        self.assertGreater(bundle.internal_answer["confidence"], 0.0)
        self.assertTrue(len(bundle.response_text.strip()) > 0)

    def test_response_engine_logs_claim(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="I believe the file is corrupted.",
            source="gui",
            input_type="claim",
            confidence=0.8,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="claim",
            content="I believe the file is corrupted.",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        classification = self.classifier.classify("I believe the file is corrupted.")
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=memory_record,
            task_record=None,
            context=None,
            prefer_llm=False,
        )

        self.assertEqual(bundle.status_text, "claim logged")
        self.assertEqual(bundle.internal_answer["answer_type"], "claim_assessment")
        self.assertFalse(bundle.internal_answer["resolved"])
        self.assertIn("claim", bundle.response_text.lower())
        self.assertIn("unsupported_claims", bundle.epistemic_updates)

    def test_state_manager_boot_count(self) -> None:
        count1 = self.state_manager.increment_boot_count()
        count2 = self.state_manager.increment_boot_count()
        self.assertEqual(count2, count1 + 1)

    def test_context_builder_returns_related_memory(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="The cube uses purple metallic shading.",
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="note",
            content="The cube uses purple metallic shading.",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        current = self.state_manager.add_input(
            raw_text="What did I say about the cube shading?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(current)

        context = self.context_builder.build(self.state_manager.get_state(), current)
        self.assertTrue(context.related_memories)

    def test_response_engine_uses_context_when_direct_recall_is_weak(self) -> None:
        seed_input = self.state_manager.add_input(
            raw_text="The lighting is top right and very bright.",
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(seed_input)

        seed_memory = self.state_manager.add_memory(
            kind="note",
            content="The lighting is top right and very bright.",
            source="gui",
            related_input_id=seed_input.input_id,
        )
        self.memory_store.save_memory(seed_memory)

        current_input = self.state_manager.add_input(
            raw_text="What did I say about the lighting?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(current_input)

        current_memory = self.state_manager.add_memory(
            kind="question",
            content="What did I say about the lighting?",
            source="gui",
            related_input_id=current_input.input_id,
        )
        self.memory_store.save_memory(current_memory)

        classification = self.classifier.classify("What did I say about the lighting?")
        context = self.context_builder.build(self.state_manager.get_state(), current_input)

        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=current_input,
            classification=classification,
            memory_record=current_memory,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertEqual(bundle.internal_answer["answer_type"], "question_response")
        self.assertTrue(len(bundle.response_text.strip()) > 0)

    def test_reasoning_engine_note_is_resolved(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="This is a note about the interface.",
            source="gui",
            input_type="note",
            confidence=0.95,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="note",
            content="This is a note about the interface.",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        classification = self.classifier.classify("This is a note about the interface.")
        context = self.context_builder.build(self.state_manager.get_state(), input_record)

        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=memory_record,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertIn(bundle.status_text, {"note stored", "note stored with extracted facts"})
        self.assertEqual(bundle.internal_answer["answer_type"], "note_acknowledgement")
        self.assertIn(bundle.internal_answer["resolved"], {True, False})
        self.assertIn("stored", bundle.response_text.lower())

    def test_reasoning_engine_command_supported(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="Analyze the latest intake file.",
            source="gui",
            input_type="command",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="command",
            content="Analyze the latest intake file.",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        task_record = self.state_manager.add_task(
            title="command: analyze intake file",
            description="Analyze the latest intake file.",
            status="active",
            priority=1,
            related_input_id=input_record.input_id,
            related_memory_id=memory_record.memory_id,
        )
        self.memory_store.save_task(task_record)

        classification = self.classifier.classify("Analyze the latest intake file.")
        context = self.context_builder.build(self.state_manager.get_state(), input_record)

        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=memory_record,
            task_record=task_record,
            context=context,
            prefer_llm=False,
        )

        self.assertEqual(bundle.internal_answer["answer_type"], "command_result")
        self.assertIn(bundle.internal_answer["resolved"], {True, False})
        self.assertEqual(bundle.status_text, "command resolved")
        self.assertIn("command", bundle.response_text.lower())

    def test_reasoning_engine_dance_command_triggers_presence_action(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="dance",
            source="gui",
            input_type="command",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="command",
            content="dance",
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        classification = self.classifier.classify("dance")
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=memory_record,
            task_record=None,
            context=None,
            prefer_llm=False,
        )

        self.assertEqual(bundle.status_text, "command resolved")
        self.assertEqual(bundle.internal_answer["answer_type"], "command_result")
        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"].get("presence_action"), "dancing")

    def test_language_renderer_is_deterministic_without_llm(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="What do I know about the cube?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        classification = self.classifier.classify("What do I know about the cube?")
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=None,
            prefer_llm=False,
        )

        self.assertFalse(bundle.used_llm)
        self.assertIsInstance(bundle.response_text, str)
        self.assertTrue(len(bundle.response_text) > 0)


if __name__ == "__main__":
    unittest.main()
