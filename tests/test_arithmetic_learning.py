from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.fact_extractor import FactExtractor
from core.input_classifier import InputClassifier
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


class TestArithmeticLearning(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"arithmetic_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.classifier = InputClassifier()
        self.context_builder = ContextBuilder(self.memory_store)
        self.reasoning_engine = ReasoningEngine()
        self.response_engine = ResponseEngine(
            reasoning_engine=self.reasoning_engine,
            language_renderer=LanguageRenderer(llm_router=None),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_fact_extractor_extracts_arithmetic_teaching(self) -> None:
        extractor = FactExtractor()

        result = extractor.extract("1 + 1 = 2 therefore any number +1 equals 1 higher number.")

        self.assertTrue(result.success)
        self.assertTrue(any(fact.fact_type == "arithmetic_equation" for fact in result.facts))
        self.assertTrue(any(fact.relation == "successor_rule" for fact in result.facts))

    def test_response_engine_resolves_simple_arithmetic_deterministically(self) -> None:
        input_record = self.state_manager.add_input(
            raw_text="What is 1 plus 1?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        classification = self.classifier.classify("What is 1 plus 1?")
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=None,
            prefer_llm=False,
        )

        self.assertEqual(bundle.internal_answer["answer_type"], "question_response")
        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertIn("2", bundle.response_text)
        self.assertIn("resolved_by_deterministic_arithmetic", bundle.internal_answer["metadata"]["resolution"]["reasons"])

    def test_taught_arithmetic_statement_becomes_support_for_followup_question(self) -> None:
        teaching_text = "1 + 1 = 2 therefore any number +1 equals 1 higher number."
        teaching_input = self.state_manager.add_input(
            raw_text=teaching_text,
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(teaching_input)

        teaching_memory = self.state_manager.add_memory(
            kind="note",
            content=teaching_text,
            source="gui",
            related_input_id=teaching_input.input_id,
        )
        self.memory_store.save_memory(teaching_memory)

        teaching_classification = self.classifier.classify(teaching_text)
        teaching_context = self.context_builder.build(self.state_manager.get_state(), teaching_input)
        teaching_bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=teaching_input,
            classification=teaching_classification,
            memory_record=teaching_memory,
            task_record=None,
            context=teaching_context,
            prefer_llm=False,
        )

        extraction = teaching_bundle.internal_answer["metadata"].get("fact_extraction", {})
        self.assertTrue(extraction.get("success"))
        self.assertTrue(any(fact.get("fact_type") == "arithmetic_equation" for fact in extraction.get("facts", [])))

        question_input = self.state_manager.add_input(
            raw_text="What is 1 plus 1?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(question_input)

        question_classification = self.classifier.classify("What is 1 plus 1?")
        question_context = self.context_builder.build(self.state_manager.get_state(), question_input)
        question_bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=question_input,
            classification=question_classification,
            memory_record=None,
            task_record=None,
            context=question_context,
            prefer_llm=False,
        )

        resolution = question_bundle.internal_answer["metadata"].get("resolution", {})
        self.assertTrue(question_bundle.internal_answer["resolved"])
        self.assertIn("2", question_bundle.response_text)
        self.assertTrue(
            any(
                reason in resolution.get("reasons", [])
                for reason in ("supported_by_learned_arithmetic_fact", "supported_by_learned_arithmetic_rule")
            )
        )


if __name__ == "__main__":
    unittest.main()
