from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.input_classifier import ClassificationResult, InputClassifier
from core.knowledge_memory_bridge import KnowledgeMemoryBridge
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngest
from memory.memory_store import MemoryStore


class TestDeterministicCoreExpansion(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"core_expansion_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.context_builder = ContextBuilder(self.memory_store)
        self.classifier = InputClassifier()
        self.response_engine = ResponseEngine(
            reasoning_engine=ReasoningEngine(),
            language_renderer=LanguageRenderer(llm_router=None),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_generic_knowledge_question_resolves_from_uploaded_text(self) -> None:
        text = (
            "CALCULUS NOTES\n\n"
            "A derivative measures the rate of change of a quantity with respect to another quantity. "
            "In applications, derivatives describe slope and instantaneous change.\n\n"
            "An integral accumulates quantities across an interval."
        )

        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=10)
        result = ingest.ingest_text(
            text=text,
            source_path="C:/books/calculus_notes.txt",
            metadata={"title": "calculus_notes.txt"},
        )
        self.assertTrue(result.success)

        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)
        self.assertTrue(bridge_result.success)

        question_text = "What is the derivative?"
        input_record = self.state_manager.add_input(
            raw_text=question_text,
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        classification = self.classifier.classify(question_text)
        context = self.context_builder.build(self.state_manager.get_state(), input_record)
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"]["resolution"]["question_type"], "generic_knowledge")
        self.assertIn("rate of change", bundle.response_text.lower())

    def test_claim_can_be_supported_by_stored_fact(self) -> None:
        fact_memory = self.state_manager.add_memory(
            kind="fact",
            content="cube color purple",
            source="knowledge_ingest",
            metadata={
                "subject": "cube",
                "relation": "color",
                "value": "purple",
                "confidence": 0.91,
                "fact_type": "attribute",
                "source_type": "technical_documentation",
            },
        )
        self.memory_store.save_memory(fact_memory)

        claim_text = "The cube is purple."
        input_record = self.state_manager.add_input(
            raw_text=claim_text,
            source="gui",
            input_type="claim",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        classification = ClassificationResult(input_type="claim", confidence=0.9)
        context = self.context_builder.build(self.state_manager.get_state(), input_record)
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"]["reasoning_source"], "claim_supported_by_fact")
        self.assertIn("grounded support", bundle.response_text.lower())

    def test_claim_conflict_is_detected_from_stored_fact(self) -> None:
        fact_memory = self.state_manager.add_memory(
            kind="fact",
            content="cube color purple",
            source="knowledge_ingest",
            metadata={
                "subject": "cube",
                "relation": "color",
                "value": "purple",
                "confidence": 0.91,
                "fact_type": "attribute",
                "source_type": "technical_documentation",
            },
        )
        self.memory_store.save_memory(fact_memory)

        claim_text = "The cube is blue."
        input_record = self.state_manager.add_input(
            raw_text=claim_text,
            source="gui",
            input_type="claim",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        classification = ClassificationResult(input_type="claim", confidence=0.9)
        context = self.context_builder.build(self.state_manager.get_state(), input_record)
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertFalse(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"]["reasoning_source"], "claim_conflicted_by_fact")
        self.assertIn("evidence against", bundle.response_text.lower())


if __name__ == "__main__":
    unittest.main()
