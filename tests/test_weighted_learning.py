from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.fact_learning_engine import FactLearningEngine
from core.input_classifier import InputClassifier
from core.knowledge_memory_bridge import KnowledgeMemoryBridge
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.schemas import MemoryRecord
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngest
from memory.memory_store import MemoryStore


class TestWeightedLearning(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"weighted_learning_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.classifier = InputClassifier()
        self.response_engine = ResponseEngine(
            reasoning_engine=ReasoningEngine(),
            language_renderer=LanguageRenderer(llm_router=None),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_book_ingest_promotes_definition_belief_and_open_question(self) -> None:
        text = (
            "CALCULUS PRIMER\n\n"
            "A derivative is the instantaneous rate of change of a function with respect to a variable.\n\n"
            "An integral is the accumulation of quantity across an interval."
        )
        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=8)
        result = ingest.ingest_text(
            text=text,
            source_path="C:/books/calculus_primer.txt",
            metadata={"title": "Calculus Primer"},
        )
        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)

        self.assertTrue(bridge_result.success)
        self.assertTrue(bridge_result.belief_records)
        derivative_belief = next(
            belief for belief in bridge_result.belief_records
            if belief.metadata.get("subject") == "derivative"
        )
        self.assertEqual(derivative_belief.metadata.get("relation"), "defined_as")
        self.assertGreaterEqual(derivative_belief.confidence_score, 0.18)
        self.assertTrue(bridge_result.open_questions_added)
        self.assertTrue(
            any("independent source confirms that derivative is" in question.lower() for question in bridge_result.open_questions_added)
        )

    def test_response_engine_can_answer_from_learned_belief(self) -> None:
        text = (
            "CALCULUS PRIMER\n\n"
            "A derivative is the instantaneous rate of change of a function with respect to a variable."
        )
        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=8)
        result = ingest.ingest_text(
            text=text,
            source_path="C:/books/calculus_primer.txt",
            metadata={"title": "Calculus Primer"},
        )
        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)

        question_text = "What is a derivative?"
        input_record = self.state_manager.add_input(
            raw_text=question_text,
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=input_record,
            classification=self.classifier.classify(question_text),
            memory_record=None,
            task_record=None,
            context=None,
            prefer_llm=False,
            belief_records=bridge_result.belief_records,
        )

        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"].get("reasoning_source"), "belief_engine")
        self.assertIn("rate of change", bundle.response_text.lower())

    def test_peer_reviewed_source_scores_higher_than_book_for_same_fact(self) -> None:
        engine = FactLearningEngine()
        book_fact = MemoryRecord(
            memory_id="mem_book_fact",
            timestamp="2026-03-19T12:00:00Z",
            kind="fact",
            content="derivative defined_as the instantaneous rate of change",
            source="knowledge_ingest",
            metadata={
                "subject": "derivative",
                "relation": "defined_as",
                "value": "the instantaneous rate of change",
                "confidence": 0.74,
                "fact_type": "definition",
                "source_type": "book",
                "source_path": "C:/books/calculus_primer.txt",
                "title": "Calculus Primer",
                "independence_group": "C:/books/calculus_primer.txt",
            },
        )
        journal_fact = MemoryRecord(
            memory_id="mem_journal_fact",
            timestamp="2026-03-19T12:00:00Z",
            kind="fact",
            content="derivative defined_as the instantaneous rate of change",
            source="knowledge_ingest",
            metadata={
                "subject": "derivative",
                "relation": "defined_as",
                "value": "the instantaneous rate of change",
                "confidence": 0.74,
                "fact_type": "definition",
                "source_type": "peer_reviewed_journal",
                "source_path": "C:/papers/calculus_study.pdf",
                "title": "Calculus Study",
                "independence_group": "doi:10.1000/test",
            },
        )

        book_belief = engine.build_beliefs_from_fact_memories([book_fact])[0]
        journal_belief = engine.build_beliefs_from_fact_memories([journal_fact])[0]
        self.assertGreater(journal_belief.confidence_score, book_belief.confidence_score)

    def test_user_says_alone_is_not_promoted_into_weighted_belief(self) -> None:
        engine = FactLearningEngine()
        user_fact = MemoryRecord(
            memory_id="mem_user_fact",
            timestamp="2026-03-19T12:00:00Z",
            kind="fact",
            content="cube color purple",
            source="gui",
            related_input_id="input_1",
            metadata={
                "subject": "cube",
                "relation": "color",
                "value": "purple",
                "confidence": 0.72,
                "fact_type": "attribute",
                "source_type": "unknown",
            },
        )

        beliefs = engine.build_beliefs_from_fact_memories([user_fact])
        self.assertEqual(beliefs, [])


if __name__ == "__main__":
    unittest.main()
