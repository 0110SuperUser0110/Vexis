from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.belief_engine import BeliefRecord
from core.context_builder import ContextBuilder
from core.epistemic_bootstrap import BOOTSTRAP_ID, EpistemicBootstrap
from core.fact_learning_engine import FactLearningEngine
from core.inquiry_engine import InquiryEngine
from core.input_classifier import InputClassifier
from core.knowledge_memory_bridge import KnowledgeMemoryBridge
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.schemas import MemoryRecord
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngest
from memory.memory_store import MemoryStore


class TestEpistemicAdvance(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"epistemic_advance_{uuid4().hex}"
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

    def _belief_records(self) -> list[BeliefRecord]:
        beliefs: list[BeliefRecord] = []
        for memory in self.state_manager.get_state().recent_memories:
            if memory.kind != "belief_candidate":
                continue
            metadata = dict(memory.metadata or {})
            beliefs.append(
                BeliefRecord(
                    belief_id=metadata.get("belief_id", f"belief_{memory.memory_id}"),
                    statement=memory.content,
                    confidence_score=float(metadata.get("confidence_score", 0.0)),
                    confidence_label=str(metadata.get("confidence_label", "low")),
                    status=str(metadata.get("status", "candidate")),
                    support_count=int(metadata.get("support_count", 0)),
                    contradiction_count=int(metadata.get("contradiction_count", 0)),
                    supporting_sources=list(metadata.get("supporting_sources", [])),
                    contradicting_sources=list(metadata.get("contradicting_sources", [])),
                    reasons=list(metadata.get("reasons", [])),
                    metadata=metadata,
                )
            )
        return beliefs

    def test_epistemic_bootstrap_seeds_methodology_principles(self) -> None:
        bootstrap = EpistemicBootstrap(self.state_manager, self.memory_store)
        result = bootstrap.ensure_bootstrapped()
        self.assertTrue(result.success)
        self.assertGreater(result.added_count, 0)

        second = bootstrap.ensure_bootstrapped()
        self.assertTrue(second.success)
        self.assertEqual(second.added_count, 0)

        bootstrap_memories = [
            memory for memory in self.memory_store.load_memories()
            if (memory.metadata or {}).get("bootstrap_id") == BOOTSTRAP_ID
        ]
        self.assertTrue(any(memory.kind == "belief_candidate" for memory in bootstrap_memories))
        self.assertTrue(any(memory.kind == "fact" for memory in bootstrap_memories))

        question_text = "Does correlation establish causation?"
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
            belief_records=self._belief_records(),
        )
        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertIn("does not establish causation", bundle.response_text.lower())

    def test_scientific_formula_is_learned_from_textbook_intake(self) -> None:
        text = (
            "PHYSICS PRIMER\n\n"
            "Force equals mass times acceleration.\n\n"
            "Acceleration increases as force increases."
        )
        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=8)
        result = ingest.ingest_text(
            text=text,
            source_path="C:/books/physics_primer.txt",
            metadata={"title": "Physics Primer"},
        )
        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)

        self.assertTrue(any(belief.metadata.get("relation") == "equals_expression" for belief in bridge_result.belief_records))
        self.assertTrue(any(belief.metadata.get("relation") == "increases_with" for belief in bridge_result.belief_records))

        question_text = "What does force equal?"
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
        self.assertIn("mass * acceleration", bundle.response_text.lower())

    def test_conflicting_sources_generate_resolution_question(self) -> None:
        engine = FactLearningEngine()
        inquiry = InquiryEngine()
        fact_a = MemoryRecord(
            memory_id="mem_light_a",
            timestamp="2026-03-19T12:00:00Z",
            kind="fact",
            content="light speed equals 299792458 m/s",
            source="knowledge_ingest",
            metadata={
                "subject": "speed of light",
                "relation": "equals",
                "value": "299792458 m/s",
                "confidence": 0.8,
                "fact_type": "measurement",
                "source_type": "official_record",
                "source_path": "C:/refs/nist.txt",
                "title": "NIST Reference",
                "independence_group": "nist",
            },
        )
        fact_b = MemoryRecord(
            memory_id="mem_light_b",
            timestamp="2026-03-19T12:00:00Z",
            kind="fact",
            content="light speed equals 300000000 m/s",
            source="knowledge_ingest",
            metadata={
                "subject": "speed of light",
                "relation": "equals",
                "value": "300000000 m/s",
                "confidence": 0.7,
                "fact_type": "measurement",
                "source_type": "book",
                "source_path": "C:/books/physics_handbook.txt",
                "title": "Physics Handbook",
                "independence_group": "physics_handbook",
            },
        )

        beliefs = engine.build_beliefs_from_fact_memories([fact_a, fact_b])
        self.assertEqual(len(beliefs), 2)
        self.assertTrue(any(belief.metadata.get("conflicting_values") for belief in beliefs))
        self.assertTrue(all(float(belief.confidence_score) < 0.24 for belief in beliefs))

        questions = inquiry.generate_questions_from_beliefs(beliefs, max_questions=4)
        self.assertTrue(any("resolves whether speed of light equals" in question.lower() for question in questions))


if __name__ == "__main__":
    unittest.main()
