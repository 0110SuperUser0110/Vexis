from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.knowledge_memory_bridge import KnowledgeMemoryBridge
from core.input_classifier import InputClassifier
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngest
from memory.memory_store import MemoryStore


class TestKnowledgeIngest(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"knowledge_{uuid4().hex}"
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

    def test_uploaded_knowledge_is_chunked_and_recallable(self) -> None:
        text = (
            "Physics Textbook\n\n"
            "Newton's second law states that force equals mass times acceleration. "
            "This principle explains how acceleration changes when force or mass changes.\n\n"
            "Conservation of energy states that energy is not created or destroyed in an isolated system."
        )

        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=10)
        result = ingest.ingest_text(text=text, source_path="C:/books/physics_textbook.txt", metadata={"title": "Physics Textbook"})
        self.assertTrue(result.success)
        self.assertEqual(result.source_type, "book")
        self.assertTrue(result.chunks)

        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)
        self.assertTrue(bridge_result.success)
        self.assertTrue(any("knowledge chunks" in note for note in bridge_result.notes))

        question_text = "What does Newton's second law say?"
        question_input = self.state_manager.add_input(
            raw_text=question_text,
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        self.memory_store.save_input(question_input)

        classification = self.classifier.classify(question_text)
        context = self.context_builder.build(self.state_manager.get_state(), question_input)
        bundle = self.response_engine.generate(
            state=self.state_manager.get_state(),
            input_record=question_input,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=context,
            prefer_llm=False,
        )

        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertIn("force equals mass times acceleration", bundle.response_text.lower())


if __name__ == "__main__":
    unittest.main()
