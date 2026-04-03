from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.input_classifier import InputClassifier
from core.knowledge_memory_bridge import KnowledgeMemoryBridge
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from ingest.knowledge_ingest import KnowledgeIngest
from memory.memory_store import MemoryStore


class TestCommandKnowledgeRecall(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"command_recall_{uuid4().hex}"
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

    def test_explain_command_answers_from_uploaded_knowledge(self) -> None:
        text = (
            "BASIC MATHEMATICS\n\n"
            "SERGE LANG\n\n"
            "Foreword\n"
            "In this part we develop systematically the rules for operations with numbers, relations among numbers, addition, multiplication, inequalities, positivity, square roots, and n-th roots.\n\n"
            "Part Two INTUITIVE GEOMETRY\n"
            "This part is concerned with the geometry of the plane. We assume basic properties of straight lines, segments, angles, and distance, and then prove further facts from them."
        )

        ingest = KnowledgeIngest(chunk_target_chars=180, chunk_overlap_chars=40, max_chunks=10)
        result = ingest.ingest_text(
            text=text,
            source_path="C:/books/basic_math.pdf",
            metadata={"title": "basic_math.pdf"},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.title, "BASIC MATHEMATICS")

        bridge = KnowledgeMemoryBridge(self.state_manager, self.memory_store)
        bridge_result = bridge.ingest_result_to_memory(result)
        self.assertTrue(bridge_result.success)

        command_text = "Explain basic mathematics."
        input_record = self.state_manager.add_input(
            raw_text=command_text,
            source="gui",
            input_type="command",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="command",
            content=command_text,
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        classification = self.classifier.classify(command_text)
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

        self.assertTrue(bundle.internal_answer["resolved"])
        self.assertEqual(bundle.internal_answer["metadata"].get("reasoning_source"), "command_knowledge_recall")
        self.assertNotIn("accepted the command", bundle.response_text.lower())
        self.assertTrue(
            "operations with numbers" in bundle.response_text.lower()
            or "geometry of the plane" in bundle.response_text.lower()
        )


if __name__ == "__main__":
    unittest.main()
