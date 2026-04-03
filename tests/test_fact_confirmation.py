from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.context_builder import ContextBuilder
from core.input_classifier import InputClassifier
from core.language_renderer import LanguageRenderer
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


class TestFactConfirmation(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"fact_confirm_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.classifier = InputClassifier()
        self.context_builder = ContextBuilder(self.memory_store)
        self.response_engine = ResponseEngine(
            reasoning_engine=ReasoningEngine(),
            language_renderer=LanguageRenderer(llm_router=None),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_user_assertion_requests_fact_confirmation_and_proof(self) -> None:
        text = "The cube is purple."
        input_record = self.state_manager.add_input(
            raw_text=text,
            source="gui",
            input_type="note",
            confidence=0.9,
        )
        self.memory_store.save_input(input_record)

        memory_record = self.state_manager.add_memory(
            kind="note",
            content=text,
            source="gui",
            related_input_id=input_record.input_id,
        )
        self.memory_store.save_memory(memory_record)

        classification = self.classifier.classify(text)
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

        self.assertEqual(bundle.status_text, "note stored pending fact verification")
        self.assertFalse(bundle.internal_answer["resolved"])
        self.assertTrue(bundle.internal_answer["metadata"].get("requires_fact_confirmation"))
        self.assertTrue(bundle.internal_answer["metadata"].get("requires_fact_proof"))
        self.assertIn("proof", bundle.response_text.lower())
        self.assertNotIn("extracted_facts", bundle.epistemic_updates)
        self.assertTrue(any("fact candidate" in question.lower() for question in bundle.epistemic_updates.get("open_questions", [])))


if __name__ == "__main__":
    unittest.main()
