from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.runtime_repair import repair_runtime_state
from core.state_manager import StateManager
from memory.memory_store import MemoryStore


class TestRuntimeRepair(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"runtime_repair_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_repair_closes_resolved_questions_and_purges_invalid_self_facts(self) -> None:
        self.state_manager.add_open_question("Where is here?")
        self.state_manager.add_open_question("what color is your cube?")

        invalid_fact = self.state_manager.add_memory(
            kind="fact",
            content="vex running_on PATTERN",
            source="knowledge_ingest",
        )
        self.memory_store.save_memory(invalid_fact)

        report = repair_runtime_state(self.state_manager, self.memory_store)
        state = self.state_manager.get_state()
        memories = self.memory_store.load_memories()

        self.assertTrue(report.changed)
        self.assertIn("Where is here?", report.removed_open_questions)
        self.assertIn("what color is your cube?", report.removed_open_questions)
        self.assertIn(invalid_fact.memory_id, report.removed_invalid_memory_ids)
        self.assertIn("what color is your cube?", report.added_resolved_questions)
        self.assertNotIn("Where is here?", state.epistemic.open_questions)
        self.assertNotIn("what color is your cube?", state.epistemic.open_questions)
        self.assertFalse(any(memory.memory_id == invalid_fact.memory_id for memory in memories))
        self.assertTrue(
            any(memory.kind == "resolved_question" and memory.content == "what color is your cube?" for memory in memories)
        )


if __name__ == "__main__":
    unittest.main()
