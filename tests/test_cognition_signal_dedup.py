from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.cognition_loop import CognitionLoop
from core.state_manager import StateManager
from core.task_goal_engine import TaskGoalEngine
from memory.memory_store import MemoryStore


class TestCognitionSignalDedup(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"cognition_signals_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        base = self.temp_root.as_posix()

        self.state_manager = StateManager(state_path=f"{base}/state/vexis_state.json")
        self.memory_store = MemoryStore(base_dir=f"{base}/memory")
        self.cognition_loop = CognitionLoop(self.state_manager, self.memory_store)
        self.goal_engine = TaskGoalEngine(self.state_manager)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_repeated_background_review_notes_are_suppressed_until_state_changes(self) -> None:
        self.state_manager.add_open_question("Where is here?")
        memory = self.state_manager.add_memory(
            kind="note",
            content="Here refers to the local desktop workspace on this machine.",
            source="gui",
        )
        self.memory_store.save_memory(memory)

        first = self.cognition_loop.run_cycle()
        second = self.cognition_loop.run_cycle()

        self.assertTrue(any("Where is here?" in note for note in first.notes))
        self.assertFalse(any("Where is here?" in note for note in second.notes))

    def test_duplicate_backlog_goal_is_not_recreated_without_new_backlog_state(self) -> None:
        self.state_manager.add_open_question("Where is here?")
        state = self.state_manager.get_state()

        first_goal = self.goal_engine.create_goal_from_open_questions(state)
        second_goal = self.goal_engine.create_goal_from_open_questions(state)

        self.assertIsNotNone(first_goal)
        self.assertIsNone(second_goal)

    def test_background_cycle_closes_resolved_open_question(self) -> None:
        self.state_manager.add_open_question("Where is here?")

        result = self.cognition_loop.run_cycle()
        state = self.state_manager.get_state()

        self.assertIn("Where is here?", result.resolved_questions)
        self.assertNotIn("Where is here?", state.epistemic.open_questions)


if __name__ == "__main__":
    unittest.main()
