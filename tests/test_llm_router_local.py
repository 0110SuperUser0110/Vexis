from __future__ import annotations

import unittest

from core.llm_router import LLMRouter


class TestLLMRouterLocal(unittest.TestCase):
    def test_router_paths_exist(self) -> None:
        router = LLMRouter()
        self.assertTrue(router.is_available())

    def test_basic_generation(self) -> None:
        router = LLMRouter(timeout_seconds=90, max_tokens=32)
        result = router.generate_response(
            prompt="Respond with exactly: VEXIS local model online."
        )
        self.assertTrue(result.success, msg=result.error or result.raw_output)
        self.assertIn("VEXIS local model online.", result.text)


if __name__ == "__main__":
    unittest.main()