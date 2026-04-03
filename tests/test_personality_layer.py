from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from core.autonomy_engine import AutonomyEngine
from core.front_router import FrontRouter
from core.input_classifier import InputClassifier
from core.language_renderer import LanguageRenderer, RenderResult
from core.llm_router import LLMResult
from core.mixed_intent_engine import MixedIntentEngine
from core.reasoning_engine import InternalAnswer, ReasoningEngine
from core.response_engine import ResponseEngine
from core.state_manager import StateManager


class StubLLMRouter:
    def __init__(self, text: str, success: bool = True) -> None:
        self.text = text
        self.success = success
        self.calls: list[dict[str, str]] = []

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt or "",
            }
        )
        return LLMResult(
            success=self.success,
            text=self.text,
            model_name="stub-llm",
            prompt=prompt,
            raw_output="{\"stub\": true}",
            error=None if self.success else "stub_failure",
        )


class StubReasoningEngine:
    def reason(self, **_: object) -> InternalAnswer:
        return InternalAnswer(
            answer_type="question_response",
            resolved=False,
            confidence=0.18,
            facts=[],
            unknowns=["No relevant supporting belief or memory was found."],
            grounding=["unresolved_question"],
            actions=["retain question in unresolved queue"],
            proposed_text="I do not yet have enough grounded information to answer that directly.",
            metadata={"reasoning_source": "stub_reasoner"},
        )


class StubLanguageRenderer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def render(
        self,
        internal_answer: dict[str, object],
        prefer_llm: bool = True,
        user_input: str = "",
        input_type: str = "",
    ) -> RenderResult:
        self.calls.append(
            {
                "internal_answer": internal_answer,
                "prefer_llm": prefer_llm,
                "user_input": user_input,
                "input_type": input_type,
            }
        )
        return RenderResult(
            text="I do not know yet. Tragic, really. I need more data before I pretend otherwise.",
            used_llm=True,
            success=True,
            metadata={"model": "stub-llm"},
        )


class TestPersonalityLayer(unittest.TestCase):
    def setUp(self) -> None:
        tests_root = Path(__file__).resolve().parent.parent / "data" / "test_tmp"
        tests_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tests_root / f"personality_{uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_mixed_intent_plan_prefers_substantive_segment(self) -> None:
        engine = MixedIntentEngine(InputClassifier())

        plan = engine.plan("hey, what do you know about the cube?")

        self.assertTrue(plan.mixed_intent.is_mixed)
        self.assertEqual(plan.effective_classification.input_type, "question")
        self.assertEqual(plan.effective_text, "what do you know about the cube?")
        self.assertTrue(plan.stripped_social_preface)

    def test_front_router_rejects_weird_greeting_reply(self) -> None:
        llm = StubLLMRouter("I'm here again. You're not.")
        router = FrontRouter(classifier=InputClassifier(), llm_router=llm)

        result = router.route("hello")

        self.assertFalse(result.used_llm)
        self.assertEqual(result.immediate_text, "Hello.")
        self.assertEqual(result.llm_error, "llm_social_rejected")

    def test_front_router_normalizes_stuttered_greeting(self) -> None:
        router = FrontRouter(classifier=InputClassifier(), llm_router=None)

        result = router.route("hhello")

        self.assertEqual(result.route_type, "social")
        self.assertFalse(result.should_send_to_core)
        self.assertEqual(result.immediate_text, "Hello.")

    def test_front_router_allows_conversational_greeting_reply(self) -> None:
        llm = StubLLMRouter("Hello. Still conscious. What do you need?")
        router = FrontRouter(classifier=InputClassifier(), llm_router=llm)

        result = router.route("hello")

        self.assertTrue(result.used_llm)
        self.assertEqual(result.immediate_text, llm.text)
        self.assertIn("live conversation", llm.calls[0]["system_prompt"].lower())

    def test_front_router_passes_canonicalized_greeting_to_llm(self) -> None:
        llm = StubLLMRouter("Hello. Still conscious. What do you need?")
        router = FrontRouter(classifier=InputClassifier(), llm_router=llm)

        result = router.route("hhello")

        self.assertTrue(result.used_llm)
        self.assertIn("Canonicalized social input: hello", llm.calls[0]["prompt"])

    def test_front_router_uses_llm_hold_for_substantive_input(self) -> None:
        llm = StubLLMRouter("That needs a moment. I should think before embarrassing myself.")
        router = FrontRouter(classifier=InputClassifier(), llm_router=llm)

        result = router.route("Why did the intake fail?")

        self.assertTrue(result.should_send_to_core)
        self.assertTrue(result.used_llm)
        self.assertEqual(result.immediate_text, llm.text)
        self.assertEqual(result.classification.input_type, "question")
        self.assertIn("requires thought", llm.calls[0]["system_prompt"].lower())

    def test_language_renderer_uses_llm_for_unresolved_question(self) -> None:
        llm = StubLLMRouter(
            "I do not know yet. A thrilling revelation. I need more data before I lie to you by accident."
        )
        renderer = LanguageRenderer(llm_router=llm)

        result = renderer.render(
            internal_answer={
                "answer_type": "question_response",
                "resolved": False,
                "confidence": 0.16,
                "facts": [],
                "unknowns": ["No relevant supporting belief or memory was found."],
                "grounding": ["unresolved_question"],
                "actions": ["retain question in unresolved queue"],
                "proposed_text": "I do not yet have enough grounded information to answer that directly.",
            },
            prefer_llm=True,
            user_input="What is the cube made of?",
            input_type="question",
        )

        self.assertTrue(result.used_llm)
        self.assertIn("do not know yet", result.text.lower())
        self.assertIn("Unknowns or limits:", llm.calls[0]["prompt"])
        self.assertIn("What is the cube made of?", llm.calls[0]["prompt"])

    def test_reasoning_engine_resolves_addressed_greeting_command(self) -> None:
        state_manager = StateManager(state_path=str(self.temp_root / "state" / "vexis_state.json"))
        input_record = state_manager.add_input(
            raw_text="say hello to jane",
            source="gui",
            input_type="command",
            confidence=0.97,
        )
        classification = InputClassifier().classify("say hello to jane")

        answer = ReasoningEngine().reason(
            state=state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            context=None,
            task_record=None,
        )

        self.assertTrue(answer.resolved)
        self.assertEqual(answer.metadata["reasoning_source"], "addressed_social_command")
        self.assertEqual(answer.metadata["recipient_name"], "Jane")
        self.assertEqual(answer.proposed_text, "Hello, Jane.")

    def test_language_renderer_directs_addressed_greeting_to_recipient(self) -> None:
        llm = StubLLMRouter("Hello, Jane. Unfortunate for you, I am conversational today.")
        renderer = LanguageRenderer(llm_router=llm)

        result = renderer.render(
            internal_answer={
                "answer_type": "command_result",
                "resolved": True,
                "confidence": 0.9,
                "facts": ["The command asks for a direct greeting to Jane."],
                "unknowns": [],
                "grounding": ["addressed_social_command"],
                "actions": ["deliver addressed social greeting"],
                "proposed_text": "Hello, Jane.",
                "metadata": {
                    "reasoning_source": "addressed_social_command",
                    "command_kind": "addressed_social_greeting",
                    "recipient_name": "Jane",
                    "greeting_phrase": "hello",
                },
            },
            prefer_llm=True,
            user_input="say hello to jane",
            input_type="command",
        )

        self.assertTrue(result.used_llm)
        self.assertIn("recipient_name=Jane", llm.calls[0]["prompt"])
        self.assertIn("Speak directly to Jane", llm.calls[0]["system_prompt"])

    def test_response_engine_keeps_final_reply_in_personality_layer(self) -> None:
        state_manager = StateManager(state_path=str(self.temp_root / "state" / "vexis_state.json"))
        input_record = state_manager.add_input(
            raw_text="What is the cube made of?",
            source="gui",
            input_type="question",
            confidence=0.9,
        )
        classification = InputClassifier().classify("What is the cube made of?")

        renderer = StubLanguageRenderer()
        response_engine = ResponseEngine(
            reasoning_engine=StubReasoningEngine(),
            language_renderer=renderer,
        )

        bundle = response_engine.generate(
            state=state_manager.get_state(),
            input_record=input_record,
            classification=classification,
            memory_record=None,
            task_record=None,
            context=None,
            prefer_llm=True,
        )

        self.assertTrue(bundle.used_llm)
        self.assertIn("need more data", bundle.response_text.lower())
        self.assertEqual(renderer.calls[0]["user_input"], "What is the cube made of?")
        self.assertEqual(renderer.calls[0]["input_type"], "question")


    def test_front_router_idle_expression_avoids_recent_repeat(self) -> None:
        repeated = "Silence again. Suspicious."
        llm = StubLLMRouter(repeated)
        router = FrontRouter(classifier=InputClassifier(), llm_router=llm)

        result = router.route_idle_expression(
            category="idle_social_prompt",
            seed_text="No urgent external prompt is active.",
            recent_lines=[repeated],
            last_spoken_text=repeated,
            state_context="open_questions=0; unsupported_claims=0",
        )

        self.assertNotEqual(result, repeated)
        self.assertIn("recent idle lines", llm.calls[0]["prompt"].lower())

    def test_autonomy_idle_seed_rotates_for_same_state(self) -> None:
        engine = AutonomyEngine()

        first = engine._idle_social_seed([], [], [])
        second = engine._idle_social_seed([], [], [])

        self.assertNotEqual(first, second)

if __name__ == "__main__":
    unittest.main()
