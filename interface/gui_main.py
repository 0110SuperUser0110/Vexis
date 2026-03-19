from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ["QSG_RHI_BACKEND"] = "opengl"

if os.environ.get("QT_QUICK_BACKEND", "").lower() == "software":
    del os.environ["QT_QUICK_BACKEND"]

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cognition_loop import CognitionLoop, CognitionCycleResult
from core.contradiction_engine import ContradictionEngine
from core.context_builder import ContextBundle, ContextBuilder
from core.front_router import FrontRouteResult, FrontRouter
from core.input_classifier import ClassificationResult, InputClassifier
from core.journal_memory_bridge import JournalMemoryBridge
from core.language_renderer import LanguageRenderer
from core.llm_router import LLMRouter
from core.llm_service import LLMService
from core.mixed_intent_engine import MixedIntentEngine, MixedIntentResult
from core.output_gate import OutputDecision, OutputGate
from core.reasoning_engine import ReasoningEngine
from core.response_engine import ResponseBundle, ResponseEngine
from core.self_model import SelfModel
from core.state_manager import StateManager
from core.task_goal_engine import GoalRecord, TaskGoalEngine
from ingest.file_ingest import FileIngest
from ingest.journal_ingest import JournalIngest
from interface.control_window import ControlWindow
from interface.presence_window import PresenceWindow
from memory.memory_store import MemoryStore

try:
    from speech.tts_engine import TTSEngine
except Exception:
    TTSEngine = None  # type: ignore


class SpeechWorker(QObject):
    finished = Signal()
    error = Signal(str)
    started = Signal()

    def __init__(self, tts_engine) -> None:
        super().__init__()
        self.tts = tts_engine

    @Slot(str)
    def speak(self, text: str) -> None:
        if not self.tts:
            self.finished.emit()
            return

        try:
            self.started.emit()
            self.tts.speak(text)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        if not self.tts:
            return
        try:
            self.tts.stop()
        except Exception as exc:
            self.error.emit(str(exc))


class CognitionWorker(QObject):
    cycle_finished = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.state_manager = StateManager()
        self.memory_store = MemoryStore()
        self.cognition_loop = CognitionLoop(
            state_manager=self.state_manager,
            memory_store=self.memory_store,
        )
        self.task_goal_engine = TaskGoalEngine(self.state_manager)
        self.output_gate = OutputGate()

    @Slot()
    def run_cycle(self) -> None:
        try:
            result = self.cognition_loop.run_cycle()
            state = self.state_manager.get_state()

            goals: list[GoalRecord] = []
            open_goal = self.task_goal_engine.create_goal_from_open_questions(state)
            if open_goal:
                goals.append(open_goal)

            claim_goal = self.task_goal_engine.create_goal_from_unsupported_claims(state)
            if claim_goal:
                goals.append(claim_goal)

            payload = {
                "cycle_result": result,
                "goals": goals,
            }
            self.cycle_finished.emit(payload)

        except Exception as exc:
            self.error.emit(f"Cognition worker error: {exc}")


class ResponseWorker(QObject):
    message_finished = Signal(object)
    files_finished = Signal(object)
    immediate_front_response = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self.state_manager = StateManager()
        self.memory_store = MemoryStore()
        self.classifier = InputClassifier()
        self.context_builder = ContextBuilder(self.memory_store)
        self.file_ingest = FileIngest()
        self.contradiction_engine = ContradictionEngine()
        self.task_goal_engine = TaskGoalEngine(self.state_manager)
        self.mixed_intent_engine = MixedIntentEngine(self.classifier)
        self.output_gate = OutputGate()

        self.llm_router = LLMRouter(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=6,
            max_tokens=48,
            temperature=0.05,
        )
        self.front_router = FrontRouter(
            classifier=self.classifier,
            llm_router=self.llm_router,
        )
        self.self_model = SelfModel(self.state_manager)
        self.language_renderer = LanguageRenderer(llm_router=self.llm_router)
        self.reasoning_engine = ReasoningEngine(
            self_model=self.self_model,
            contradiction_engine=self.contradiction_engine,
        )
        self.response_engine = ResponseEngine(
            reasoning_engine=self.reasoning_engine,
            language_renderer=self.language_renderer,
        )
        self.journal_ingest = JournalIngest()
        self.journal_bridge = JournalMemoryBridge(
            state_manager=self.state_manager,
            memory_store=self.memory_store,
        )

    @Slot(str)
    def process_message(self, text: str) -> None:
        try:
            front_route: FrontRouteResult = self.front_router.route(text)
            mixed_intent: MixedIntentResult = self.mixed_intent_engine.analyze(text)

            classification: ClassificationResult = front_route.classification
            core_text = (
                self.mixed_intent_engine.rewrite_for_core(mixed_intent, text)
                if mixed_intent.is_mixed
                else text
            )

            input_record = self.state_manager.add_input(
                raw_text=text,
                source="gui",
                input_type=classification.input_type,
                confidence=classification.confidence,
                metadata={
                    "reasons": classification.reasons,
                    "features": classification.features,
                    "mixed_intent": mixed_intent.to_dict(),
                    "front_route": front_route.to_dict(),
                },
            )
            self.memory_store.save_input(input_record)

            memory_record = self.state_manager.add_memory(
                kind=classification.input_type,
                content=text,
                source="gui",
                related_input_id=input_record.input_id,
                status="active",
                metadata={
                    "confidence": classification.confidence,
                    "reasons": classification.reasons,
                    "mixed_intent": mixed_intent.to_dict(),
                },
            )
            self.memory_store.save_memory(memory_record)

            if front_route.immediate_text:
                self.immediate_front_response.emit(front_route.immediate_text)

            if classification.input_type == "social" and not front_route.should_send_to_core:
                bundle = ResponseBundle(
                    response_text=front_route.immediate_text,
                    internal_answer={
                        "answer_type": "social_response",
                        "resolved": True,
                        "confidence": classification.confidence,
                        "facts": ["Handled by front router social lane."],
                        "unknowns": [],
                        "grounding": ["front_router"],
                        "actions": ["store social interaction"],
                        "proposed_text": front_route.immediate_text,
                        "metadata": {"front_route": front_route.to_dict()},
                    },
                    memory_record=memory_record,
                    task_record=None,
                    status_text="social response ready",
                    visual_state="speaking",
                    epistemic_updates={},
                    used_llm=front_route.used_llm,
                    render_error=front_route.llm_error,
                    immediate_front_text=front_route.immediate_text,
                    metadata={
                        "front_route": front_route.to_dict(),
                        "mixed_intent": mixed_intent.to_dict(),
                    },
                )
                payload = {
                    "text": text,
                    "classification_type": classification.input_type,
                    "classification_confidence": classification.confidence,
                    "bundle": bundle,
                    "input_record_id": input_record.input_id,
                    "front_route": front_route.to_dict(),
                    "mixed_intent": mixed_intent.to_dict(),
                }
                self.message_finished.emit(payload)
                return

            task_result = self.task_goal_engine.handle_user_input(
                input_record=input_record,
                memory_record=memory_record,
            )
            task_record = task_result.created_tasks[0] if task_result.created_tasks else None

            core_input_record = input_record
            if core_text != text:
                core_input_record = self.state_manager.add_input(
                    raw_text=core_text,
                    source="gui_mixed_intent_core_slice",
                    input_type=mixed_intent.dominant_intent,
                    confidence=classification.confidence,
                    metadata={
                        "parent_input_id": input_record.input_id,
                        "origin": "mixed_intent_rewrite",
                    },
                    interaction_context={
                        **input_record.interaction_context,
                        "interaction_type": mixed_intent.dominant_intent,
                        "personality_allowed": False,
                    },
                )
                self.memory_store.save_input(core_input_record)

            context: ContextBundle = self.context_builder.build(
                self.state_manager.get_state(),
                core_input_record,
            )

            belief_records = self._belief_records_from_recent_memories()

            bundle: ResponseBundle = self.response_engine.generate(
                state=self.state_manager.get_state(),
                input_record=core_input_record,
                classification=classification,
                memory_record=memory_record,
                task_record=task_record,
                context=context,
                prefer_llm=True,
                front_route=front_route,
                mixed_intent=mixed_intent,
                belief_records=belief_records,
            )

            payload = {
                "text": text,
                "classification_type": classification.input_type,
                "classification_confidence": classification.confidence,
                "bundle": bundle,
                "input_record_id": input_record.input_id,
                "front_route": front_route.to_dict(),
                "mixed_intent": mixed_intent.to_dict(),
                "task_notes": task_result.notes,
            }
            self.message_finished.emit(payload)

        except Exception as exc:
            self.error.emit(f"Response worker message error: {exc}")

    @Slot(list)
    def process_files(self, files: list[str]) -> None:
        try:
            ingest_results = self.file_ingest.ingest_files(files)
            system_lines: list[str] = []
            success_count = 0
            failed_count = 0

            for result in ingest_results:
                self.state_manager.queue_file_for_ingest(result.filepath)
                self.state_manager.mark_file_processing(result.filepath)

                if result.success:
                    success_count += 1
                else:
                    failed_count += 1

                input_record = self.state_manager.add_input(
                    raw_text=f"file added: {result.filename}",
                    source="file_ingest",
                    input_type="command",
                    confidence=1.0 if result.success else 0.2,
                    metadata={
                        "filepath": result.filepath,
                        "file_type": result.file_type,
                        "success": result.success,
                        "error": result.error,
                    },
                )
                self.memory_store.save_input(input_record)

                if result.success and result.file_type in {"text", "json", "csv"}:
                    journal_result = self.journal_ingest.ingest_text(
                        text=result.extracted_text,
                        source_path=result.filepath,
                        metadata={
                            "title": result.filename,
                            "source_type": "peer_reviewed_journal",
                        },
                    )
                    if journal_result.success:
                        bridge_result = self.journal_bridge.ingest_result_to_memory(journal_result)
                        system_lines.extend(bridge_result.notes)
                    preview = result.extracted_text[:220].replace("\n", " ")
                    system_lines.append(
                        f"ingested {result.filename} [{result.file_type}] | preview: {preview}"
                    )
                elif result.success:
                    system_lines.append(
                        f"registered {result.filename} [{result.file_type}] | parser not yet implemented"
                    )
                else:
                    system_lines.append(
                        f"failed to ingest {result.filename}: {result.error}"
                    )

                self.state_manager.mark_file_completed(result.filepath)

            response_text = (
                f"I completed intake for {success_count} file or files."
                if failed_count == 0
                else f"I completed intake for {success_count} file or files, with {failed_count} failures."
            )

            payload = {
                "response_text": response_text,
                "system_lines": system_lines,
                "success_count": success_count,
                "failed_count": failed_count,
            }
            self.files_finished.emit(payload)

        except Exception as exc:
            self.error.emit(f"Response worker file error: {exc}")

    def _belief_records_from_recent_memories(self) -> list:
        beliefs = []
        for memory in self.state_manager.get_state().recent_memories:
            if memory.kind == "belief_candidate":
                metadata = memory.metadata or {}
                try:
                    from core.belief_engine import BeliefRecord
                    beliefs.append(
                        BeliefRecord(
                            belief_id=metadata.get("belief_id", f"belief_{memory.memory_id}"),
                            statement=memory.content,
                            confidence_score=float(metadata.get("confidence_score", 0.25)),
                            confidence_label=metadata.get("confidence_label", "low"),
                            status=metadata.get("status", "candidate"),
                            support_count=int(metadata.get("support_count", 1)),
                            contradiction_count=int(metadata.get("contradiction_count", 0)),
                            supporting_sources=metadata.get("supporting_sources", []),
                            contradicting_sources=metadata.get("contradicting_sources", []),
                            reasons=metadata.get("reasons", []),
                            metadata=metadata,
                        )
                    )
                except Exception:
                    continue
        return beliefs


class VexisGuiController(QObject):
    request_speech = Signal(str)
    request_stop_speech = Signal()
    request_process_message = Signal(str)
    request_process_files = Signal(list)
    request_cognition_cycle = Signal()

    def __init__(self, app: QApplication, control: ControlWindow, presence: PresenceWindow) -> None:
        super().__init__()
        self.app = app
        self.control = control
        self.presence = presence

        self.state_manager = StateManager()
        self.output_gate = OutputGate()
        self.llm_service = LLMService()

        self.tts = None
        self.speech_thread: QThread | None = None
        self.speech_worker: SpeechWorker | None = None
        self._speech_active = False

        self.response_thread: QThread | None = None
        self.response_worker: ResponseWorker | None = None
        self._busy = False

        self.cognition_thread: QThread | None = None
        self.cognition_worker: CognitionWorker | None = None

        self._shutting_down = False

        self._boot_state()
        self._setup_response_thread()
        self._setup_cognition_thread()
        self._setup_cognition_timer()

        if TTSEngine is not None:
            try:
                self.tts = TTSEngine("config/voices.json")
                self._setup_speech_thread()
            except Exception as exc:
                self.control.append_response("system", f"Voice initialization failed: {exc}")

        self.control.user_message.connect(self.handle_user_message)
        self.control.files_added.connect(self.handle_files_added)
        self.control.state_changed.connect(self._external_state_change)
        self.control.shutdown_requested.connect(self.shutdown_all)

    def _boot_state(self) -> None:
        boot_count = self.state_manager.increment_boot_count()

        self.control.set_status("VEXIS active | starting model")
        self.presence.set_state("thinking")
        self.presence.set_thought_lines(
            [
                "initializing core",
                "starting persistent llm service",
                "loading qwen model",
            ]
        )

        self.llm_service.start()

        self.state_manager.set_visual_state("idle", "ready")
        state = self.state_manager.get_state()

        self.control.set_status(f"VEXIS active | ready | boot {boot_count}")
        self.presence.set_state(state.presence.visual_state)
        self.presence.set_thought_lines(
            [
                "vexis core initialized",
                f"boot count: {boot_count}",
                "awaiting input",
            ]
        )

    def _setup_response_thread(self) -> None:
        self.response_thread = QThread()
        self.response_worker = ResponseWorker()
        self.response_worker.moveToThread(self.response_thread)

        self.request_process_message.connect(self.response_worker.process_message)
        self.request_process_files.connect(self.response_worker.process_files)
        self.response_worker.immediate_front_response.connect(self._on_immediate_front_response)
        self.response_worker.message_finished.connect(self._on_message_finished)
        self.response_worker.files_finished.connect(self._on_files_finished)
        self.response_worker.error.connect(self._on_worker_error)

        self.response_thread.start()

    def _setup_cognition_thread(self) -> None:
        self.cognition_thread = QThread()
        self.cognition_worker = CognitionWorker()
        self.cognition_worker.moveToThread(self.cognition_thread)

        self.request_cognition_cycle.connect(self.cognition_worker.run_cycle)
        self.cognition_worker.cycle_finished.connect(self._on_cognition_cycle_finished)
        self.cognition_worker.error.connect(self._on_worker_error)

        self.cognition_thread.start()

    def _setup_cognition_timer(self) -> None:
        self.cognition_timer = QTimer()
        self.cognition_timer.setInterval(15000)
        self.cognition_timer.timeout.connect(self._trigger_cognition_cycle)
        self.cognition_timer.start()

    def _trigger_cognition_cycle(self) -> None:
        if not self._shutting_down:
            self.request_cognition_cycle.emit()

    def _setup_speech_thread(self) -> None:
        if self.tts is None:
            return

        self.speech_thread = QThread()
        self.speech_worker = SpeechWorker(self.tts)
        self.speech_worker.moveToThread(self.speech_thread)

        self.request_speech.connect(self.speech_worker.speak)
        self.request_stop_speech.connect(self.speech_worker.stop)
        self.speech_worker.started.connect(self._on_speech_started)
        self.speech_worker.finished.connect(self._on_speech_finished)
        self.speech_worker.error.connect(self._on_speech_error)

        self.speech_thread.start()

    @Slot()
    def _on_speech_started(self) -> None:
        self._speech_active = True
        self.state_manager.set_speaking(True)
        self.state_manager.set_visual_state("speaking", "speaking")

        self.control.set_status("VEXIS active | speaking")
        self.presence.set_state("speaking")
        self.presence.set_thought_lines(
            [
                "voice output active",
                "speech thread running",
                "maintaining live motion",
            ]
        )

    @Slot()
    def _on_speech_finished(self) -> None:
        self._speech_active = False
        self.state_manager.set_speaking(False)
        self.state_manager.set_visual_state("idle", "ready")

        if not self._busy and not self._shutting_down:
            self.control.set_status("VEXIS active | ready")
            self.presence.set_state("idle")

    @Slot(str)
    def _on_speech_error(self, message: str) -> None:
        self.control.append_response("system", f"Speech error: {message}")

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self._busy = False
        self.state_manager.set_visual_state("idle", "ready")
        self.state_manager.set_thinking(False, current_focus="worker error")
        self.control.set_status("VEXIS active | error")
        self.presence.set_state("idle")
        self.presence.set_thought_lines(
            [
                "worker failure",
                "see system log",
            ]
        )
        self.control.append_response("system", message)

    def _speak_async(self, text: str) -> None:
        if self.speech_worker is None or self._shutting_down:
            return
        self.request_speech.emit(text)

    @Slot(str)
    def _on_immediate_front_response(self, text: str) -> None:
        if not text or self._shutting_down:
            return
        self.control.append_response("VEXIS", text)

    def _external_state_change(self, state: str) -> None:
        self.state_manager.set_visual_state(state, state)
        self.presence.set_state(state)

    def _apply_epistemic_updates(self, updates: dict[str, list[str]]) -> None:
        for question in updates.get("open_questions", []):
            self.state_manager.add_open_question(question)

        for claim in updates.get("open_claims", []):
            self.state_manager.add_open_claim(claim)

        for claim in updates.get("unsupported_claims", []):
            self.state_manager.add_unsupported_claim(claim)

        for contradiction in updates.get("contradictions", []):
            self.state_manager.add_contradiction(contradiction)

        extracted_facts = updates.get("extracted_facts", [])
        for packed in extracted_facts:
            try:
                subject, relation, value = packed.split("|", 2)
            except ValueError:
                continue

            fact_text = f"{subject} {relation} {value}"
            fact_memory = self.state_manager.add_memory(
                kind="fact",
                content=fact_text,
                source="reasoning_engine",
                status="active",
                metadata={
                    "subject": subject,
                    "relation": relation,
                    "value": value,
                    "source_type": "internal_extracted_fact",
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_fact_store",
                    "source": "reasoning_engine",
                    "interaction_type": "fact",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )

            try:
                if self.response_worker is not None:
                    self.response_worker.memory_store.save_memory(fact_memory)
            except Exception:
                pass

        resolved_questions = updates.get("resolved_questions", [])
        for question in resolved_questions:
            resolved_memory = self.state_manager.add_memory(
                kind="resolved_question",
                content=question,
                source="resolution_engine",
                status="resolved",
                metadata={
                    "source_type": "internal_resolution",
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_resolution_engine",
                    "source": "resolution_engine",
                    "interaction_type": "resolved_question",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )

            try:
                if self.response_worker is not None:
                    self.response_worker.memory_store.save_memory(resolved_memory)
            except Exception:
                pass

    def handle_user_message(self, text: str) -> None:
        if self._busy or self._shutting_down:
            self.control.append_response("system", "VEXIS is still processing the previous request.")
            return

        self.output_gate.note_user_activity()
        self._busy = True
        self.state_manager.set_visual_state("thinking", "thinking")
        self.state_manager.set_thinking(True, current_focus="input analysis")

        self.control.set_status("VEXIS active | thinking")
        self.presence.set_state("thinking")
        self.presence.set_thought_lines(
            [
                "parsing input",
                "routing front layer",
                "preparing core review",
                text[:42],
            ]
        )

        self.request_process_message.emit(text)

    def handle_files_added(self, files: list[str]) -> None:
        if self._busy or self._shutting_down:
            self.control.append_response("system", "VEXIS is still processing the previous request.")
            return

        self.output_gate.note_user_activity()
        self._busy = True
        self.state_manager.set_visual_state("thinking", "reviewing files")
        self.state_manager.set_thinking(True, current_focus="file ingest")

        short = [Path(f).name for f in files]
        self.control.set_status("VEXIS active | ingesting files")
        self.presence.set_state("thinking")
        self.presence.set_thought_lines(
            [
                "new files received",
                *short[:6],
                "extracting contents",
            ]
        )

        self.request_process_files.emit(files)

    @Slot(object)
    def _on_message_finished(self, payload: Any) -> None:
        if self._shutting_down:
            return

        self._busy = False

        bundle: ResponseBundle = payload["bundle"]
        classification_type: str = payload["classification_type"]
        classification_confidence: float = payload["classification_confidence"]
        input_record_id: str = payload["input_record_id"]

        self._apply_epistemic_updates(bundle.epistemic_updates)

        self.state_manager.set_thinking(
            False,
            current_focus=bundle.status_text,
            current_input_id=input_record_id,
            last_classification=classification_type,
        )
        self.state_manager.set_visual_state("thinking", bundle.status_text)

        self.control.append_response(
            "system",
            f"classified as {classification_type} (confidence {classification_confidence:.2f})",
        )

        if bundle.internal_answer:
            internal = bundle.internal_answer
            resolved = internal.get("resolved", False)
            confidence = float(internal.get("confidence", 0.0))
            self.control.append_response(
                "system",
                f"reasoning: resolved={resolved} confidence={confidence:.2f}",
            )

        if bundle.used_llm:
            self.control.append_response("system", "language layer: local qwen active")
        elif bundle.render_error:
            self.control.append_response("system", f"language render fallback: {bundle.render_error}")
            llm_debug = bundle.internal_answer.get("metadata", {}).get("llm_debug_text") if bundle.internal_answer else None
            if llm_debug:
                self.control.append_response("system", f"llm debug: {llm_debug}")
        else:
            self.control.append_response("system", "language layer: deterministic renderer active")

        if not bundle.immediate_front_text or bundle.response_text != bundle.immediate_front_text:
            self.control.append_response("VEXIS", bundle.response_text)

        self.presence.set_thought_lines(
            [
                f"type: {classification_type}",
                f"confidence: {classification_confidence:.2f}",
                "reasoning complete",
                "queueing speech",
            ]
        )

        decision: OutputDecision = self.output_gate.evaluate_direct_response(
            text=bundle.response_text,
            state=self.state_manager.get_state(),
        )
        if decision.allow_speech_output and bundle.response_text and "â–" not in bundle.response_text:
            self._speak_async(bundle.response_text)
            self.output_gate.note_spoken_output()

        if self.speech_worker is None:
            self.control.set_status("VEXIS active | ready")
            self.state_manager.set_visual_state("idle", "ready")
            self.presence.set_state("idle")

    @Slot(object)
    def _on_files_finished(self, payload: Any) -> None:
        if self._shutting_down:
            return

        self._busy = False

        for line in payload["system_lines"]:
            self.control.append_response("system", line)

        self.control.append_response("VEXIS", payload["response_text"])

        self.state_manager.set_thinking(False, current_focus="file intake complete")
        self.state_manager.set_visual_state("idle", "ready")

        self.presence.set_thought_lines(
            [
                f"files complete: {payload['success_count']}",
                f"failures: {payload['failed_count']}",
                "queueing speech",
            ]
        )

        decision = self.output_gate.evaluate_direct_response(
            text=payload["response_text"],
            state=self.state_manager.get_state(),
        )
        if decision.allow_speech_output and payload["response_text"]:
            self._speak_async(payload["response_text"])
            self.output_gate.note_spoken_output()

        if self.speech_worker is None:
            self.control.set_status("VEXIS active | ready")
            self.presence.set_state("idle")

    @Slot(object)
    def _on_cognition_cycle_finished(self, payload: Any) -> None:
        if self._shutting_down:
            return

        cycle_result: CognitionCycleResult = payload["cycle_result"]
        goals: list[GoalRecord] = payload["goals"]

        for note in cycle_result.notes[:3]:
            self.control.append_response("system", note)

        for goal in goals[:2]:
            self.control.append_response("system", f"goal created: {goal.title}")

        for action in cycle_result.actions:
            decision = self.output_gate.evaluate_action(
                action=action,
                state=self.state_manager.get_state(),
            )
            if decision.allow_gui_output and decision.text:
                self.control.append_response("VEXIS", decision.text)

            if decision.allow_speech_output and decision.text:
                self._speak_async(decision.text)
                self.output_gate.note_spoken_output()

    def shutdown_all(self) -> None:
        if self._shutting_down:
            return

        self._shutting_down = True

        try:
            self.request_stop_speech.emit()
        except Exception:
            pass

        try:
            if hasattr(self, "cognition_timer") and self.cognition_timer is not None:
                self.cognition_timer.stop()
                self.cognition_timer.deleteLater()
                self.cognition_timer = None
        except Exception:
            pass

        try:
            if self.speech_thread is not None:
                self.speech_thread.quit()
                self.speech_thread.wait(3000)
                self.speech_thread.deleteLater()
                self.speech_thread = None
        except Exception:
            pass

        try:
            if self.response_thread is not None:
                self.response_thread.quit()
                self.response_thread.wait(3000)
                self.response_thread.deleteLater()
                self.response_thread = None
        except Exception:
            pass

        try:
            if self.cognition_thread is not None:
                self.cognition_thread.quit()
                self.cognition_thread.wait(3000)
                self.cognition_thread.deleteLater()
                self.cognition_thread = None
        except Exception:
            pass

        try:
            self.llm_service.stop()
        except Exception:
            pass

        try:
            self.presence.close()
        except Exception:
            pass

        try:
            self.control.close()
        except Exception:
            pass

        try:
            self.app.quit()
        except Exception:
            pass


def main() -> None:
    app = QApplication(sys.argv)

    control = ControlWindow()
    presence = PresenceWindow()

    control.move(80, 120)
    presence.setPosition(900, 100)

    controller = VexisGuiController(app, control, presence)
    app.aboutToQuit.connect(controller.shutdown_all)

    control.show()
    presence.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()