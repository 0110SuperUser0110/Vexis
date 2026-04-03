from __future__ import annotations

import json

import os
import sys
import threading
import traceback
from datetime import datetime
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

RUNTIME_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "controller_runtime.log"


def _append_runtime_log(level: str, message: str) -> None:
    try:
        RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{level}] {message.rstrip()}\n")
    except Exception:
        pass


def _install_exception_logging() -> None:
    previous_sys_excepthook = sys.excepthook

    def handle_sys_exception(exc_type, exc_value, exc_traceback) -> None:
        formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _append_runtime_log("ERROR", f"Unhandled exception\n{formatted}")
        previous_sys_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_sys_exception

    if hasattr(threading, "excepthook"):
        previous_thread_excepthook = threading.excepthook

        def handle_thread_exception(args) -> None:
            thread_name = args.thread.name if args.thread is not None else "unknown"
            formatted = "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
            _append_runtime_log("ERROR", f"Unhandled thread exception [{thread_name}]\n{formatted}")
            previous_thread_excepthook(args)

        threading.excepthook = handle_thread_exception

from core.cognition_loop import CognitionLoop, CognitionCycleResult
from core.epistemic_bootstrap import EpistemicBootstrap
from core.contradiction_engine import ContradictionEngine
from core.context_builder import ContextBundle, ContextBuilder
from core.front_router import FrontRouteResult, FrontRouter
from core.input_classifier import ClassificationResult, InputClassifier
from core.journal_memory_bridge import JournalMemoryBridge
from core.knowledge_memory_bridge import KnowledgeBridgeResult, KnowledgeMemoryBridge
from core.language_renderer import LanguageRenderer
from core.llm_router import LLMRouter
from core.llm_service import LLMService
from core.mixed_intent_engine import MixedIntentEngine, MixedIntentResult
from core.output_gate import OutputDecision, OutputGate
from core.reasoning_engine import ReasoningEngine
from core.runtime_repair import repair_runtime_state
from core.response_engine import ResponseBundle, ResponseEngine
from core.self_model import SelfModel
from core.state_manager import StateManager
from core.task_goal_engine import GoalRecord, TaskGoalEngine
from ingest.file_ingest import FileIngest
from ingest.journal_ingest import JournalIngest
from ingest.knowledge_ingest import KnowledgeIngest, KnowledgeIngestResult
from interface.control_window import ControlWindow
from interface.presence_window import NullPresenceWindow, PresenceWindow
from memory.memory_store import MemoryStore
from interface.unreal_bridge import UnrealPresenceBridgeServer

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
            _append_runtime_log("ERROR", f"SpeechWorker.speak failed\n{traceback.format_exc()}")
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
            _append_runtime_log("ERROR", f"SpeechWorker.speak failed\n{traceback.format_exc()}")
            self.error.emit(str(exc))


class CognitionWorker(QObject):
    cycle_finished = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.state_manager = StateManager()
        self.memory_store = MemoryStore()
        self.epistemic_bootstrap = EpistemicBootstrap(self.state_manager, self.memory_store)
        self.epistemic_bootstrap.ensure_bootstrapped()
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
            _append_runtime_log("ERROR", f"CognitionWorker.run_cycle failed\n{traceback.format_exc()}")
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
        self.epistemic_bootstrap = EpistemicBootstrap(self.state_manager, self.memory_store)
        self.epistemic_bootstrap.ensure_bootstrapped()
        self.classifier = InputClassifier()
        self.context_builder = ContextBuilder(self.memory_store)
        self.file_ingest = FileIngest()
        self.contradiction_engine = ContradictionEngine()
        self.task_goal_engine = TaskGoalEngine(self.state_manager)
        self.mixed_intent_engine = MixedIntentEngine(self.classifier)
        self.output_gate = OutputGate()

        self.front_llm_router = LLMRouter(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=8,
            max_tokens=64,
            temperature=0.6,
        )
        self.render_llm_router = LLMRouter(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=14,
            max_tokens=140,
            temperature=0.4,
        )
        self.front_router = FrontRouter(
            classifier=self.classifier,
            llm_router=self.front_llm_router,
        )
        self.self_model = SelfModel(self.state_manager)
        self.language_renderer = LanguageRenderer(llm_router=self.render_llm_router)
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
        self.knowledge_ingest = KnowledgeIngest()
        self.knowledge_bridge = KnowledgeMemoryBridge(
            state_manager=self.state_manager,
            memory_store=self.memory_store,
        )

    @Slot(str)
    def process_message(self, text: str) -> None:
        try:
            routing_plan = self.mixed_intent_engine.plan(text)
            mixed_intent: MixedIntentResult = routing_plan.mixed_intent
            classification: ClassificationResult = routing_plan.effective_classification
            core_text = routing_plan.effective_text or text
            front_route: FrontRouteResult = self.front_router.route(
                core_text,
                classification=classification,
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
                    "original_classification": self._classification_to_dict(
                        routing_plan.original_classification,
                    ),
                    "effective_classification": self._classification_to_dict(classification),
                    "core_text": core_text,
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
                    "original_classification": self._classification_to_dict(
                        routing_plan.original_classification,
                    ),
                    "effective_classification": self._classification_to_dict(classification),
                    "core_text": core_text,
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
                        "original_classification": self._classification_to_dict(
                            routing_plan.original_classification,
                        ),
                        "effective_classification": self._classification_to_dict(classification),
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
            _append_runtime_log("ERROR", f"ResponseWorker.process_message failed\n{traceback.format_exc()}")
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

                if result.success and result.file_type in {"text", "json", "csv", "pdf", "image"}:
                    knowledge_result: KnowledgeIngestResult = self.knowledge_ingest.ingest_text(
                        text=result.extracted_text,
                        source_path=result.filepath,
                        metadata={
                            "title": result.filename,
                        },
                    )
                    if knowledge_result.success and result.extracted_text.strip():
                        knowledge_bridge_result: KnowledgeBridgeResult = self.knowledge_bridge.ingest_result_to_memory(knowledge_result)
                        system_lines.extend(knowledge_bridge_result.notes)

                        if knowledge_result.source_type == "peer_reviewed_journal":
                            journal_result = self.journal_ingest.ingest_text(
                                text=result.extracted_text,
                                source_path=result.filepath,
                                metadata={
                                    "title": result.filename,
                                    "source_type": knowledge_result.source_type,
                                },
                            )
                            if journal_result.success:
                                bridge_result = self.journal_bridge.ingest_result_to_memory(journal_result)
                                system_lines.extend(bridge_result.notes)
                    elif knowledge_result.success:
                        system_lines.append(
                            f"parsed {result.filename} [{result.file_type}] but no readable text was extracted"
                        )

                    preview = result.extracted_text[:220].replace("\n", " ")
                    system_lines.append(
                        f"ingested {result.filename} [{result.file_type}] as {knowledge_result.source_type if knowledge_result.success else 'unknown'} | preview: {preview}"
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
            _append_runtime_log("ERROR", f"ResponseWorker.process_files failed\n{traceback.format_exc()}")
            self.error.emit(f"Response worker file error: {exc}")

    def _classification_to_dict(
        self,
        classification: ClassificationResult,
    ) -> dict[str, Any]:
        return {
            "input_type": classification.input_type,
            "confidence": classification.confidence,
            "reasons": classification.reasons,
            "features": classification.features,
        }

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
        self.memory_store = MemoryStore()
        self.epistemic_bootstrap = EpistemicBootstrap(self.state_manager, self.memory_store)
        self.bootstrap_result = self.epistemic_bootstrap.ensure_bootstrapped()
        self.runtime_repair = repair_runtime_state(self.state_manager, self.memory_store)
        if self.runtime_repair.changed:
            _append_runtime_log(
                "INFO",
                "Runtime repair removed "
                f"{len(self.runtime_repair.removed_invalid_memory_ids)} invalid memories and "
                f"closed {len(self.runtime_repair.removed_open_questions)} open questions.",
            )
        self.output_gate = OutputGate()
        self.llm_service = LLMService()
        self.idle_llm_router = LLMRouter(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=8,
            max_tokens=48,
            temperature=0.85,
        )
        self.idle_front_router = FrontRouter(
            classifier=InputClassifier(),
            llm_router=self.idle_llm_router,
        )
        self._recent_idle_lines: list[str] = []
        self._active_presence_action = ""
        self._queued_presence_action = ""
        self._queued_presence_action_duration_ms = 0
        self._presence_action_timer = QTimer(self)
        self._presence_action_timer.setSingleShot(True)
        self._presence_action_timer.timeout.connect(self._clear_presence_action)
        self._presence_state = "idle"
        self._presence_thought_lines: list[str] = ["VEXIS online", "Awaiting input"]
        self._last_response_text = ""
        bridge_port = int(os.environ.get("VEXIS_UNREAL_BRIDGE_PORT", "8765"))
        self.unreal_bridge = UnrealPresenceBridgeServer(port=bridge_port)
        self.unreal_bridge.start()

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
        self._boot_in_progress = True
        self._boot_count = 0
        self._publish_unreal_snapshot()

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
        self.control.shutdown_requested.connect(self.shutdown_all)
        self._start_boot_sequence()

    def _start_boot_sequence(self) -> None:
        self._boot_count = self.state_manager.increment_boot_count()

        self.control.set_status("VEXIS active | starting model")
        self._set_presence_state("thinking")
        self._set_presence_thought_lines(
            [
                "initializing core",
                "loading epistemic substrate",
                "starting persistent llm service",
                "loading qwen model",
            ]
        )
        self._publish_unreal_snapshot()
        QTimer.singleShot(0, self._complete_boot_sequence)

    @Slot()
    def _complete_boot_sequence(self) -> None:
        try:
            self.llm_service.start()
        except Exception as exc:
            self._boot_in_progress = False
            self.state_manager.set_visual_state("idle", "model startup failure")
            self.state_manager.set_thinking(False, current_focus="model startup failure")
            self.control.set_status("VEXIS active | model startup failed")
            self._set_presence_state("idle")
            self._set_presence_thought_lines(
                [
                    "model startup failed",
                    str(exc)[:96],
                    "check local llm service configuration",
                ]
            )
            self.control.append_response("system", f"Model startup failed: {exc}")
            self._publish_unreal_snapshot()
            return

        self.state_manager.set_visual_state("idle", "ready")
        state = self.state_manager.get_state()
        self._boot_in_progress = False

        self.control.set_status(f"VEXIS active | ready | boot {self._boot_count}")
        self._set_presence_state(state.presence.visual_state)
        self._set_presence_thought_lines(
            [
                "vexis core initialized",
                f"boot count: {self._boot_count}",
                self.bootstrap_result.notes[0] if getattr(self, "bootstrap_result", None) and self.bootstrap_result.notes else "epistemic substrate ready",
                "awaiting input",
            ]
        )
        self._publish_unreal_snapshot()

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
        if not self._shutting_down and not self._boot_in_progress:
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
        self._set_presence_state("speaking")
        self._set_presence_thought_lines(
            [
                "voice output active",
                "speech thread running",
                "maintaining live motion",
            ]
        )
        self._publish_unreal_snapshot()

    @Slot()
    def _on_speech_finished(self) -> None:
        self._speech_active = False
        self.state_manager.set_speaking(False)

        if self._queued_presence_action:
            queued_action = self._queued_presence_action
            duration_ms = self._queued_presence_action_duration_ms
            self._queued_presence_action = ""
            self._queued_presence_action_duration_ms = 0
            self._trigger_presence_action(queued_action, duration_ms)
            return

        if self._active_presence_action:
            self.state_manager.set_visual_state(self._active_presence_action, self._active_presence_action)
            if not self._busy and not self._shutting_down:
                self.control.set_status(f"VEXIS active | {self._active_presence_action}")
                self._set_presence_state(self._active_presence_action)
            return

        self.state_manager.set_visual_state("idle", "ready")

        if not self._busy and not self._shutting_down:
            self.control.set_status("VEXIS active | ready")
            self._set_presence_state("idle")
        self._publish_unreal_snapshot()

    @Slot(str)
    def _on_speech_error(self, message: str) -> None:
        _append_runtime_log("ERROR", f"Speech error: {message}")
        self.control.append_response("system", f"Speech error: {message}")

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        _append_runtime_log("ERROR", f"Worker error: {message}")
        self._busy = False
        self._active_presence_action = ""
        self._queued_presence_action = ""
        self._queued_presence_action_duration_ms = 0
        self._presence_action_timer.stop()
        self.state_manager.set_visual_state("idle", "ready")
        self.state_manager.set_thinking(False, current_focus="worker error")
        self.control.set_status("VEXIS active | error")
        self._set_presence_state("idle")
        self._set_presence_thought_lines(
            [
                "worker failure",
                "see system log",
            ]
        )
        self.control.append_response("system", message)
        self._publish_unreal_snapshot()

    def _speak_async(self, text: str) -> None:
        if self.speech_worker is None or self._shutting_down or not text.strip():
            return
        self.state_manager.set_speaking(True, last_spoken_text=text)
        self.request_speech.emit(text)

    def _render_spontaneous_action_text(self, action: Any, fallback_text: str) -> str:
        text = (fallback_text or "").strip()
        if not text:
            return ""
        if action.action_type != "idle_social_prompt":
            return text

        state = self.state_manager.get_state()
        rendered = self.idle_front_router.route_idle_expression(
            category=action.action_type,
            seed_text=text,
            personality_allowed=True,
            recent_lines=self._recent_idle_lines[-4:],
            last_spoken_text=state.speech.last_spoken_text,
            state_context=self._idle_state_context(action, state),
        )
        normalized = self._normalize_spoken_text(rendered)
        if not normalized:
            return ""

        recent_norms = {
            self._normalize_spoken_text(line)
            for line in self._recent_idle_lines[-4:]
            if line.strip()
        }
        last_norm = self._normalize_spoken_text(state.speech.last_spoken_text)
        if normalized == last_norm or normalized in recent_norms:
            return ""
        return rendered

    def _idle_state_context(self, action: Any, state: Any) -> str:
        return (
            f"open_questions={len(state.epistemic.open_questions)}; "
            f"unsupported_claims={len(state.epistemic.unsupported_claims)}; "
            f"last_classification={state.cognition.last_classification or 'unknown'}; "
            f"status={state.presence.status_text}; "
            f"seed={action.metadata.get('seed_text', '')}"
        )

    def _remember_idle_line(self, text: str) -> None:
        if not text.strip():
            return
        self._recent_idle_lines.append(text.strip())
        self._recent_idle_lines = self._recent_idle_lines[-6:]

    def _normalize_spoken_text(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().lower().split())
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch.isspace())
        return " ".join(cleaned.split())

    @Slot(str)
    def _on_immediate_front_response(self, text: str) -> None:
        if not text or self._shutting_down:
            return
        self.control.append_response("VEXIS", text)
        self._last_response_text = text
        self._publish_unreal_snapshot()

    def _set_presence_state(self, state: str) -> None:
        self._presence_state = state
        self.presence.set_state(state)

    def _set_presence_thought_lines(self, lines: list[str]) -> None:
        self._presence_thought_lines = lines[-12:] if lines else ["..."]
        self.presence.set_thought_lines(self._presence_thought_lines)

    def _publish_unreal_snapshot(self) -> None:
        state = self.state_manager.get_state()
        snapshot = {
            "visual_state": self._presence_state or state.presence.visual_state,
            "status_text": state.presence.status_text,
            "is_thinking": bool(state.cognition.is_thinking),
            "is_speaking": bool(self._speech_active or state.speech.is_speaking),
            "boot_in_progress": bool(self._boot_in_progress),
            "current_focus": state.cognition.current_focus or "",
            "last_classification": state.cognition.last_classification or "",
            "active_presence_action": self._active_presence_action,
            "last_response_text": self._last_response_text,
            "last_spoken_text": state.speech.last_spoken_text,
            "thought_lines": list(self._presence_thought_lines),
            "open_question_count": len(state.epistemic.open_questions),
            "unsupported_claim_count": len(state.epistemic.unsupported_claims),
            "completed_file_count": len(state.ingest.completed_files),
        }
        self.unreal_bridge.publish_snapshot(snapshot)

    def _external_state_change(self, state: str) -> None:
        self.state_manager.set_visual_state(state, state)
        self._set_presence_state(state)
        self._publish_unreal_snapshot()

    def _trigger_presence_action(self, action_name: str, duration_ms: int = 5000) -> None:
        if not action_name or self._shutting_down:
            return

        self._active_presence_action = action_name
        self._presence_action_timer.start(max(int(duration_ms), 1000))
        self.state_manager.set_visual_state(action_name, action_name)
        self.control.set_status(f"VEXIS active | {action_name}")
        self._set_presence_state(action_name)
        self._set_presence_thought_lines(
            [
                "presence action",
                action_name,
                "maintaining live motion",
            ]
        )
        self._publish_unreal_snapshot()

    def _clear_presence_action(self) -> None:
        self._active_presence_action = ""
        if self._shutting_down:
            return

        if self._speech_active:
            self.state_manager.set_visual_state("speaking", "speaking")
            self.control.set_status("VEXIS active | speaking")
            self._set_presence_state("speaking")
            self._publish_unreal_snapshot()
            return

        if self._busy:
            self.state_manager.set_visual_state("thinking", "thinking")
            self.control.set_status("VEXIS active | thinking")
            self._set_presence_state("thinking")
            self._publish_unreal_snapshot()
            return

        self.state_manager.set_visual_state("idle", "ready")
        self.control.set_status("VEXIS active | ready")
        self._set_presence_state("idle")
        self._publish_unreal_snapshot()

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
            details: dict[str, Any] = {}
            subject = ""
            relation = ""
            value = ""

            if isinstance(packed, str) and packed.strip().startswith("{"):
                try:
                    details = json.loads(packed)
                except Exception:
                    details = {}

            if details:
                subject = str(details.get("subject", "")).strip()
                relation = str(details.get("relation", "")).strip()
                value = str(details.get("value", "")).strip()
            else:
                try:
                    subject, relation, value = packed.split("|", 2)
                except ValueError:
                    continue

            fact_text = f"{subject} {relation} {value}"
            fact_metadata = {
                "subject": subject,
                "relation": relation,
                "value": value,
                "source_type": "internal_extracted_fact",
            }
            if details:
                fact_metadata.update(
                    {
                        "confidence": details.get("confidence", 0.72),
                        "fact_type": details.get("fact_type", "fact"),
                        "reasons": details.get("reasons", []),
                        **dict(details.get("metadata", {}) or {}),
                    }
                )

            fact_memory = self.state_manager.add_memory(
                kind="fact",
                content=fact_text,
                source="reasoning_engine",
                status="active",
                metadata=fact_metadata,
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
                self.memory_store.save_memory(fact_memory)
            except Exception:
                pass

        resolved_questions = updates.get("resolved_questions", [])
        for question in resolved_questions:
            self.state_manager.remove_open_question(question)
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
                self.memory_store.save_memory(resolved_memory)
            except Exception:
                pass

    def _persist_belief_updates(self, beliefs: list[Any]) -> None:
        current_state = self.state_manager.get_state()
        existing_ids = {
            str(memory.metadata.get("belief_id"))
            for memory in current_state.recent_memories
            if memory.kind == "belief_candidate" and memory.metadata.get("belief_id")
        }

        for belief in beliefs:
            belief_id = getattr(belief, "belief_id", None)
            if not belief_id or belief_id in existing_ids:
                continue

            belief_memory = self.state_manager.add_memory(
                kind="belief_candidate",
                content=str(getattr(belief, "statement", "")),
                source="cognition_loop",
                status=str(getattr(belief, "status", "candidate")),
                metadata={
                    "belief_id": belief_id,
                    "confidence_score": float(getattr(belief, "confidence_score", 0.0)),
                    "confidence_label": str(getattr(belief, "confidence_label", "very_low")),
                    "status": str(getattr(belief, "status", "candidate")),
                    "support_count": int(getattr(belief, "support_count", 0)),
                    "contradiction_count": int(getattr(belief, "contradiction_count", 0)),
                    "supporting_sources": list(getattr(belief, "supporting_sources", [])),
                    "contradicting_sources": list(getattr(belief, "contradicting_sources", [])),
                    "reasons": list(getattr(belief, "reasons", [])),
                    **dict(getattr(belief, "metadata", {}) or {}),
                },
                interaction_context={
                    "speaker_role": "system",
                    "speaker_id": "vex_cognition_loop",
                    "source": "cognition_loop",
                    "interaction_type": "belief_candidate",
                    "importance": "high",
                    "store_as_evidence": True,
                    "expects_reply": False,
                    "expects_reasoning": True,
                    "personality_allowed": False,
                },
            )
            try:
                self.memory_store.save_memory(belief_memory)
            except Exception:
                pass
            existing_ids.add(belief_id)

    def handle_user_message(self, text: str) -> None:
        if self._boot_in_progress:
            self.control.append_response("system", "VEXIS is still starting the local model.")
            return

        if self._busy or self._shutting_down:
            self.control.append_response("system", "VEXIS is still processing the previous request.")
            return

        self.output_gate.note_user_activity()
        self._busy = True
        self.state_manager.set_visual_state("thinking", "thinking")
        self.state_manager.set_thinking(True, current_focus="input analysis")

        self.control.set_status("VEXIS active | thinking")
        self._set_presence_state("thinking")
        self._set_presence_thought_lines(
            [
                "parsing input",
                "routing front layer",
                "preparing core review",
                text[:42],
            ]
        )
        self._publish_unreal_snapshot()

        self.request_process_message.emit(text)

    def handle_files_added(self, files: list[str]) -> None:
        if self._boot_in_progress:
            self.control.append_response("system", "VEXIS is still starting the local model.")
            return

        if self._busy or self._shutting_down:
            self.control.append_response("system", "VEXIS is still processing the previous request.")
            return

        self.output_gate.note_user_activity()
        self._busy = True
        self.state_manager.set_visual_state("thinking", "reviewing files")
        self.state_manager.set_thinking(True, current_focus="file ingest")

        short = [Path(f).name for f in files]
        self.control.set_status("VEXIS active | ingesting files")
        self._set_presence_state("thinking")
        self._set_presence_thought_lines(
            [
                "new files received",
                *short[:6],
                "extracting contents",
            ]
        )
        self._publish_unreal_snapshot()

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
            self._last_response_text = bundle.response_text

        self._set_presence_thought_lines(
            [
                f"type: {classification_type}",
                f"confidence: {classification_confidence:.2f}",
                "reasoning complete",
                "queueing speech",
            ]
        )

        internal_metadata = bundle.internal_answer.get("metadata", {}) if bundle.internal_answer else {}
        presence_action = str(internal_metadata.get("presence_action", "")).strip()
        presence_action_duration_ms = int(internal_metadata.get("presence_action_duration_ms", 5000) or 5000)

        decision: OutputDecision = self.output_gate.evaluate_direct_response(
            text=bundle.response_text,
            state=self.state_manager.get_state(),
        )
        will_speak = bool(decision.allow_speech_output and bundle.response_text and self.speech_worker is not None)
        if will_speak:
            if presence_action:
                self._queued_presence_action = presence_action
                self._queued_presence_action_duration_ms = presence_action_duration_ms
            self._speak_async(bundle.response_text)
            self.output_gate.note_spoken_output()
        elif presence_action:
            self._trigger_presence_action(presence_action, presence_action_duration_ms)
        else:
            self.control.set_status("VEXIS active | ready")
            self.state_manager.set_visual_state("idle", "ready")
            self._set_presence_state("idle")
        self._publish_unreal_snapshot()

    @Slot(object)
    def _on_files_finished(self, payload: Any) -> None:
        if self._shutting_down:
            return

        self._busy = False

        for line in payload["system_lines"]:
            self.control.append_response("system", line)

        self.control.append_response("VEXIS", payload["response_text"])
        self._last_response_text = payload["response_text"]

        self.state_manager.set_thinking(False, current_focus="file intake complete")
        self.state_manager.set_visual_state("idle", "ready")

        self._set_presence_thought_lines(
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
            self._set_presence_state("idle")
        self._publish_unreal_snapshot()

    @Slot(object)
    def _on_cognition_cycle_finished(self, payload: Any) -> None:
        if self._shutting_down:
            return

        cycle_result: CognitionCycleResult = payload["cycle_result"]
        goals: list[GoalRecord] = payload["goals"]

        if cycle_result.belief_updates:
            self._persist_belief_updates(cycle_result.belief_updates)

        for note in cycle_result.notes[:3]:
            self.control.append_response("system", note)

        for belief in cycle_result.belief_updates[:2]:
            self.control.append_response(
                "system",
                f"belief candidate: {belief.statement} [{belief.confidence_label}]",
            )

        for goal in goals[:2]:
            self.control.append_response("system", f"goal created: {goal.title}")

        for action in cycle_result.actions:
            decision = self.output_gate.evaluate_action(
                action=action,
                state=self.state_manager.get_state(),
            )
            action_text = decision.text
            if decision.allow_gui_output or decision.allow_speech_output:
                action_text = self._render_spontaneous_action_text(action, decision.text)
                if not action_text:
                    continue

            if decision.allow_gui_output and action_text:
                self.control.append_response("VEXIS", action_text)
                self._last_response_text = action_text
                if action.action_type == "idle_social_prompt":
                    self._remember_idle_line(action_text)

            if decision.allow_speech_output and action_text:
                self._speak_async(action_text)
                self.output_gate.note_spoken_output()
        self._publish_unreal_snapshot()

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
            self.unreal_bridge.stop()
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
    _install_exception_logging()
    _append_runtime_log("INFO", "Controller startup")
    app = QApplication(sys.argv)

    control = ControlWindow()
    external_presence = os.environ.get("VEXIS_EXTERNAL_PRESENCE", "").strip().lower() in {"1", "true", "yes", "on"}
    internal_presence = os.environ.get("VEXIS_INTERNAL_PRESENCE", "").strip().lower() in {"1", "true", "yes", "on"}
    presence = PresenceWindow() if internal_presence and not external_presence else NullPresenceWindow()

    control.move(80, 120)
    presence.setPosition(900, 100)

    controller = VexisGuiController(app, control, presence)
    app.aboutToQuit.connect(controller.shutdown_all)

    control.show()
    presence.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()





