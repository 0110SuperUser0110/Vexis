from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import QDateTime, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ControlWindow(QMainWindow):
    user_message = Signal(str)
    files_added = Signal(list)
    state_changed = Signal(str)
    shutdown_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VEXIS // Command Deck")
        self.setMinimumSize(1320, 860)

        self._message_count = 0
        self._speaker_counts = {"you": 0, "vexis": 0, "system": 0}
        self._activity_lines: list[tuple[str, str, str]] = []
        self._session_started = QDateTime.currentDateTime()
        self._state_buttons: dict[str, QPushButton] = {}

        self._build_ui()
        self._apply_styles()
        self._update_status_mode("ready")
        self._refresh_telemetry()
        QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 22, 22, 22)
        outer.setSpacing(18)

        self.shell = QFrame()
        self.shell.setObjectName("shell")
        shell_layout = QVBoxLayout(self.shell)
        shell_layout.setContentsMargins(24, 24, 24, 24)
        shell_layout.setSpacing(18)
        outer.addWidget(self.shell)

        shell_layout.addWidget(self._build_header())

        self.body_splitter = QSplitter(Qt.Horizontal)
        self.body_splitter.setObjectName("bodySplitter")
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(10)
        self.body_splitter.addWidget(self._build_primary_stack())
        self.body_splitter.addWidget(self._build_sidebar())
        self.body_splitter.setStretchFactor(0, 7)
        self.body_splitter.setStretchFactor(1, 4)
        shell_layout.addWidget(self.body_splitter, 1)

    def _build_primary_stack(self) -> QWidget:
        self.primary_splitter = QSplitter(Qt.Vertical)
        self.primary_splitter.setObjectName("primarySplitter")
        self.primary_splitter.setChildrenCollapsible(False)
        self.primary_splitter.setHandleWidth(10)
        self.primary_splitter.addWidget(self._build_conversation_panel())
        self.primary_splitter.addWidget(self._build_composer_panel())
        self.primary_splitter.setStretchFactor(0, 5)
        self.primary_splitter.setStretchFactor(1, 2)
        return self.primary_splitter

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("headerPanel")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(18)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(4)

        eyebrow = QLabel("DETERMINISTIC STATE INTELLIGENCE")
        eyebrow.setObjectName("eyebrowLabel")
        title_stack.addWidget(eyebrow)

        title = QLabel("VEXIS COMMAND DECK")
        title.setObjectName("titleLabel")
        title_stack.addWidget(title)

        subtitle = QLabel("Evidence-bound cognition, persistent expressive shell, dark-violet command presence.")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        title_stack.addWidget(subtitle)

        layout.addLayout(title_stack, 1)

        status_stack = QVBoxLayout()
        status_stack.setSpacing(8)
        status_stack.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.status_badge = QLabel("READY")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignCenter)
        status_stack.addWidget(self.status_badge, 0, Qt.AlignRight)

        self.status_label = QLabel("VEXIS active | ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight)
        self.status_label.setWordWrap(True)
        status_stack.addWidget(self.status_label)

        layout.addLayout(status_stack)
        return frame

    def _build_conversation_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("conversationPanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        heading_row = QHBoxLayout()
        heading_row.setSpacing(12)

        heading = QLabel("Conversation Stream")
        heading.setObjectName("sectionTitle")
        heading_row.addWidget(heading)

        heading_row.addStretch(1)

        hint = QLabel("CTRL+ENTER // transmit")
        hint.setObjectName("sectionHint")
        heading_row.addWidget(hint)

        layout.addLayout(heading_row)

        self.output_box = QTextBrowser()
        self.output_box.setObjectName("outputBox")
        self.output_box.setOpenExternalLinks(True)
        self.output_box.setPlaceholderText("VEXIS telemetry, responses, and internal notes will appear here.")
        self.output_box.document().setDocumentMargin(0)
        self.output_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_box.setMinimumHeight(360)
        layout.addWidget(self.output_box, 1)

        footer = QHBoxLayout()
        footer.setSpacing(10)

        self.channel_label = QLabel("CHANNEL // LIVE")
        self.channel_label.setObjectName("footerBadge")
        footer.addWidget(self.channel_label)

        footer.addStretch(1)

        self.log_counter = QLabel("LOG 000")
        self.log_counter.setObjectName("footerStat")
        footer.addWidget(self.log_counter)

        layout.addLayout(footer)
        return frame

    def _build_sidebar(self) -> QWidget:
        self.sidebar_splitter = QSplitter(Qt.Vertical)
        self.sidebar_splitter.setObjectName("sidebarSplitter")
        self.sidebar_splitter.setChildrenCollapsible(False)
        self.sidebar_splitter.setHandleWidth(10)
        self.sidebar_splitter.setMinimumWidth(420)
        self.sidebar_splitter.addWidget(self._build_overview_panel())
        self.sidebar_splitter.addWidget(self._build_activity_panel())
        self.sidebar_splitter.addWidget(self._build_protocol_panel())
        self.sidebar_splitter.setStretchFactor(0, 3)
        self.sidebar_splitter.setStretchFactor(1, 4)
        self.sidebar_splitter.setStretchFactor(2, 2)
        return self.sidebar_splitter

    def _build_overview_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sideCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Runtime Overview")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        subtitle = QLabel("Live deck telemetry for the active runtime. Resize the pane if you want to stare at numbers longer.")
        subtitle.setObjectName("sideText")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(12)
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)

        self.mode_value = self._metric_value("READY")
        self.voice_value = self._metric_value("STANDBY")
        self.user_count_value = self._metric_value("0")
        self.vex_count_value = self._metric_value("0")
        self.system_count_value = self._metric_value("0")
        self.uptime_value = self._metric_value("00:00:00")

        cards = [
            ("Mode", self.mode_value),
            ("Audio", self.voice_value),
            ("Operator", self.user_count_value),
            ("Vexis", self.vex_count_value),
            ("System", self.system_count_value),
            ("Uptime", self.uptime_value),
        ]
        for index, (label_text, value_label) in enumerate(cards):
            metrics_grid.addWidget(self._metric_card(label_text, value_label), index // 2, index % 2)

        layout.addLayout(metrics_grid)

        divider = QFrame()
        divider.setObjectName("dividerLine")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        note = QLabel("Presence rendering is now driven externally for testing, so this pane stays observational instead of forcing states.")
        note.setObjectName("sideText")
        note.setWordWrap(True)
        layout.addWidget(note)
        return frame

    def _build_activity_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sideCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Signal Trace")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        subtitle = QLabel("Recent system notes, intake events, and dialogue transitions. It now gets enough room to be legible, which is a radical improvement.")
        subtitle.setObjectName("sideText")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.activity_box = QTextBrowser()
        self.activity_box.setObjectName("activityBox")
        self.activity_box.setOpenExternalLinks(False)
        self.activity_box.setPlaceholderText("Recent system and dialogue signals will appear here.")
        self.activity_box.document().setDocumentMargin(0)
        self.activity_box.setMinimumHeight(240)
        self.activity_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.activity_box, 1)
        return frame

    def _build_protocol_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sideCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Operator Notes")
        title.setObjectName("sideTitle")
        layout.addWidget(title)

        self.notes_box = QTextBrowser()
        self.notes_box.setObjectName("notesBox")
        self.notes_box.setOpenExternalLinks(False)
        self.notes_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.notes_box.setMinimumHeight(170)
        self.notes_box.setHtml(
            "<div style='color:#d7c6f8;font-size:13px;line-height:1.6;'>"
            "Upload source material for deterministic ingest.<br><br>"
            "Social chatter stays in the language shell.<br>"
            "Substantive questions route to the evidence core and return once grounded.<br><br>"
            "Claims should arrive with proof if you expect them to be promoted above rumor."
            "</div>"
        )
        layout.addWidget(self.notes_box, 1)
        return frame

    def _build_composer_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("composerPanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title_row = QHBoxLayout()
        title_row.setSpacing(12)

        title = QLabel("Transmit")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)

        title_row.addStretch(1)

        composer_hint = QLabel("QUESTIONS // CLAIMS // NOTES // FILE INTAKE")
        composer_hint.setObjectName("sectionHint")
        title_row.addWidget(composer_hint)

        layout.addLayout(title_row)

        self.input_box = QTextEdit()
        self.input_box.setObjectName("inputBox")
        self.input_box.setMinimumHeight(170)
        self.input_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.input_box.setPlaceholderText(
            "Send a message, question, note, contradiction, or source request. "
            "The expressive shell will acknowledge; the core will ground the answer."
        )
        layout.addWidget(self.input_box, 1)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self.add_file_button = QPushButton("Upload Source")
        self.add_file_button.setObjectName("secondaryButton")
        self.add_file_button.clicked.connect(self._choose_files)
        button_row.addWidget(self.add_file_button)

        self.ready_hint = QLabel("Ready for dialogue and ingest")
        self.ready_hint.setObjectName("composerHint")
        self.ready_hint.setWordWrap(True)
        self.ready_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_row.addWidget(self.ready_hint, 1)

        self.send_button = QPushButton("Transmit")
        self.send_button.setObjectName("primaryButton")
        self.send_button.clicked.connect(self._emit_message)
        button_row.addWidget(self.send_button)

        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.setObjectName("dangerButton")
        self.shutdown_button.clicked.connect(self.shutdown_requested.emit)
        button_row.addWidget(self.shutdown_button)

        layout.addLayout(button_row)
        return frame

    def _metric_value(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("metricValue")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return label

    def _metric_card(self, label_text: str, value_label: QLabel) -> QWidget:
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        label = QLabel(label_text.upper())
        label.setObjectName("metricLabel")
        layout.addWidget(label)
        layout.addWidget(value_label)
        layout.addStretch(1)
        return card

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #09030f,
                    stop: 0.42 #14061f,
                    stop: 1 #220a36
                );
            }

            QFrame#shell {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #100419,
                    stop: 0.45 #1a0827,
                    stop: 1 #280d3d
                );
                border: 1px solid rgba(201, 167, 255, 0.22);
                border-radius: 30px;
            }

            QFrame#headerPanel,
            QFrame#conversationPanel,
            QFrame#composerPanel,
            QFrame#sideCard,
            QFrame#metricCard {
                background: rgba(22, 9, 35, 0.94);
                border: 1px solid rgba(192, 132, 252, 0.18);
                border-radius: 22px;
            }

            QFrame#metricCard {
                background: rgba(33, 12, 52, 0.92);
                border-radius: 18px;
            }

            QFrame#dividerLine {
                background: rgba(205, 180, 255, 0.18);
                border: none;
            }

            QSplitter#bodySplitter::handle,
            QSplitter#primarySplitter::handle,
            QSplitter#sidebarSplitter::handle {
                background: transparent;
            }

            QSplitter#bodySplitter::handle:hover,
            QSplitter#primarySplitter::handle:hover,
            QSplitter#sidebarSplitter::handle:hover {
                background: rgba(192, 132, 252, 0.16);
                border-radius: 4px;
            }

            QLabel#eyebrowLabel {
                color: #d8b4fe;
                font-size: 11px;
                font-family: "Cascadia Mono";
                letter-spacing: 2px;
            }

            QLabel#titleLabel {
                color: #faf5ff;
                font-size: 32px;
                font-weight: 700;
                font-family: "Bahnschrift";
                letter-spacing: 1px;
            }

            QLabel#subtitleLabel {
                color: #d9c6fb;
                font-size: 13px;
            }

            QLabel#statusBadge {
                min-width: 140px;
                padding: 10px 16px;
                border-radius: 16px;
                font-family: "Cascadia Mono";
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
                color: #f8f4ff;
                background: rgba(147, 51, 234, 0.20);
                border: 1px solid rgba(216, 180, 254, 0.44);
            }

            QLabel#statusBadge[statusMode="thinking"],
            QLabel#statusBadge[statusMode="ingesting"] {
                color: #faf5ff;
                background: rgba(124, 58, 237, 0.24);
                border: 1px solid rgba(196, 181, 253, 0.48);
            }

            QLabel#statusBadge[statusMode="speaking"] {
                color: #fff7fb;
                background: rgba(168, 85, 247, 0.28);
                border: 1px solid rgba(233, 213, 255, 0.52);
            }

            QLabel#statusBadge[statusMode="error"] {
                color: #fff1f2;
                background: rgba(127, 29, 29, 0.28);
                border: 1px solid rgba(252, 165, 165, 0.46);
            }

            QLabel#statusLabel {
                color: #eadcff;
                font-size: 15px;
                font-family: "Cascadia Mono";
            }

            QLabel#sectionTitle,
            QLabel#sideTitle {
                color: #faf5ff;
                font-size: 19px;
                font-weight: 650;
                font-family: "Bahnschrift";
            }

            QLabel#sectionHint,
            QLabel#footerBadge,
            QLabel#footerStat,
            QLabel#composerHint,
            QLabel#metricLabel {
                color: #d8b4fe;
                font-size: 11px;
                font-family: "Cascadia Mono";
                letter-spacing: 1px;
            }

            QLabel#footerBadge {
                padding: 6px 10px;
                border-radius: 12px;
                border: 1px solid rgba(192, 132, 252, 0.22);
                background: rgba(31, 11, 48, 0.96);
            }

            QLabel#metricValue {
                color: #faf5ff;
                font-size: 20px;
                font-family: "Bahnschrift";
                font-weight: 700;
            }

            QLabel#sideText,
            QLabel#composerHint {
                color: #d9c6fb;
                font-size: 13px;
            }

            QTextBrowser#outputBox,
            QTextBrowser#activityBox,
            QTextBrowser#notesBox,
            QTextEdit#inputBox {
                background: rgba(11, 4, 19, 0.96);
                color: #f5ebff;
                border: 1px solid rgba(192, 132, 252, 0.16);
                border-radius: 18px;
                padding: 14px;
                selection-background-color: rgba(168, 85, 247, 0.34);
                selection-color: #fff8ff;
            }

            QTextBrowser#outputBox,
            QTextBrowser#activityBox,
            QTextBrowser#notesBox {
                font-size: 14px;
            }

            QTextEdit#inputBox {
                font-size: 15px;
                border: 1px solid rgba(216, 180, 254, 0.20);
            }

            QPushButton {
                padding: 11px 18px;
                min-width: 112px;
                border-radius: 16px;
                font-size: 13px;
                font-weight: 650;
                font-family: "Bahnschrift";
                border: 1px solid rgba(192, 132, 252, 0.26);
                background: rgba(31, 11, 48, 0.98);
                color: #f5eaff;
            }

            QPushButton:hover {
                background: rgba(45, 15, 68, 1.0);
                border: 1px solid rgba(221, 214, 254, 0.46);
            }

            QPushButton:pressed {
                background: rgba(19, 7, 29, 1.0);
            }

            QPushButton#primaryButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #7c3aed, stop: 1 #c026d3);
                color: #fff8ff;
                border: 1px solid rgba(233, 213, 255, 0.54);
            }

            QPushButton#primaryButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #8b5cf6, stop: 1 #d946ef);
            }

            QPushButton#secondaryButton {
                background: rgba(24, 8, 37, 0.98);
            }

            QPushButton#dangerButton {
                background: rgba(67, 16, 34, 0.98);
                border: 1px solid rgba(253, 164, 175, 0.32);
                color: #ffe7f0;
            }

            QPushButton#dangerButton:hover {
                background: rgba(93, 22, 46, 0.98);
            }

            QPushButton#stateButton {
                min-width: 0;
                padding: 10px 12px;
                border-radius: 14px;
                font-size: 12px;
                font-family: "Cascadia Mono";
            }

            QPushButton#stateButton:checked {
                background: rgba(147, 51, 234, 0.22);
                border: 1px solid rgba(221, 214, 254, 0.50);
                color: #fff7ff;
            }
            """
        )

        self.setFont(QFont("Segoe UI", 10))
        self.output_box.setFont(QFont("Segoe UI", 10))
        self.activity_box.setFont(QFont("Cascadia Mono", 9))
        self.notes_box.setFont(QFont("Segoe UI", 10))
        self.input_box.setFont(QFont("Segoe UI", 11))

    def _apply_initial_splitter_sizes(self) -> None:
        self.body_splitter.setSizes([960, 520])
        self.primary_splitter.setSizes([640, 240])
        self.sidebar_splitter.setSizes([320, 340, 200])

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        mode = self._status_mode_from_text(text)
        self._update_status_mode(mode)
        self.ready_hint.setText(f"Signal // {text}")
        self._refresh_telemetry()

    def append_response(self, speaker: str, text: str) -> None:
        safe_text = text.rstrip()
        if not safe_text:
            return

        speaker_key = speaker.strip().lower()
        if speaker_key not in self._speaker_counts:
            speaker_key = "system"
        self._speaker_counts[speaker_key] += 1
        self._message_count += 1

        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        alignment = "right" if speaker_key == "you" else "left"
        accent = {
            "you": ("#07131b", "#2dd4bf", "#ccfbf1", "#5eead4"),
            "vexis": ("#1a0b29", "#a855f7", "#f5eaff", "#d8b4fe"),
            "system": ("#261507", "#f59e0b", "#fff1db", "#fcd34d"),
        }[speaker_key]
        background, border, text_color, speaker_color = accent

        payload = html.escape(safe_text).replace("\n", "<br>")
        speaker_label = "YOU" if speaker_key == "you" else "VEXIS" if speaker_key == "vexis" else "SYSTEM"

        block = f"""
        <div style="width:100%; text-align:{alignment}; margin:0 0 12px 0;">
          <div style="display:inline-block; max-width:90%; background:{background}; border:1px solid {border}; border-radius:18px; padding:12px 14px; text-align:left; box-shadow:0 0 0 1px rgba(255,255,255,0.02);">
            <div style="font-family:'Cascadia Mono'; font-size:10px; letter-spacing:1px; color:{speaker_color}; margin-bottom:7px;">{speaker_label} // {timestamp}</div>
            <div style="font-size:14px; line-height:1.58; color:{text_color};">{payload}</div>
          </div>
        </div>
        """

        cursor = self.output_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(block)
        cursor.insertBlock()
        self.output_box.setTextCursor(cursor)
        self.output_box.ensureCursorVisible()

        trace_line = safe_text if len(safe_text) <= 132 else safe_text[:129].rstrip() + "..."
        self._activity_lines.append((timestamp, speaker_label, trace_line))
        self._activity_lines = self._activity_lines[-10:]
        self._refresh_activity_panel()
        self._refresh_telemetry()

    def clear_input(self) -> None:
        self.input_box.clear()

    def _emit_message(self) -> None:
        text = self.input_box.toPlainText().strip()
        if not text:
            return

        self.append_response("you", text)
        self.user_message.emit(text)
        self.clear_input()

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files for VEXIS",
            str(Path.home()),
            "All Files (*.*)",
        )
        if files:
            self.files_added.emit(files)
            count = len(files)
            self._activity_lines.append(
                (QDateTime.currentDateTime().toString("HH:mm:ss"), "SYSTEM", f"Queued {count} file{'s' if count != 1 else ''} for ingest."),
            )
            self._activity_lines = self._activity_lines[-10:]
            self._refresh_activity_panel()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() == Qt.ControlModifier:
            self._emit_message()
            return
        super().keyPressEvent(event)

    def _update_status_mode(self, mode: str) -> None:
        badge_text = {
            "ready": "READY",
            "thinking": "THINKING",
            "speaking": "SPEAKING",
            "ingesting": "INGEST",
            "error": "ERROR",
            "listening": "LISTEN",
        }.get(mode, mode.upper())
        self.status_badge.setText(badge_text)
        self.status_badge.setProperty("statusMode", mode)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.status_badge.update()

        mapped_button = {
            "ready": "idle",
            "thinking": "thinking",
            "speaking": "speaking",
            "ingesting": "thinking",
            "error": "idle",
            "listening": "listening",
        }.get(mode)
        if mapped_button and mapped_button in self._state_buttons:
            self._state_buttons[mapped_button].setChecked(True)

    def _status_mode_from_text(self, text: str) -> str:
        lowered = (text or "").lower()
        if "error" in lowered or "failed" in lowered:
            return "error"
        if "ingest" in lowered or "file" in lowered:
            return "ingesting"
        if "speaking" in lowered or "voice" in lowered:
            return "speaking"
        if "thinking" in lowered or "processing" in lowered:
            return "thinking"
        if "listen" in lowered:
            return "listening"
        return "ready"

    def _refresh_activity_panel(self) -> None:
        if not self._activity_lines:
            self.activity_box.setHtml(
                "<div style='color:#d8b4fe;font-family:Cascadia Mono;font-size:11px;'>No signal trace recorded yet.</div>"
            )
            return

        items = []
        for timestamp, speaker, line in reversed(self._activity_lines[-8:]):
            color = "#d8b4fe" if speaker == "VEXIS" else "#fcd34d" if speaker == "SYSTEM" else "#5eead4"
            items.append(
                f"<div style='margin:0 0 12px 0;'>"
                f"<div style='font-family:Cascadia Mono;font-size:10px;color:{color};'>{speaker} // {timestamp}</div>"
                f"<div style='font-size:12px;line-height:1.55;color:#f3e8ff;'>{html.escape(line)}</div>"
                f"</div>"
            )
        self.activity_box.setHtml("".join(items))

    def _refresh_telemetry(self) -> None:
        self.mode_value.setText(self.status_badge.text())
        self.voice_value.setText("LIVE" if self.status_badge.text() == "SPEAKING" else "STANDBY")
        self.user_count_value.setText(str(self._speaker_counts["you"]))
        self.vex_count_value.setText(str(self._speaker_counts["vexis"]))
        self.system_count_value.setText(str(self._speaker_counts["system"]))
        elapsed = self._session_started.secsTo(QDateTime.currentDateTime())
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.uptime_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.log_counter.setText(f"LOG {self._message_count:03d}")
