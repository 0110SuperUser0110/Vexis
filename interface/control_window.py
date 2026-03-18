from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
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
        self.setWindowTitle("VEXIS Interface")
        self.setMinimumSize(760, 560)

        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.status_label = QLabel("VEXIS active | ready")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setObjectName("outputBox")
        self.output_box.setPlaceholderText("VEXIS responses and history will appear here...")
        layout.addWidget(self.output_box, 1)

        self.input_box = QTextEdit()
        self.input_box.setObjectName("inputBox")
        self.input_box.setFixedHeight(120)
        self.input_box.setPlaceholderText("Speak to VEXIS here. Type a message, question, note, or claim.")
        layout.addWidget(self.input_box)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.add_file_button = QPushButton("+ Add File")
        self.add_file_button.clicked.connect(self._choose_files)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._emit_message)

        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.shutdown_requested.emit)

        button_row.addWidget(self.add_file_button)
        button_row.addStretch(1)
        button_row.addWidget(self.send_button)
        button_row.addWidget(self.shutdown_button)

        layout.addLayout(button_row)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #130f26;
            }

            QLabel#statusLabel {
                color: #f4f0ff;
                font-size: 18px;
                font-weight: 600;
                padding: 6px 2px;
            }

            QTextEdit#outputBox, QTextEdit#inputBox {
                background-color: #1a1238;
                color: #f4f0ff;
                border: 1px solid #8b6bff;
                border-radius: 14px;
                padding: 10px;
                font-size: 15px;
            }

            QPushButton {
                background-color: #24164d;
                color: #ffffff;
                border: 1px solid #8b6bff;
                border-radius: 14px;
                padding: 10px 18px;
                min-width: 110px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #31206a;
            }

            QPushButton:pressed {
                background-color: #1b113b;
            }
            """
        )

        font = QFont("Segoe UI", 10)
        self.setFont(font)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def append_response(self, speaker: str, text: str) -> None:
        safe_text = text.rstrip()
        self.output_box.append(f"{speaker}: {safe_text}")

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

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() == Qt.ControlModifier:
            self._emit_message()
            return
        super().keyPressEvent(event)