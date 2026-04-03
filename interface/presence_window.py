from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QColor
from PySide6.QtQuick import QQuickView


class NullPresenceWindow:
    def __init__(self) -> None:
        self._pending_state = "idle"
        self._pending_thought_lines = ["VEXIS external presence active"]

    def set_state(self, state: str) -> None:
        self._pending_state = state

    def set_thought_lines(self, lines: list[str]) -> None:
        self._pending_thought_lines = lines[-12:] if lines else ["..."]

    def setPosition(self, *_args) -> None:  # noqa: N802
        return

    def show(self) -> None:
        return

    def close(self) -> None:
        return


class PresenceWindow(QQuickView):
    """
    Floating VEXIS presence window using QML + View3D.

    This is the correct path for:
    - transparent floating object
    - custom animated visuals
    - future text cascades on faces
    """

    def __init__(self) -> None:
        super().__init__()

        self.setColor(QColor(0, 0, 0, 0))
        self.setFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setResizeMode(QQuickView.SizeRootObjectToView)
        self.resize(680, 760)

        self._drag_offset: Optional[QPoint] = None
        self._pending_state = "idle"
        self._pending_thought_lines = ["VEXIS online", "Awaiting input"]

        qml_path = Path(__file__).resolve().parent / "qml" / "Presence.qml"
        self.setSource(QUrl.fromLocalFile(str(qml_path)))

        if self.status() == QQuickView.Error:
            for err in self.errors():
                print(err.toString())

        root = self.rootObject()
        if root is not None:
            root.setProperty("stateName", self._pending_state)
            root.setProperty("thoughtLines", self._pending_thought_lines)

    def set_state(self, state: str) -> None:
        self._pending_state = state
        root = self.rootObject()
        if root is not None:
            root.setProperty("stateName", state)

    def set_thought_lines(self, lines: list[str]) -> None:
        self._pending_thought_lines = lines[-12:] if lines else ["..."]
        root = self.rootObject()
        if root is not None:
            root.setProperty("thoughtLines", self._pending_thought_lines)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.position()
            event.accept()
            return

        if event.button() == Qt.RightButton:
            self.close()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.setPosition(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)
