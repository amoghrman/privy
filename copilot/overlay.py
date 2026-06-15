"""overlay: always-on-top, screen-capture-EXCLUDED PySide6 panel.

Renders streaming answer tokens and the latest detected question. Global
hotkeys trigger an ask or toggle visibility.

Capture exclusion (Windows): after the native window exists we call
SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE). Zoom/Meet/OBS and the
Windows capture stack then composite the window out of any shared/recorded
frame, while it stays visible on the local display.

Qt owns the main thread. The event bus runs on its own thread, so bus handlers
must NOT touch widgets directly — they emit Qt signals, which Qt queues onto the
GUI thread.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, QObject, Signal, QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QGraphicsDropShadowEffect,
)
from PySide6.QtGui import QColor

import config
from events import (
    AnswerToken, AskRequested, EventBus, StatusUpdate, ToggleOverlay, TranscriptUpdate,
)

# WDA_EXCLUDEFROMCAPTURE = 0x11 (Win10 2004+); WDA_MONITOR = 0x01 (older, blacks out)
WDA_EXCLUDEFROMCAPTURE = 0x00000011
WDA_MONITOR = 0x00000001


def exclude_from_capture(hwnd: int) -> bool:
    """Hide this window from screen capture/recording. Returns True on success."""
    user32 = ctypes.windll.user32
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    if user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE):
        return True
    # fall back to the older affinity (blacks the window out in captures)
    return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_MONITOR))


class _Bridge(QObject):
    """Thread-safe relay from the event bus into the Qt GUI thread."""
    token = Signal(str, bool, bool)   # text, first, done
    status = Signal(str)
    question = Signal(str)
    asked = Signal(str)
    toggle = Signal()


class OverlayWindow(QWidget):
    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self.bus = bus
        self._drag_offset: QPoint | None = None
        self._answer = ""

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(config.OVERLAY_OPACITY)
        self.resize(config.OVERLAY_WIDTH, config.OVERLAY_HEIGHT)
        self._build_ui()

        # bridge bus -> GUI
        self._bridge = _Bridge()
        self._bridge.token.connect(self._on_token)
        self._bridge.status.connect(lambda s: self.footer.setText(s))
        self._bridge.question.connect(self._on_question)
        self._bridge.asked.connect(self._on_asked)
        self._bridge.toggle.connect(self._toggle)

        bus.subscribe(AnswerToken, lambda e: self._bridge.token.emit(e.text, e.first, e.done))
        bus.subscribe(StatusUpdate, lambda e: self._bridge.status.emit(e.text))
        bus.subscribe(AskRequested, lambda e: self._bridge.asked.emit(e.reason))
        bus.subscribe(ToggleOverlay, lambda e: self._bridge.toggle.emit())
        bus.subscribe(
            TranscriptUpdate,
            lambda e: self._bridge.question.emit(e.text)
            if (e.is_final and e.speaker == "other") else None,
        )

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        root.setStyleSheet(
            "#root { background: rgba(20,22,28,235); border-radius: 14px;"
            " border: 1px solid rgba(255,255,255,28); }"
        )
        shadow = QGraphicsDropShadowEffect(blurRadius=28, xOffset=0, yOffset=6)
        shadow.setColor(QColor(0, 0, 0, 160))
        root.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)

        self.title = QLabel("privy  ·  meeting copilot")
        self.title.setStyleSheet("color: #8ab4ff; font-weight: 600;")
        self.title.setFont(QFont("Segoe UI", 10))
        lay.addWidget(self.title)

        self.question_lbl = QLabel("Listening…")
        self.question_lbl.setWordWrap(True)
        self.question_lbl.setStyleSheet("color: rgba(255,255,255,150);")
        self.question_lbl.setFont(QFont("Segoe UI", 9))
        lay.addWidget(self.question_lbl)

        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setFrameStyle(0)
        self.answer.setStyleSheet(
            "QTextEdit { background: transparent; color: #f0f2f6; border: none; }"
        )
        self.answer.setFont(QFont("Segoe UI", 11))
        lay.addWidget(self.answer, 1)

        self.footer = QLabel(
            f"{config.HOTKEY_ASK}  ask   ·   {config.HOTKEY_TOGGLE}  hide"
        )
        self.footer.setStyleSheet("color: rgba(255,255,255,90);")
        self.footer.setFont(QFont("Segoe UI", 8))
        lay.addWidget(self.footer)

    # ── slots (GUI thread) ────────────────────────────────────
    def _on_token(self, text: str, first: bool, done: bool) -> None:
        if first:
            self._answer = ""
        self._answer += text
        self.answer.setPlainText(self._answer)
        sb = self.answer.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_question(self, text: str) -> None:
        self.question_lbl.setText(f"❝ {text}")

    def _on_asked(self, reason: str) -> None:
        self.answer.setPlainText("")
        self._answer = ""
        self.question_lbl.setText(self.question_lbl.text())
        self.footer.setText(f"thinking… ({reason})")
        if not self.isVisible():
            self.show()

    def _toggle(self) -> None:
        self.hide() if self.isVisible() else self.show()

    # ── capture exclusion (after native handle exists) ────────
    def apply_capture_exclusion(self) -> None:
        ok = exclude_from_capture(int(self.winId()))
        self.bus.publish(
            StatusUpdate("capture-excluded ✓" if ok else "capture-exclusion FAILED")
        )

    def showEvent(self, e) -> None:  # noqa: ANN001
        super().showEvent(e)
        self.apply_capture_exclusion()

    # ── dragging (frameless) ──────────────────────────────────
    def mousePressEvent(self, e) -> None:  # noqa: ANN001
        if e.button() == Qt.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:  # noqa: ANN001
        if self._drag_offset is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, e) -> None:  # noqa: ANN001
        self._drag_offset = None


def position_top_right(win: OverlayWindow) -> None:
    screen = QApplication.primaryScreen().availableGeometry()
    win.move(screen.right() - win.width() - 24, screen.top() + 24)
