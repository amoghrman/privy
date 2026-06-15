"""app: wire every module through the event bus and run the overlay.

    python app.py

Flow (all decoupled via EventBus):
    audio_capture --AudioFrame--> transcriber --TranscriptUpdate-->
        context_builder --(memory.recall)--> inference --AnswerToken--> overlay
    hotkeys --AskRequested/ToggleOverlay--> (inference / overlay)

Qt owns the main thread; capture/STT/inference run on background threads and
talk to the GUI only via Qt signals inside the overlay. 100% local.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication
from pynput import keyboard

import config
from events import (
    AnswerToken, AskRequested, EventBus, StatusUpdate, ToggleOverlay, TranscriptUpdate,
)
from copilot.audio_capture import AudioCapture
from copilot.transcriber import Transcriber
from copilot.memory import NullMemoryProvider
from copilot.context_builder import ContextBuilder
from copilot.inference import InferenceEngine
from copilot.overlay import OverlayWindow, position_top_right


def main() -> None:
    app = QApplication(sys.argv)

    bus = EventBus()

    # optional console logging (COPILOT_DEBUG=1) — the product UI is the overlay,
    # but this gives visibility while developing/debugging.
    if os.environ.get("COPILOT_DEBUG") == "1":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        def _log_tx(e: TranscriptUpdate) -> None:
            if e.is_final:
                print(f"[{e.speaker}] {e.text}", flush=True)

        def _log_ask(e: AskRequested) -> None:
            print(f">>> ASK ({e.reason})", flush=True)

        def _log_tok(e: AnswerToken) -> None:
            if e.first:
                print("<<< ANSWER: ", end="", flush=True)
            if e.text:
                print(e.text, end="", flush=True)
            if e.done:
                print("\n--- answer done ---", flush=True)

        bus.subscribe(TranscriptUpdate, _log_tx)
        bus.subscribe(AskRequested, _log_ask)
        bus.subscribe(AnswerToken, _log_tok)
        bus.subscribe(StatusUpdate, lambda e: print(f"· {e.text}", flush=True))

    bus.start()

    # GUI (main thread)
    overlay = OverlayWindow(bus)
    position_top_right(overlay)
    overlay.show()

    # services (background threads)
    memory = NullMemoryProvider()          # swap for your SQLite provider later
    builder = ContextBuilder(bus, memory)
    builder.start()
    engine = InferenceEngine(bus, builder)
    engine.start()
    transcriber = Transcriber(bus)
    transcriber.start()
    capture = AudioCapture(bus)
    capture.start()

    # global hotkeys
    hotkeys = keyboard.GlobalHotKeys({
        config.HOTKEY_ASK: lambda: bus.publish(AskRequested(reason="hotkey")),
        config.HOTKEY_TOGGLE: lambda: bus.publish(ToggleOverlay()),
    })
    hotkeys.start()

    def shutdown() -> None:
        capture.stop()
        transcriber.stop()
        engine.stop()
        hotkeys.stop()
        bus.stop()

    app.aboutToQuit.connect(shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
