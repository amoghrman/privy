"""Milestone 2: full pipeline minus overlay.

audio -> transcriber -> context_builder -> inference, all over the event bus.
Transcripts print live (tagged me/oth). Press ENTER to ask the copilot about the
latest question; the answer streams in. Auto-surface also fires when the other
speaker asks something question-shaped. Ctrl+C to quit.

Requires Ollama running with the configured model. No overlay/hotkey yet.
"""
from __future__ import annotations

import sys
import threading
import time

from events import (
    AnswerToken, AskRequested, EventBus, StatusUpdate, TranscriptUpdate,
)
from copilot.audio_capture import AudioCapture
from copilot.transcriber import Transcriber
from copilot.memory import NullMemoryProvider
from copilot.context_builder import ContextBuilder
from copilot.inference import InferenceEngine

_CLEAR = "\r\033[K"
_ask_time = {"t": 0.0}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    bus = EventBus()

    def on_status(e: StatusUpdate) -> None:
        print(f"{_CLEAR}· {e.text}")

    def on_transcript(e: TranscriptUpdate) -> None:
        tag = "me " if e.speaker == "me" else "oth"
        if e.is_final:
            print(f"{_CLEAR}[{tag}] {e.text}")
        else:
            sys.stdout.write(f"{_CLEAR}[{tag}] {e.text} …")
            sys.stdout.flush()

    def on_token(e: AnswerToken) -> None:
        if e.first:
            dt = time.time() - _ask_time["t"] if _ask_time["t"] else 0.0
            print(f"{_CLEAR}\n💡 (first token {dt:.2f}s):")
        if e.text:
            sys.stdout.write(e.text)
            sys.stdout.flush()
        if e.done:
            print("\n")

    def on_ask(e: AskRequested) -> None:
        _ask_time["t"] = time.time()
        print(f"{_CLEAR}\n[asking — {e.reason}]")

    bus.subscribe(StatusUpdate, on_status)
    bus.subscribe(TranscriptUpdate, on_transcript)
    bus.subscribe(AnswerToken, on_token)
    bus.subscribe(AskRequested, on_ask)
    bus.start()

    memory = NullMemoryProvider()
    builder = ContextBuilder(bus, memory)
    builder.start()
    engine = InferenceEngine(bus, builder)
    engine.start()
    transcriber = Transcriber(bus)
    transcriber.start()
    capture = AudioCapture(bus)
    capture.start()

    print("Live. Speak normally. Press ENTER to ask about the latest question. Ctrl+C to quit.")

    def reader() -> None:
        for _ in sys.stdin:
            bus.publish(AskRequested(reason="hotkey"))

    threading.Thread(target=reader, daemon=True).start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        capture.stop()
        transcriber.stop()
        engine.stop()
        bus.stop()


if __name__ == "__main__":
    main()
