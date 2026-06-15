"""Milestone 1: prove live transcription works (mic + system, tagged).

    python run_transcribe.py --list-devices   # see input + loopback devices
    python run_transcribe.py                   # start live transcription

Speak into your mic (tagged [me]) and play audio / join a call (tagged [other]).
Partial results overwrite the current line; finals print on their own line.
Ctrl+C to quit. No network, no LLM, no overlay yet.
"""
from __future__ import annotations

import sys
import time

from events import EventBus, StatusUpdate, TranscriptUpdate
from copilot.audio_capture import AudioCapture, list_devices
from copilot.transcriber import Transcriber

# ANSI helpers for an in-place partial line
_CLEAR_LINE = "\r\033[K"
_last_was_partial = {"me": False, "other": False}


def _on_status(e: StatusUpdate) -> None:
    print(f"{_CLEAR_LINE}· {e.text}")


def _on_transcript(e: TranscriptUpdate) -> None:
    tag = "me " if e.speaker == "me" else "oth"
    if e.is_final:
        print(f"{_CLEAR_LINE}[{tag}] {e.text}")
        _last_was_partial[e.speaker] = False
    else:
        # show partial in-place (best-effort; finals from the other speaker may interleave)
        sys.stdout.write(f"{_CLEAR_LINE}[{tag}] {e.text} …")
        sys.stdout.flush()
        _last_was_partial[e.speaker] = True


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252 by default
    except Exception:
        pass

    if "--list-devices" in sys.argv:
        print(list_devices())
        return

    bus = EventBus()
    bus.subscribe(StatusUpdate, _on_status)
    bus.subscribe(TranscriptUpdate, _on_transcript)
    bus.start()

    transcriber = Transcriber(bus)
    transcriber.start()

    capture = AudioCapture(bus)
    capture.start()

    print("Listening — speak (mic=[me]) and play audio (system=[oth]). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        capture.stop()
        transcriber.stop()
        bus.stop()


if __name__ == "__main__":
    main()
