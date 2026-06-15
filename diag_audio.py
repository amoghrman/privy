"""Diagnostic: are AudioFrames flowing, and does VAD trip? Runs ~12s.

Prints per-speaker: frames received, peak RMS, and how many frames VAD called
speech. Speak into the mic AND play some audio while it runs.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np
import webrtcvad

import config
from events import AudioFrame, EventBus, StatusUpdate
from copilot.audio_capture import AudioCapture

stats = {
    "me": {"frames": 0, "peak": 0.0, "voiced": 0},
    "other": {"frames": 0, "peak": 0.0, "voiced": 0},
}
vads = {"me": webrtcvad.Vad(config.VAD_AGGRESSIVENESS),
        "other": webrtcvad.Vad(config.VAD_AGGRESSIVENESS)}
lock = threading.Lock()


def on_frame(e: AudioFrame) -> None:
    s = e.samples
    rms = float(np.sqrt(np.mean(s * s)) + 1e-12)
    pcm16 = np.clip(s * 32768.0, -32768, 32767).astype(np.int16).tobytes()
    voiced = vads[e.speaker].is_speech(pcm16, config.SAMPLE_RATE)
    with lock:
        st = stats[e.speaker]
        st["frames"] += 1
        st["peak"] = max(st["peak"], rms)
        st["voiced"] += int(voiced)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    bus = EventBus()
    bus.subscribe(AudioFrame, on_frame)
    bus.subscribe(StatusUpdate, lambda e: print("·", e.text))
    bus.start()
    cap = AudioCapture(bus)
    cap.start()
    print("DIAG: speak into mic + play audio for ~12s …")
    for i in range(12):
        time.sleep(1)
        with lock:
            m, o = stats["me"], stats["other"]
        print(f"  t={i+1:2d}s  me[frames={m['frames']:4d} peak={m['peak']:.4f} voiced={m['voiced']:4d}]"
              f"  other[frames={o['frames']:4d} peak={o['peak']:.4f} voiced={o['voiced']:4d}]")
    cap.stop()
    bus.stop()
    print("DIAG done.")


if __name__ == "__main__":
    main()
