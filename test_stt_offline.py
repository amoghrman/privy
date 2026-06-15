"""Offline STT validation: feed a known WAV through the real transcriber worker.

Proves the Whisper + VAD + gain pipeline works independent of live audio.
"""
from __future__ import annotations

import sys
import threading
import time
import wave

import numpy as np

import config
from events import EventBus, TranscriptUpdate
from copilot.transcriber import _StreamWorker

WAV = sys.argv[1] if len(sys.argv) > 1 else "_tts_test.wav"


def load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == config.SAMPLE_RATE, w.getframerate()
        assert w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    audio = load_wav_16k_mono(WAV)
    print(f"loaded {WAV}: {len(audio)/config.SAMPLE_RATE:.1f}s, peak={np.max(np.abs(audio)):.3f}")

    from faster_whisper import WhisperModel
    model = WhisperModel(config.WHISPER_MODEL, device=config.WHISPER_DEVICE,
                         compute_type=config.WHISPER_COMPUTE_TYPE)
    print("model loaded")

    # 1) sanity: direct transcription of the whole clip
    segs, _ = model.transcribe(audio, language=config.WHISPER_LANGUAGE, beam_size=1)
    print("DIRECT  ->", repr(" ".join(s.text.strip() for s in segs).strip()))

    # 2) through the real streaming worker, frame by frame, then trailing silence
    bus = EventBus()
    finals: list[str] = []
    bus.subscribe(TranscriptUpdate, lambda e: finals.append(e.text) if e.is_final else None)
    bus.start()
    w = _StreamWorker("other", bus, model, threading.Lock())  # 'other' => gain=1 path
    w.start()

    n = config.FRAME_SAMPLES
    frames = [audio[i:i + n] for i in range(0, len(audio) - n, n)]
    silence = [np.zeros(n, dtype=np.float32)] * 40  # ~1.2s trailing silence -> final
    from events import AudioFrame
    for fr in frames + silence:
        w.submit(AudioFrame("other", fr))           # feed the worker directly
        time.sleep(config.FRAME_MS / 1000 / 4)  # quasi-real-time, sped up

    # wait until the worker drains its backlog and emits the end-of-utterance final
    deadline = time.time() + 30
    while time.time() < deadline and not finals:
        time.sleep(0.2)
    print("WORKER FINALS ->", finals)
    bus.stop()
    w.stop()


if __name__ == "__main__":
    main()
