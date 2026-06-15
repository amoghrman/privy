"""transcriber: AudioFrame stream -> streaming TranscriptUpdate events.

Per speaker ("me" / "other") we run an independent VAD-gated stream worker.
A worker buffers speech, and:
  - emits a PARTIAL transcript every PARTIAL_INTERVAL_MS while speech continues
    (we do NOT wait for full silence), and
  - emits a FINAL transcript when trailing silence marks end-of-utterance.

A single faster-whisper model is shared across both workers behind a lock
(int8 on CPU is cheap enough that serialized calls stay well under budget).
"""
from __future__ import annotations

import os
import queue
import threading
from time import time

_DEBUG = os.environ.get("COPILOT_DEBUG") == "1"

import numpy as np

import config
from events import AudioFrame, EventBus, StatusUpdate, TranscriptUpdate


class _StreamWorker:
    """VAD-gated streaming transcription for one speaker."""

    def __init__(self, speaker: str, bus: EventBus, model, model_lock: threading.Lock):
        import webrtcvad

        self.speaker = speaker
        self.bus = bus
        self.model = model
        self.model_lock = model_lock
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        # Two stages, two threads:
        #   _ingest_loop : VAD + buffering (cheap, MUST keep real-time)
        #   _stt_loop    : transcription (slow); coalesces partials so it can
        #                  never fall behind and stall end-of-utterance detection.
        self.q: "queue.Queue[AudioFrame | None]" = queue.Queue()
        self._cv = threading.Condition()
        self._pending_partial: np.ndarray | None = None   # latest snapshot only
        self._pending_finals: list[np.ndarray] = []
        self._stop = False
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop, name=f"vad-{speaker}", daemon=True
        )
        self._stt_thread = threading.Thread(
            target=self._stt_loop, name=f"stt-{speaker}", daemon=True
        )

        # framing math
        self._silence_end_frames = config.SILENCE_END_MS // config.FRAME_MS
        self._min_speech_frames = config.MIN_SPEECH_MS // config.FRAME_MS
        self._preroll_frames = config.PREROLL_MS // config.FRAME_MS
        self._max_frames = config.MAX_UTTERANCE_MS // config.FRAME_MS
        self._partial_interval_frames = max(1, config.PARTIAL_INTERVAL_MS // config.FRAME_MS)

    def start(self) -> None:
        self._stt_thread.start()
        self._ingest_thread.start()

    def submit(self, frame: AudioFrame) -> None:
        self.q.put(frame)

    def stop(self) -> None:
        self.q.put(None)
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _is_speech(self, samples: np.ndarray) -> bool:
        pcm16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm16, config.SAMPLE_RATE)

    # ── stage 1: VAD + buffering (real-time) ──────────────────
    def _ingest_loop(self) -> None:
        preroll: list[np.ndarray] = []
        speech: list[np.ndarray] = []
        in_speech = False
        silence = 0
        frames_since_partial = 0

        while True:
            frame = self.q.get()
            if frame is None:
                break
            samples = frame.samples
            voiced = self._is_speech(samples)

            if not in_speech:
                preroll.append(samples)
                if len(preroll) > self._preroll_frames:
                    preroll.pop(0)
                if voiced:
                    in_speech = True
                    speech = preroll + [samples]
                    preroll = []
                    silence = 0
                    frames_since_partial = 0
                    if _DEBUG:
                        print(f"[dbg {self.speaker}] utterance START")
                continue

            # in speech
            speech.append(samples)
            silence = 0 if voiced else silence + 1
            frames_since_partial += 1

            if silence >= self._silence_end_frames or len(speech) >= self._max_frames:
                self._queue_final(np.concatenate(speech).astype(np.float32))
                if _DEBUG:
                    print(f"[dbg {self.speaker}] end-of-utterance "
                          f"({len(speech) * config.FRAME_MS / 1000:.1f}s)")
                in_speech = False
                speech = []
                silence = 0
                continue

            if (len(speech) >= self._min_speech_frames
                    and frames_since_partial >= self._partial_interval_frames):
                frames_since_partial = 0
                self._queue_partial(np.concatenate(speech).astype(np.float32))

    def _queue_partial(self, audio: np.ndarray) -> None:
        with self._cv:
            self._pending_partial = audio   # coalesce: keep only the newest
            self._cv.notify()

    def _queue_final(self, audio: np.ndarray) -> None:
        with self._cv:
            self._pending_finals.append(audio)
            self._pending_partial = None     # drop stale partial for this utterance
            self._cv.notify()

    # ── stage 2: transcription (slow; finals prioritised) ─────
    def _stt_loop(self) -> None:
        last_partial_text = ""
        while True:
            with self._cv:
                while not self._stop and not self._pending_finals and self._pending_partial is None:
                    self._cv.wait()
                if self._stop and not self._pending_finals and self._pending_partial is None:
                    break
                if self._pending_finals:
                    audio, is_final = self._pending_finals.pop(0), True
                else:
                    audio, is_final = self._pending_partial, False
                    self._pending_partial = None

            text = self._transcribe(audio)
            if _DEBUG and is_final:
                print(f"[dbg {self.speaker}] FINAL -> {text!r}")
            if not text:
                continue
            if is_final:
                last_partial_text = ""
                self.bus.publish(TranscriptUpdate(self.speaker, text, is_final=True))
            elif text != last_partial_text:
                last_partial_text = text
                self.bus.publish(TranscriptUpdate(self.speaker, text, is_final=False))

    def _apply_gain(self, audio: np.ndarray) -> np.ndarray:
        # Manual MIC_GAIN/SYSTEM_GAIN is applied upstream at capture (pre-VAD).
        # Here we only do per-utterance auto-leveling toward the target peak.
        if config.AUTO_GAIN:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1e-4 and peak < config.AUTO_GAIN_TARGET:
                gain = min(config.AUTO_GAIN_MAX, config.AUTO_GAIN_TARGET / peak)
                audio = audio * gain
        return np.clip(audio, -1.0, 1.0)

    def _transcribe(self, raw: np.ndarray) -> str:
        audio = self._apply_gain(raw)
        if _DEBUG:
            print(f"[dbg {self.speaker}] peak raw={float(np.max(np.abs(raw))):.3f} "
                  f"-> fed={float(np.max(np.abs(audio))):.3f}")
        with self.model_lock:
            segments, _ = self.model.transcribe(
                audio,
                language=config.WHISPER_LANGUAGE,
                beam_size=config.WHISPER_BEAM_SIZE,
                vad_filter=False,                  # we gate with webrtcvad upstream
                condition_on_previous_text=False,  # avoid runaway hallucination
            )
            kept = [s.text.strip() for s in segments if self._keep_segment(s)]
            return " ".join(kept).strip()

    @staticmethod
    def _keep_segment(seg) -> bool:
        """Reject Whisper's low-confidence / non-speech hallucinations."""
        if getattr(seg, "no_speech_prob", 0.0) > config.HALLUCINATION_NO_SPEECH_PROB:
            return False
        if getattr(seg, "avg_logprob", 0.0) < config.HALLUCINATION_MIN_LOGPROB:
            return False
        if getattr(seg, "compression_ratio", 0.0) > config.HALLUCINATION_MAX_COMPRESSION:
            return False
        return True


class Transcriber:
    """Subscribes to AudioFrame, routes per speaker, publishes TranscriptUpdate."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._model = None
        self._lock = threading.Lock()
        self._workers: dict[str, _StreamWorker] = {}

    def _load_model(self):
        from faster_whisper import WhisperModel

        self.bus.publish(
            StatusUpdate(
                f"loading Whisper '{config.WHISPER_MODEL}' "
                f"({config.WHISPER_COMPUTE_TYPE}/{config.WHISPER_DEVICE})…"
            )
        )
        return WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def start(self) -> None:
        self._model = self._load_model()
        for speaker in ("me", "other"):
            w = _StreamWorker(speaker, self.bus, self._model, self._lock)
            w.start()
            self._workers[speaker] = w
        self.bus.subscribe(AudioFrame, self._on_frame)
        self.bus.publish(StatusUpdate("transcriber ready"))

    def _on_frame(self, event: AudioFrame) -> None:
        w = self._workers.get(event.speaker)
        if w is not None:
            w.submit(event)

    def stop(self) -> None:
        for w in self._workers.values():
            w.stop()
