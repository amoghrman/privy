"""Event bus + event payloads.

The whole app is wired through a tiny in-process pub/sub bus. Modules never
import each other directly for runtime data flow — they publish and subscribe
to typed events. This keeps the pipeline decoupled (and makes the memory seam,
overlay, etc. swappable).

Dispatch happens on a single background thread so publishers never block on a
slow subscriber. Subscribers that do heavy work (transcription, inference)
should hand off to their own thread/queue rather than block the dispatcher.
"""
from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from time import time
from typing import Callable

import numpy as np


# ─────────────────────────── Event payloads ───────────────────────────

Speaker = str  # "me" | "other"


@dataclass
class AudioFrame:
    """A single 30 ms mono frame at 16 kHz, float32 in [-1, 1]."""
    speaker: Speaker
    samples: np.ndarray
    timestamp: float = field(default_factory=time)


@dataclass
class TranscriptUpdate:
    """A streaming transcript result. `is_final` marks end-of-utterance."""
    speaker: Speaker
    text: str
    is_final: bool
    timestamp: float = field(default_factory=time)


@dataclass
class AskRequested:
    """User (hotkey) or auto-surface asked the copilot for an answer."""
    reason: str = "hotkey"        # "hotkey" | "auto"
    timestamp: float = field(default_factory=time)


@dataclass
class AnswerToken:
    """A streamed token from the LLM. `done` marks the end of the answer."""
    text: str
    done: bool = False
    first: bool = False           # True for the very first token (latency marker)
    timestamp: float = field(default_factory=time)


@dataclass
class ToggleOverlay:
    timestamp: float = field(default_factory=time)


@dataclass
class StatusUpdate:
    """Human-readable status for the overlay footer / logs."""
    text: str
    timestamp: float = field(default_factory=time)


# ─────────────────────────── Bus ───────────────────────────

class EventBus:
    """Thread-safe typed pub/sub with a single dispatch thread."""

    def __init__(self) -> None:
        self._subs: dict[type, list[Callable]] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue[object]" = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def subscribe(self, event_type: type, handler: Callable[[object], None]) -> None:
        with self._lock:
            self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: object) -> None:
        self._q.put(event)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="event-bus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._q.put(_SHUTDOWN)

    def _loop(self) -> None:
        while self._running:
            event = self._q.get()
            if event is _SHUTDOWN:
                break
            with self._lock:
                handlers = list(self._subs.get(type(event), ()))
            for handler in handlers:
                try:
                    handler(event)
                except Exception:  # never let one subscriber kill the bus
                    traceback.print_exc()


_SHUTDOWN = object()
