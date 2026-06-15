"""context_builder: recent transcript + recalled memory -> LLM prompt.

Tracks a rolling window of FINAL transcripts and, on every turn, builds a
(system, user) prompt pair. It ALWAYS calls `memory.recall(...)` and injects
the results under a "Relevant context you remember:" section — even when the
provider returns nothing today (NullMemoryProvider). The pipeline depends only
on the `MemoryProvider` ABC, never on a concrete implementation.

Also optionally auto-surfaces: when the other speaker asks something that looks
like a question, it publishes `AskRequested(reason="auto")`.
"""
from __future__ import annotations

import re
from collections import deque
from time import time

import config
from events import AskRequested, EventBus, TranscriptUpdate
from copilot.memory import MemoryProvider

_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(who|what|when|where|why|how|which|can|could|would|should|"
    r"do|does|did|is|are|was|were|will|tell me|explain|walk me)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a real-time meeting copilot running locally on the user's machine. "
    "The user ('me') is in a live conversation with another person ('other'). "
    "Give a brief, direct, immediately useful answer the user can act on or say "
    "next. Prefer 2-5 short sentences or tight bullets. No preamble, no "
    "'as an AI', no restating the question. If unsure, say what you'd need."
)


class ContextBuilder:
    def __init__(self, bus: EventBus, memory: MemoryProvider) -> None:
        self.bus = bus
        self.memory = memory
        self._window: deque[tuple[float, str, str]] = deque()  # (ts, speaker, text)

    def start(self) -> None:
        self.bus.subscribe(TranscriptUpdate, self._on_transcript)

    # ── transcript window ─────────────────────────────────────
    def _on_transcript(self, e: TranscriptUpdate) -> None:
        if not e.is_final or not e.text.strip():
            return
        self._window.append((e.timestamp, e.speaker, e.text.strip()))
        self._evict()
        if config.AUTO_SURFACE and e.speaker == "other" and self._looks_like_question(e.text):
            self.bus.publish(AskRequested(reason="auto"))

    def _evict(self) -> None:
        cutoff = time() - config.TRANSCRIPT_WINDOW_SECONDS
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return bool(_QUESTION_RE.search(text.strip()))

    # ── prompt assembly ───────────────────────────────────────
    def _latest_question(self) -> str:
        """Best guess at what needs answering: last 'other' line, else last line."""
        for ts, speaker, text in reversed(self._window):
            if speaker == "other":
                return text
        return self._window[-1][2] if self._window else ""

    def _render_transcript(self) -> str:
        lines = []
        for _, speaker, text in self._window:
            who = "Me" if speaker == "me" else "Other"
            lines.append(f"{who}: {text}")
        return "\n".join(lines) if lines else "(no transcript yet)"

    def build(self) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for the inference layer."""
        self._evict()
        question = self._latest_question()
        memories = self.memory.recall(question, k=config.MEMORY_RECALL_K)

        if memories:
            memory_block = "Relevant context you remember:\n" + "\n".join(
                f"- {m}" for m in memories
            )
        else:
            memory_block = "Relevant context you remember:\n- (none)"

        user_prompt = (
            f"{memory_block}\n\n"
            f"Recent conversation (most recent last):\n{self._render_transcript()}\n\n"
            f"The other person just said / asked:\n\"{question}\"\n\n"
            f"Give me the answer or the helpful next thing to say."
        )
        return SYSTEM_PROMPT, user_prompt
