"""Milestone 2 test (no audio needed).

1) Proves the memory seam: builds a prompt and shows the
   "Relevant context you remember:" injection (with a fake provider so you can
   see recall() output land in the prompt).
2) If Ollama is running, fires an AskRequested and prints the streamed answer
   plus first-token latency.

    python test_ask.py
"""
from __future__ import annotations

import sys
import time

from events import AnswerToken, AskRequested, EventBus, StatusUpdate, TranscriptUpdate
from copilot.memory import MemoryProvider, NullMemoryProvider
from copilot.context_builder import ContextBuilder
from copilot.inference import InferenceEngine


class DemoMemory(MemoryProvider):
    """Stand-in to visibly prove recall() output reaches the prompt.
    (Swap for NullMemoryProvider to see the real MVP default.)"""

    def recall(self, query: str, k: int = 5) -> list[str]:
        return [
            "The user is the founder of a startup called 'privy'.",
            "privy is a local-first, privacy-preserving memory layer.",
        ][:k]

    def remember(self, text: str, meta: dict | None = None) -> None:
        pass


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    use_null = "--null" in sys.argv
    memory = NullMemoryProvider() if use_null else DemoMemory()
    print(f"memory provider: {type(memory).__name__}\n")

    bus = EventBus()
    bus.subscribe(StatusUpdate, lambda e: print(f"· {e.text}"))
    bus.start()

    builder = ContextBuilder(bus, memory)
    builder.start()

    # Inject a fake conversation (as if the transcriber produced these finals)
    bus.publish(TranscriptUpdate("me", "Hey, thanks for hopping on.", is_final=True))
    bus.publish(TranscriptUpdate("other", "So what exactly does your product do, and how is it different?", is_final=True))
    time.sleep(0.2)  # let the window fill

    system, user = builder.build()
    print("================ SYSTEM ================\n" + system)
    print("\n================ USER (prompt) ================\n" + user)
    print("\n" + "=" * 48)

    # Now stream an answer (requires Ollama running)
    ask_time = {"t": 0.0}
    first_seen = {"t": 0.0}

    def on_token(e: AnswerToken) -> None:
        if e.first:
            first_seen["t"] = time.time()
            print(f"\n[first token in {first_seen['t'] - ask_time['t']:.2f}s]\n", flush=True)
        if e.text:
            sys.stdout.write(e.text)
            sys.stdout.flush()
        if e.done:
            print("\n\n[done]")

    bus.subscribe(AnswerToken, on_token)
    engine = InferenceEngine(bus, builder)
    engine.start()

    print("\nAsking Ollama (Ctrl+C to quit if it hangs)…")
    time.sleep(1.0)  # give warmup a beat
    ask_time["t"] = time.time()
    bus.publish(AskRequested(reason="test"))

    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        bus.stop()


if __name__ == "__main__":
    main()
