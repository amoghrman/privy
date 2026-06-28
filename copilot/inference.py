"""inference: AskRequested -> streamed AnswerToken events.

Two layers, deliberately separated so the LLM backend is swappable:

  - InferenceProvider (ABC)   — the seam. `stream(system, user, cancelled)`
    yields text deltas. `OllamaInferenceProvider` is the local default. A future
    cloud provider (Claude/OpenAI) or a bigger-hardware Ollama can drop in behind
    THIS interface with zero pipeline changes — same idea as the memory seam.

  - InferenceEngine           — wiring: listens for AskRequested, asks the
    ContextBuilder for the prompt, drives the provider, publishes AnswerToken.
    Serialises asks and cancels a stale generation if a newer ask arrives.

Local-first: OllamaInferenceProvider talks only to localhost. Choosing a cloud
provider would send the transcript off-device — a deliberate, opt-in decision.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Callable, Iterator

import httpx

import config
from events import AnswerToken, AskRequested, EventBus, StatusUpdate
from copilot.context_builder import ContextBuilder


# ─────────────────────────── the seam ───────────────────────────

class InferenceProvider(ABC):
    @abstractmethod
    def stream(
        self, system: str, user: str, cancelled: Callable[[], bool]
    ) -> Iterator[str]:
        """Yield answer text deltas. Stop early if `cancelled()` returns True."""
        ...

    def warmup(self) -> str | None:
        """Optionally preload. Return a status string, or None."""
        return None


class OllamaInferenceProvider(InferenceProvider):
    """Local default: streams from the Ollama server on localhost."""

    def __init__(self) -> None:
        self.host = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL

    def _options(self) -> dict:
        return {
            "temperature": config.OLLAMA_TEMPERATURE,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "num_gpu": config.OLLAMA_NUM_GPU,
            "num_ctx": config.OLLAMA_NUM_CTX,
        }

    def warmup(self) -> str | None:
        httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {**self._options(), "num_predict": 1},
            },
            timeout=180,
        )
        return f"LLM ready: {self.model}"

    def stream(
        self, system: str, user: str, cancelled: Callable[[], bool]
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": self._options(),
        }
        with httpx.stream(
            "POST", f"{self.host}/api/chat", json=payload, timeout=None
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if cancelled():
                    return
                if not line:
                    continue
                obj = json.loads(line)
                delta = obj.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if obj.get("done"):
                    return


class OpenAICompatibleProvider(InferenceProvider):
    """Cloud provider (opt-in): OpenAI-compatible streaming chat completions.

    Works for any OpenAI-compatible endpoint — Groq and NVIDIA NIM (Nemotron)
    are just different (base_url, model, key) presets. Higher accuracy than the
    local model, but the transcript leaves the device. Requires an API key in
    the given environment variable.
    """

    def __init__(self, name: str, base: str, model: str, key_env: str, signup: str) -> None:
        self.name = name
        self.base = base
        self.model = model
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"{key_env} is not set. Get a free key at {signup} and set it, "
                f'e.g.  setx {key_env} "your_key_here"  (then restart the shell).'
            )

    def warmup(self) -> str | None:
        return f"Cloud LLM ({self.name}): {self.model}"

    def stream(
        self, system: str, user: str, cancelled: Callable[[], bool]
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "temperature": config.OLLAMA_TEMPERATURE,
            "max_tokens": config.OLLAMA_NUM_PREDICT,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.stream(
            "POST", f"{self.base}/chat/completions",
            json=payload, headers=headers, timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if cancelled():
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    yield delta


def make_provider() -> InferenceProvider:
    """Pick the inference backend from config (Ollama local by default)."""
    backend = config.INFERENCE_BACKEND
    if backend == "groq":
        return OpenAICompatibleProvider(
            "Groq", config.GROQ_API_BASE, config.GROQ_MODEL,
            config.GROQ_API_KEY_ENV, "https://console.groq.com",
        )
    if backend == "nvidia":
        return OpenAICompatibleProvider(
            "NVIDIA NIM", config.NVIDIA_API_BASE, config.NVIDIA_MODEL,
            config.NVIDIA_API_KEY_ENV, "https://build.nvidia.com",
        )
    return OllamaInferenceProvider()


# ─────────────────────────── engine (wiring) ───────────────────────────

class InferenceEngine:
    def __init__(
        self,
        bus: EventBus,
        builder: ContextBuilder,
        provider: InferenceProvider | None = None,
    ) -> None:
        self.bus = bus
        self.builder = builder
        self.provider = provider or make_provider()
        self._cv = threading.Condition()
        self._pending = False
        self._stop = False
        self._gen = 0
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)

    def start(self) -> None:
        self.bus.subscribe(AskRequested, self._on_ask)
        self._thread.start()
        threading.Thread(target=self._warmup, name="llm-warmup", daemon=True).start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._gen += 1
            self._cv.notify()

    def _warmup(self) -> None:
        try:
            msg = self.provider.warmup()
            if msg:
                self.bus.publish(StatusUpdate(msg))
        except Exception as e:  # noqa: BLE001
            self.bus.publish(StatusUpdate(f"LLM warmup failed: {e}"))

    # ── coalesce asks + cancel stale generations ──────────────
    def _on_ask(self, e: AskRequested) -> None:
        with self._cv:
            self._pending = True
            self._gen += 1            # bump => any in-flight generation cancels
            self._cv.notify()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop:
                    break
                self._pending = False
                gen = self._gen
            self._run_inference(gen)

    def _cancelled(self, gen: int) -> bool:
        return gen != self._gen

    def _run_inference(self, gen: int) -> None:
        system, user = self.builder.build()
        first = True
        try:
            for delta in self.provider.stream(system, user, lambda: self._cancelled(gen)):
                if self._cancelled(gen):
                    return
                self.bus.publish(AnswerToken(text=delta, first=first))
                first = False
            if not self._cancelled(gen):
                self.bus.publish(AnswerToken(text="", done=True))
        except Exception as e:  # noqa: BLE001
            if not self._cancelled(gen):
                self.bus.publish(
                    AnswerToken(text=f"[inference error: {e}]", first=False, done=True)
                )
