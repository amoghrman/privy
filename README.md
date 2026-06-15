# privy — local-first meeting copilot

A privacy-preserving, **on-device** real-time meeting copilot for **Windows**. It
captures your **mic + system audio**, transcribes **live**, and shows fast AI
answers in a small **always-on-top overlay that is excluded from screen-share /
recording capture**. Trigger help with a global hotkey, or let it auto-surface
an answer when the other person asks a question.

By default **everything runs locally** — faster-whisper for speech-to-text and a
local LLM via Ollama. No audio or transcript leaves your machine. (An optional
cloud LLM backend exists for higher accuracy — strictly opt-in; see below.)

> A meeting copilot — **not** an exam/proctoring tool.

## Features
- 🎙️ Mic + system audio capture (WASAPI loopback), tagged `me` / `other`
- ⚡ Streaming partial transcripts (VAD-gated; doesn't wait for silence)
- 🧠 Local LLM via Ollama, streamed token-by-token, model kept resident
- 🪟 Overlay that is **invisible to Zoom/Meet/OBS capture** (`SetWindowDisplayAffinity`)
- ⌨️ Global hotkeys + auto-surface on detected questions
- 🔌 Clean **memory seam** and **inference seam** for swapping implementations

## Architecture

Event-driven — modules talk only through an in-process `EventBus`, never direct calls.

```
audio_capture ─AudioFrame▶ transcriber ─TranscriptUpdate▶ context_builder
                                                               │ memory.recall()
                                                               ▼
overlay ◀─AnswerToken─ inference ◀─(system,user)─ context_builder
   ▲ hotkey ─AskRequested────────────────┘
```

| Module | Role |
|---|---|
| `copilot/audio_capture.py` | mic + system → 16 kHz mono frames |
| `copilot/transcriber.py` | faster-whisper, VAD-gated streaming partials |
| `copilot/memory.py` | **`MemoryProvider` seam** + `NullMemoryProvider` |
| `copilot/context_builder.py` | transcript window + `recall()` → prompt |
| `copilot/inference.py` | **`InferenceProvider` seam**: Ollama (local) / Groq (cloud) |
| `copilot/overlay.py` | capture-excluded always-on-top UI |
| `app.py` | wires it all together |

### The memory seam
`context_builder` depends only on the abstract `MemoryProvider` (`recall` /
`remember`) and injects results under a "Relevant context you remember:" section
every turn. The MVP ships `NullMemoryProvider` (no-ops). A local SQLite/HippoRAG
provider can drop in behind the **exact** interface with zero pipeline changes.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull a model sized for your machine:
```powershell
ollama pull qwen2.5:1.5b-instruct    # light; good on 2GB GPU / low RAM
# bigger/better if you have the VRAM:  ollama pull qwen2.5:7b-instruct
```
Set the model in [`config.py`](config.py) (`OLLAMA_MODEL`).

System audio needs **no virtual cable** on Windows — we use WASAPI loopback on
your default playback device.

## Run

```powershell
python app.py
```
- Overlay appears top-right (drag to move).
- **`Ctrl+Alt+Space`** — answer the latest question on demand.
- **`Ctrl+Alt+H`** — show/hide the overlay.
- Questions from the *other* speaker auto-surface an answer.
- `COPILOT_DEBUG=1 python app.py` logs transcripts/asks/answers to the console.

A **desktop shortcut** (“privy”) is created by `make_shortcut.ps1`.

### Optional: Groq cloud backend (higher accuracy, opt-in)
The local 1.5B model is fast but limited. For sharper answers you can route the
**LLM** (not the audio/STT) to [Groq](https://console.groq.com)'s free API.
⚠️ This sends the transcript off-device — opt-in by design.

```powershell
setx GROQ_API_KEY "gsk_your_key_here"     # restart shell after
# then in config.py:  INFERENCE_BACKEND = "groq"
```
STT stays 100% local either way.

## Hardware notes
First-token latency depends heavily on your GPU/RAM. On a 2 GB MX350 / 8 GB RAM
laptop, a small local model gives ~2.5–3 s to first token; the cloud backend or
better hardware gets you under the 2 s target. The model and gain defaults in
`config.py` are tuned for that low-end machine — raise them if you have more.

## Roadmap (v2)
- **HippoRAG memory provider** behind the existing `MemoryProvider` seam
  (persistent, retrieval-augmented meeting memory).
- Configurable settings UI; macOS support.

## Dev / test harnesses
- `run_transcribe.py` — transcription only (mic + system)
- `test_stt_offline.py` — deterministic STT test via synthesized speech
- `test_ask.py` — memory-seam + prompt + inference (no audio)
- `run_ask.py` — full pipeline minus overlay (press Enter to ask)
- `overlay_smoke.py` — overlay + capture-exclusion smoke test
