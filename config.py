"""Central configuration for the meeting copilot.

Everything here is local-only. No remote endpoints other than the Ollama
server you run yourself on localhost.
"""
from __future__ import annotations

# ─────────────────────────── Audio ───────────────────────────
SAMPLE_RATE = 16_000          # canonical rate fed to Whisper + VAD
FRAME_MS = 30                 # VAD frame size (10/20/30 ms only)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples @ 16 kHz

# Device selection. None = auto-pick a sensible default.
#   - MIC_DEVICE: sounddevice input index (run `python run_transcribe.py --list-devices`)
#   - SYSTEM_DEVICE: pyaudiowpatch loopback index; None = default render device's loopback
MIC_DEVICE: int | None = None
SYSTEM_DEVICE: int | None = None

# ─────────────────────────── VAD / streaming ───────────────────────────
VAD_AGGRESSIVENESS = 2        # 0..3, higher = more aggressive at calling non-speech
SILENCE_END_MS = 600          # trailing silence that marks end-of-utterance
MIN_SPEECH_MS = 200           # ignore blips shorter than this
PARTIAL_INTERVAL_MS = 500     # how often to emit a streaming partial while talking
PREROLL_MS = 300              # audio kept before VAD trips, so we don't clip onsets
MAX_UTTERANCE_MS = 20_000     # force a final if someone monologues forever

# ─────────────────────────── Input gain / leveling ───────────────────────────
# Many built-in mics capture very quietly. We peak-normalize each utterance up
# to AUTO_GAIN_TARGET before transcription (never attenuate — loud system audio
# is left alone). MIC_GAIN/SYSTEM_GAIN are extra manual multipliers if needed.
AUTO_GAIN = True
AUTO_GAIN_TARGET = 0.25       # target peak amplitude fed to Whisper
AUTO_GAIN_MAX = 60.0          # cap so we don't amplify silence into noise
# MIC_GAIN is applied at CAPTURE (before VAD) so VAD sees a healthy level too.
# Keep it modest to avoid clipping (which garbles STT); per-utterance AUTO_GAIN
# normalizes each utterance to AUTO_GAIN_TARGET for Whisper without clipping.
MIC_GAIN = 6.0
SYSTEM_GAIN = 1.0

# Drop low-confidence / non-speech Whisper segments (kills "Thank you"/"You"
# style hallucinations on quiet or noisy gaps).
HALLUCINATION_NO_SPEECH_PROB = 0.6   # drop seg if model thinks it's non-speech
HALLUCINATION_MIN_LOGPROB = -1.0     # drop seg if avg token logprob below this
HALLUCINATION_MAX_COMPRESSION = 2.4  # drop seg if text is hyper-repetitive

# ─────────────────────────── Whisper (faster-whisper) ───────────────────────────
WHISPER_MODEL = "small.en"    # English-only, much better accuracy than base
WHISPER_DEVICE = "cpu"        # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = "int8" # int8 on CPU; "int8_float16"/"float16" on CUDA
WHISPER_LANGUAGE = "en"       # set None to autodetect (slower)
WHISPER_BEAM_SIZE = 1         # greedy = fastest, fine for live partials

# ─────────────────────────── LLM backend selection ───────────────────────────
# "ollama" = 100% local (default, private). "groq"/"nvidia" = cloud APIs for
# higher accuracy/speed — NOTE: these send the transcript off-device. Opt-in.
# Each cloud backend needs an API key in its env var (never commit keys).
INFERENCE_BACKEND = "ollama"           # "ollama" | "groq" | "nvidia"

# ─────────────────────────── LLM (Ollama, local) ───────────────────────────
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct" # Sized for this machine: 7.7GB RAM,
                                       # NVIDIA MX350 w/ only 2GB VRAM. Larger
                                       # models thrash. First-token ~2.6s here
                                       # (a hardware floor); swap providers/HW
                                       # to hit <2s. See inference.py seam.
OLLAMA_KEEP_ALIVE = "30m"              # keep model resident between asks
OLLAMA_NUM_PREDICT = 256               # cap answer length for snappiness
OLLAMA_NUM_GPU = 99                    # offload all layers; MX350 has spare VRAM
OLLAMA_NUM_CTX = 2048                  # smaller context = faster prompt eval
OLLAMA_TEMPERATURE = 0.3

# ─────────────────────────── LLM (Groq, cloud — opt-in) ───────────────────────────
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile" # large, accurate, still fast on Groq
GROQ_API_KEY_ENV = "GROQ_API_KEY"      # read from this env var; never hard-code

# ─────────────────────────── LLM (NVIDIA NIM / Nemotron, cloud — opt-in) ───────────
# Free tier at build.nvidia.com (nvapi- key, 1000 credits, 40 req/min, no card).
# OpenAI-compatible. Nemotron-3 options:
#   nvidia/nemotron-3-nano-30b-a3b   (fast)
#   nvidia/nemotron-3-super-120b-a12b (best accuracy)  ← default
#   nvidia/nemotron-3-ultra-550b-a55b (largest)
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"  # read from this env var; never hard-code

# ─────────────────────────── Context builder ───────────────────────────
TRANSCRIPT_WINDOW_SECONDS = 90   # how much recent conversation to feed the LLM
MEMORY_RECALL_K = 5              # how many memories to pull each turn

# ─────────────────────────── Overlay / hotkeys ───────────────────────────
HOTKEY_ASK = "<ctrl>+<alt>+<space>"   # trigger an answer for the latest question
HOTKEY_TOGGLE = "<ctrl>+<alt>+h"      # show/hide the overlay
AUTO_SURFACE = True                   # auto-ask when a question from "other" is detected
OVERLAY_WIDTH = 460
OVERLAY_HEIGHT = 360
OVERLAY_OPACITY = 0.92
