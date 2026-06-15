"""audio_capture: mic + system audio -> tagged 16 kHz mono frames.

Two sources, each tagged with a speaker:
  - mic    -> "me"    (sounddevice input stream)
  - system -> "other" (WASAPI loopback via pyaudiowpatch)

Each source captures at the device's native rate/channels, gets downmixed to
mono and resampled to 16 kHz, then sliced into exact 30 ms frames and published
as `AudioFrame` events. Nothing here touches the network.
"""
from __future__ import annotations

import threading
from math import gcd

import numpy as np
from scipy.signal import resample_poly

import config
from events import AudioFrame, EventBus, StatusUpdate


def _to_mono_f32(data: np.ndarray, channels: int) -> np.ndarray:
    """Downmix interleaved/2D audio to mono float32."""
    if data.dtype != np.float32:
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.reshape(-1)


def _resample_to_16k(mono: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == config.SAMPLE_RATE:
        return mono
    g = gcd(src_rate, config.SAMPLE_RATE)
    up = config.SAMPLE_RATE // g
    down = src_rate // g
    return resample_poly(mono, up, down).astype(np.float32)


class _FrameChopper:
    """Accumulates resampled mono audio and emits exact FRAME_SAMPLES frames."""

    def __init__(self, speaker: str, bus: EventBus, gain: float = 1.0) -> None:
        self.speaker = speaker
        self.bus = bus
        self.gain = gain
        self._buf = np.empty(0, dtype=np.float32)

    def feed(self, mono16k: np.ndarray) -> None:
        if self.gain != 1.0:
            mono16k = np.clip(mono16k * self.gain, -1.0, 1.0)
        self._buf = np.concatenate((self._buf, mono16k))
        n = config.FRAME_SAMPLES
        while len(self._buf) >= n:
            frame = self._buf[:n].copy()
            self._buf = self._buf[n:]
            self.bus.publish(AudioFrame(speaker=self.speaker, samples=frame))


class AudioCapture:
    """Owns both capture streams and publishes AudioFrame events."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._mic_stream = None
        self._sys_thread: threading.Thread | None = None
        self._sys_stop = threading.Event()
        self._mic_chopper = _FrameChopper("me", bus, gain=config.MIC_GAIN)
        self._sys_chopper = _FrameChopper("other", bus, gain=config.SYSTEM_GAIN)

    # ── microphone (sounddevice) ──────────────────────────────
    def _start_mic(self) -> None:
        import sounddevice as sd

        dev = config.MIC_DEVICE
        info = sd.query_devices(dev, "input") if dev is not None else sd.query_devices(
            sd.default.device[0], "input"
        )
        src_rate = int(info["default_samplerate"])
        channels = 1  # ask the driver for mono; it'll downmix if it can

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                pass  # over/underflows are non-fatal for an MVP
            mono = _to_mono_f32(indata.copy(), channels)
            self._mic_chopper.feed(_resample_to_16k(mono, src_rate))

        self._mic_stream = sd.InputStream(
            device=dev,
            samplerate=src_rate,
            channels=channels,
            dtype="float32",
            blocksize=0,
            callback=callback,
        )
        self._mic_stream.start()
        self.bus.publish(StatusUpdate(f"mic: {info['name']} @ {src_rate} Hz"))

    # ── system audio (WASAPI loopback) ────────────────────────
    def _resolve_loopback(self, pa):  # noqa: ANN001
        """Find the loopback device for the default render endpoint."""
        if config.SYSTEM_DEVICE is not None:
            return pa.get_device_info_by_index(config.SYSTEM_DEVICE)
        import pyaudiowpatch as pyaudio

        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if not speakers.get("isLoopbackDevice", False):
            for lb in pa.get_loopback_device_info_generator():
                if speakers["name"] in lb["name"]:
                    return lb
            raise RuntimeError(
                "No WASAPI loopback device found. Is any audio endpoint enabled?"
            )
        return speakers

    def _start_system(self) -> None:
        import pyaudiowpatch as pyaudio

        def run() -> None:
            pa = pyaudio.PyAudio()
            try:
                dev = self._resolve_loopback(pa)
                src_rate = int(dev["defaultSampleRate"])
                channels = int(dev["maxInputChannels"])
                frames_per_buffer = max(256, int(src_rate * config.FRAME_MS / 1000))

                def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
                    arr = np.frombuffer(in_data, dtype=np.int16)
                    mono = _to_mono_f32(arr, channels)
                    self._sys_chopper.feed(_resample_to_16k(mono, src_rate))
                    return (None, pyaudio.paContinue)

                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=src_rate,
                    frames_per_buffer=frames_per_buffer,
                    input=True,
                    input_device_index=dev["index"],
                    stream_callback=callback,
                )
                self.bus.publish(
                    StatusUpdate(f"system: {dev['name']} @ {src_rate} Hz (loopback)")
                )
                stream.start_stream()
                while not self._sys_stop.is_set() and stream.is_active():
                    self._sys_stop.wait(0.1)
                stream.stop_stream()
                stream.close()
            finally:
                pa.terminate()

        self._sys_thread = threading.Thread(target=run, name="system-audio", daemon=True)
        self._sys_thread.start()

    # ── lifecycle ─────────────────────────────────────────────
    def start(self) -> None:
        self._start_mic()
        self._start_system()

    def stop(self) -> None:
        if self._mic_stream is not None:
            self._mic_stream.stop()
            self._mic_stream.close()
        self._sys_stop.set()
        if self._sys_thread is not None:
            self._sys_thread.join(timeout=2)


def list_devices() -> str:
    """Return a human-readable dump of input + loopback devices."""
    lines: list[str] = ["── sounddevice (mic) inputs ──"]
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            lines.append(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch)")

    lines.append("── WASAPI loopback (system) ──")
    try:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        for lb in pa.get_loopback_device_info_generator():
            lines.append(f"  [{lb['index']}] {lb['name']}  ({lb['maxInputChannels']} ch)")
        pa.terminate()
    except Exception as e:  # noqa: BLE001
        lines.append(f"  (could not enumerate loopback devices: {e})")
    return "\n".join(lines)
