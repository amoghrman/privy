"""Probe several mic devices at once to find which actually has signal.
Usage: python mic_probe.py 1 5 9 17 18    (defaults to a sensible set)
"""
import sys
import numpy as np
import sounddevice as sd

cands = [int(a) for a in sys.argv[1:]] or [1, 5, 9, 16, 17, 18]
peaks = {d: 0.0 for d in cands}
streams = []

def mk(d):
    def cb(indata, frames, t, status):
        peaks[d] = max(peaks[d], float(np.sqrt(np.mean(indata**2))))
    return cb

for d in cands:
    try:
        info = sd.query_devices(d, "input")
        sr = int(info["default_samplerate"])
        s = sd.InputStream(device=d, samplerate=sr, channels=1, dtype="float32", callback=mk(d))
        s.start()
        streams.append(s)
        print(f"opened [{d}] {info['name']} @ {sr}")
    except Exception as e:
        print(f"  [{d}] FAILED: {e}")

print("speak for 6s…")
sd.sleep(6000)
for s in streams:
    s.stop(); s.close()
print("--- PEAK RMS per device ---")
for d in cands:
    mark = "  <-- LIVE" if peaks[d] > 0.02 else ""
    print(f"  [{d}] {peaks[d]:.4f}{mark}")
