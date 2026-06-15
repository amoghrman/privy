"""Smoke test: launch the overlay, stream fake tokens, verify capture exclusion.
Auto-quits after ~5s. Proves the Qt path without audio/LLM."""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from events import EventBus, AnswerToken, TranscriptUpdate, StatusUpdate
from copilot.overlay import OverlayWindow, position_top_right, exclude_from_capture

app = QApplication(sys.argv)
bus = EventBus(); bus.start()
win = OverlayWindow(bus); position_top_right(win); win.show()

# capture exclusion result (printed for verification)
ok = exclude_from_capture(int(win.winId()))
print(f"SetWindowDisplayAffinity success = {ok}", flush=True)

# simulate a detected question + streamed answer
bus.publish(TranscriptUpdate("other", "What's your pricing model?", is_final=True))
parts = ["Privy ", "is ", "free ", "and ", "local-first ", "— ", "no ", "cloud ", "fees."]
def feed(i=[0]):
    n = i[0]
    if n == 0:
        bus.publish(AnswerToken("", first=True))
    if n < len(parts):
        bus.publish(AnswerToken(parts[n], first=(n == 0)))
        i[0] += 1
    else:
        bus.publish(AnswerToken("", done=True))
        t.stop()
t = QTimer(); t.timeout.connect(feed); t.start(180)

QTimer.singleShot(5000, app.quit)
rc = app.exec()
bus.stop()
print(f"overlay exited cleanly rc={rc}", flush=True)
