from brain_rythm import BrainRhythm
import threading
import time


class BrainState:
    def __init__(self):
        self.rhythms = {
            "gamma": BrainRhythm(0),
            "beta": BrainRhythm(0),
            "alpha": BrainRhythm(0),
            "theta": BrainRhythm(0),
            "delta": BrainRhythm(0),
        }
        self.state = {
            "gamma": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
            "theta": 0.0,
            "delta": 0.0,
        }

    def _update_loop(self):
        while True:
            for name, rhythm in self.rhythms.items():
                rhythm.update()
                self.state[name] = rhythm.value
            time.sleep(0.001)

    def start(self):
        """Start global rhythm update thread"""
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
