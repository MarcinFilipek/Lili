from brain_rythm import BrainRhythm

RHYTHMS = {
    "theta": BrainRhythm(4),  # Hz
    "delta": BrainRhythm(1),  # Hz
}

BRAIN_STATE = {"gamma": 0.0, "theta": 0.0, "delta": 0.0}


def update_brain_state_loop():
    """
    Wątek aktualizujący rytmy z akumulacją fazy.
    """
    while True:
        for name, rhythm in RHYTHMS.items():
            rhythm.update()
            BRAIN_STATE[name] = rhythm.value

        # Wysoka częstotliwość odświeżania dla płynności
        time.sleep(0.001)


def start_global_rhythms():
    print("Startuje globalne rytmy mózgowe (Phase Accumulator)...")
    t = threading.Thread(target=update_brain_state_loop, daemon=True)
    t.start()


class Hippocampus:
    def __init__(self):
        self._theta_processed = True
        self._delta_processed = True

        start_global_rhythms()

    def life(self):
        self._theta_life()
        self._delta_life()

    def _theta_life(self):
        val = BRAIN_STATE["theta"]
        if val > 0.95 and not self._theta_processed:
            self._theta_processed = True
            print("Processing theta frame")
        elif val < 0:
            self._theta_processed = False

    def _delta_life(self):
        val = BRAIN_STATE["delta"]
        if val > 0.95 and not self._delta_processed:
            self._delta_processed = True
            print("Processing delta frame")
        elif val < 0:
            self._delta_processed = False
