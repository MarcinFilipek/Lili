from brain_rythm import BrainRhythm
import threading
import time

RHYTHMS = {
    "theta": BrainRhythm(0),  # Hz
    "delta": BrainRhythm(0),  # Hz
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
        self.stm_buffers = {"brightness": []}
        self.max_buffer_size = 20

        self._theta_processed = True
        self._delta_processed = True
        RHYTHMS["theta"].start(4)
        RHYTHMS["delta"].start(1)

        start_global_rhythms()

    def life(self, features, action):
        self._theta_life(features, action)
        self._delta_life()

    def _theta_life(self, features, action):
        theta_val = BRAIN_STATE["theta"]
        if theta_val > 0.95 and not self._theta_processed:
            self._theta_processed = True
            print("Processing theta frame")

            for feature_name, feature_vec in features.items():
                if feature_name in self.stm_buffers:
                    buffer = self.stm_buffers[feature_name]
                    buffer.append(
                        {"type": "feature", "data": feature_vec.copy(), "theta": theta}
                    )

        elif theta_val < 0:
            self._theta_processed = False

        return theta_val

    def _delta_life(self):
        val = BRAIN_STATE["delta"]
        if val > 0.95 and not self._delta_processed:
            self._delta_processed = True
            print("Processing delta frame")
        elif val < 0:
            self._delta_processed = False

    # def extract_sequences(self, verbose=False):
    #     """Extract sequences with optional debug output"""
    #     sequences_by_feature = {}

    #     for feature_name, buffer in self.stm_buffers.items():
    #         sequences = []

    #         if verbose:
    #             print(f"\n=== Extracting sequences for '{feature_name}' ===")
    #             print(f"Buffer size: {len(buffer)}")

    #         i = 0
    #         while i < len(buffer) - 2:

    #             # Check pattern
    #             is_feature = buffer[i]["type"] == "feature"
    #             is_action = buffer[i + 1]["type"] == "action"
    #             is_next_feature = buffer[i + 2]["type"] == "feature"

    #             if is_feature and is_action and is_next_feature:
    #                 # Found sequence!
    #                 sequences.append(
    #                     {
    #                         "feature_t": buffer[i]["data"],
    #                         "action": buffer[i + 1]["data"],
    #                         "feature_t1": buffer[i + 2]["data"],
    #                     }
    #                 )

    #                 if verbose:
    #                     print(f"  ✓ Sequence found at indices [{i}, {i+1}, {i+2}]")
    #                     print(f"    Action: {np.argmax(buffer[i+1]['data'])}")

    #                 i += 3
    #             else:
    #                 if verbose:
    #                     print(
    #                         f"  ✗ No sequence at index {i} "
    #                         f"(pattern: {buffer[i]['type']}, "
    #                         f"{buffer[i+1]['type'] if i+1 < len(buffer) else 'N/A'}, "
    #                         f"{buffer[i+2]['type'] if i+2 < len(buffer) else 'N/A'})"
    #                     )
    #                 i += 1

    #         sequences_by_feature[feature_name] = sequences

    #         if verbose:
    #             print(f"Total sequences extracted: {len(sequences)}\n")

    #     return sequences_by_feature
