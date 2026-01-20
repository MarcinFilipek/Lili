from brain_rythm import BrainRhythm
import threading
import time

RHYTHMS = {
    "gamma": BrainRhythm(0),  # Hz
    "beta": BrainRhythm(0),  # Hz
    "alpha": BrainRhythm(0),  # Hz
    "theta": BrainRhythm(0),  # Hz
    "delta": BrainRhythm(0),  # Hz
}

BRAIN_STATE = {
    "gamma": 0.0,
    "beta": 0.0,
    "alpha": 0.0,
    "theta": 0.0,
    "delta": 0.0,
}


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

        self.ltm_sequences = []

        self._gamma_processed = True
        self._beta_processed = True
        self._alpha_processed = True
        self._theta_processed = True
        self._delta_processed = True

        RHYTHMS["gamma"].start(40)
        RHYTHMS["beta"].start(13)
        RHYTHMS["alpha"].start(8)
        RHYTHMS["theta"].start(4)
        RHYTHMS["delta"].start(1)

        start_global_rhythms()

    def life(self, features, action):
        self._gamma_life()
        self._beta_life()
        self._alpha_life()
        self._theta_life(features, action)
        self._delta_life()

    def _gamma_life(self, features, action):
        gamma_val = BRAIN_STATE["gamma"]
        if gamma_val > 0.95 and not self._gamma_processed:
            self._gamma_processed = True
            print("🔴 γ - gamma processing")

            for feature_name, feature_vec in features.items():
                if feature_name in self.stm_buffers:
                    buffer = self.stm_buffers[feature_name]
                    buffer.append(
                        {
                            "type": "feature",
                            "data": feature_vec.copy(),
                            "gamma": gamma_val,
                        }
                    )

                    buffer.append(
                        {"type": "action", "data": action.copy(), "gamma": gamma_val}
                    )

                    # Trim buffer if too large
                    if len(buffer) > self.max_buffer_size:
                        buffer.pop(0)

        elif gamma_val < 0:
            self._gamma_processed = False

    def _beta_life(self, features, action):
        beta_val = BRAIN_STATE["beta"]
        if beta_val > 0.95 and not self._beta_processed:
            self._beta_processed = True
            print("🟠 β - beta processing")
        elif beta_val < 0:
            self._beta_processed = False

    def _alpha_life(self, features, action):
        alpha_val = BRAIN_STATE["alpha"]
        if alpha_val > 0.95 and not self._alpha_processed:
            self._alpha_processed = True
            print("🟡 α - alpha processing")
        elif alpha_val < 0:
            self._alpha_processed = False

    def _theta_life(self, features, action):
        theta_val = BRAIN_STATE["theta"]
        if theta_val > 0.95 and not self._theta_processed:
            self._theta_processed = True
            print("🟢 θ - theta processing")

            # for feature_name, feature_vec in features.items():
            #     if feature_name in self.stm_buffers:
            #         buffer = self.stm_buffers[feature_name]
            #         buffer.append(
            #             {
            #                 "type": "feature",
            #                 "data": feature_vec.copy(),
            #                 "theta": theta_val,
            #             }
            #         )

            #         buffer.append(
            #             {"type": "action", "data": action.copy(), "theta": theta_val}
            #         )

            #         # Trim buffer if too large
            #         if len(buffer) > self.max_buffer_size:
            #             buffer.pop(0)

        elif theta_val < 0:
            self._theta_processed = False

        return theta_val

    def _delta_life(self, features, action):
        val = BRAIN_STATE["delta"]
        if val > 0.95 and not self._delta_processed:
            self._delta_processed = True
            print("🟣 δ - delta processing")
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


# DOC
"""
🔵 Delta (δ)
Zakres: ~0,5–4 Hz
Najczęściej odpowiadają za:
- głęboki sen (NREM, faza snu wolnofalowego)
- regenerację organizmu
- procesy naprawcze mózgu i ciała
📌 U dorosłych na jawie – zwykle patologiczne lub bardzo rzadkie

🟢 Theta (θ)
Zakres: ~4–8 Hz
Najczęściej odpowiadają za:
- sen płytki, stan przed zaśnięciem i tuż po przebudzeniu
- głęboką relaksację
- kreatywność, wyobraźnię
- przetwarzanie emocji i pamięć (szczególnie hipokamp)

🟡 Alpha (α)
Zakres: ~8–13 Hz
Najczęściej odpowiadają za:
- spokojną czujność
- relaks przy zachowanej świadomości
- „reset” uwagi
-  stan flow o niskim pobudzeniu
📌 Silne przy zamkniętych oczach

🟠 Beta (β)
Zakres: ~13–30 Hz
Najczęściej odpowiadają za:
- aktywne myślenie
- koncentrację i analizę
- rozwiązywanie problemów
- napięcie i stres (wysokie beta)

🔴 Gamma (γ)
Zakres: ~30–100+ Hz
Najczęściej odpowiadają za:
- integrację informacji z różnych obszarów mózgu
- wysoki poziom koncentracji
- uczenie się i zapamiętywanie
- świadomość i przetwarzanie złożonych bodźców
"""
