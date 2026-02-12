import numpy as np


class HippocampusWithoutRhythms:
    def __init__(self):
        self.stm_buffers = {"brightness": [], "corners": []}
        self.max_buffer_size = 20

        self.ltm_sequences = []
        self.sleeping = False

    def life(self, features, action):
        self._gamma_life(features, action)
        self._beta_life(features, action)
        self._alpha_life(features, action)
        self._theta_life(features, action)
        self._delta_life(features, action)

    def go_sleep(self):
        self.sleeping = True
        print("\n💤 Entering sleep...")

    def wake_up(self):
        self.sleeping = False
        print("\n☀️ Waking up...")

    def _gamma_life(self, features, action):
        print("🔴 γ - gamma processing")
        for feature_name, feature_vec in features.items():
            if feature_name in self.stm_buffers:
                buffer = self.stm_buffers[feature_name]
                buffer.append(
                    {
                        "type": "feature",
                        "data": feature_vec.copy(),
                    }
                )

                buffer.append({"type": "action", "data": action.copy()})

                # Trim buffer if too large
                if len(buffer) > self.max_buffer_size:
                    buffer.pop(0)

    def _beta_life(self, features, action):
        print("🟠 β - beta processing")

    def _alpha_life(self, features, action):
        print("🟡 α - alpha processing")

    def _theta_life(self, features, action):
        print("🟢 θ - theta processing")

    def _delta_life(self, features, action):
        print("🟣 δ - delta processing")

    def extract_sequences(self, verbose=False):
        """Extract sequences with optional debug output"""
        sequences_by_feature = {}

        for feature_name, buffer in self.stm_buffers.items():
            sequences = []

            if verbose:
                print(f"\n=== Extracting sequences for '{feature_name}' ===")
                print(f"Buffer size: {len(buffer)}")

            i = 0
            while i < len(buffer) - 2:

                # Check pattern
                is_feature = buffer[i]["type"] == "feature"
                is_action = buffer[i + 1]["type"] == "action"
                is_next_feature = buffer[i + 2]["type"] == "feature"

                if is_feature and is_action and is_next_feature:
                    # Found sequence!
                    sequences.append(
                        {
                            "feature_t": buffer[i]["data"],
                            "action": buffer[i + 1]["data"],
                            "feature_t1": buffer[i + 2]["data"],
                        }
                    )

                    if verbose:
                        print(f"  ✓ Sequence found at indices [{i}, {i+1}, {i+2}]")
                        print(f"    Action: {np.argmax(buffer[i+1]['data'])}")

                    i += 3
                else:
                    if verbose:
                        print(
                            f"  ✗ No sequence at index {i} "
                            f"(pattern: {buffer[i]['type']}, "
                            f"{buffer[i+1]['type'] if i+1 < len(buffer) else 'N/A'}, "
                            f"{buffer[i+2]['type'] if i+2 < len(buffer) else 'N/A'})"
                        )
                    i += 1

            sequences_by_feature[feature_name] = sequences

            if verbose:
                print(f"Total sequences extracted: {len(sequences)}\n")

        return sequences_by_feature


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
