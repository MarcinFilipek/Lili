from collections import deque

STM_CAPACITY = 30


class Hippocampus:
    def __init__(self):
        self.sleeping = True
        self.stm = deque(maxlen=STM_CAPACITY)
        # LTM: lista wszystkich historycznych sekwencji per cecha
        # {"brightness": [...], "corners": [...]}
        self.ltm = {}

    def wake_up(self):
        self.sleeping = False
        print("\n☀️  Waking up — STM cleared, LTM intact")
        self.stm.clear()

    def go_sleep(self):
        self.sleeping = True
        print("\n🌙  Going to sleep — Consolidation phase")

    def record(self, features, action, position):
        if not self.sleeping:
            self.stm.append({
                "features": features,
                "action":   action,
                "position": position,   # np.array([pan_deg, tilt_deg])
            })

    def extract_sequences(self):
        """
        Buduje trojki (f_t, action, f_t+1) z bufora STM.
        Zwraca dict {feature_name: [seq, ...]} gotowy do consolidate().
        """
        sequences_by_feature = {}
        if len(self.stm) < 2:
            return sequences_by_feature

        stm_list = list(self.stm)
        feature_names = stm_list[0]["features"].keys()

        for name in feature_names:
            sequences_by_feature[name] = []
            for i in range(len(stm_list) - 1):
                sequences_by_feature[name].append({
                    "feature_t":  stm_list[i]["features"][name],
                    "action":     stm_list[i]["action"],
                    "feature_t1": stm_list[i + 1]["features"][name],
                })

        return sequences_by_feature

    def extract_place_samples(self) -> list:
        """
        Buduje listę próbek dla modelu miejsca z całego STM.
        Każda próbka: {'feature': brightness(121,), 'pan': float, 'tilt': float}

        Używa każdego wpisu STM niezależnie (nie potrzebuje par t/t+1).
        """
        samples = []
        for entry in self.stm:
            if "brightness" not in entry["features"]:
                continue
            pan, tilt = float(entry["position"][0]), float(entry["position"][1])
            samples.append({
                "feature": entry["features"]["brightness"],
                "pan":     pan,
                "tilt":    tilt,
            })
        return samples

    def consolidate_to_ltm(self, sequences_by_feature: dict):
        """
        Przenosi sekwencje ze STM do LTM po każdym śnie.
        LTM rośnie kumulatywnie — historia wszystkich epizodów.

        Args:
            sequences_by_feature: dict z extract_sequences()
        """
        for feature_name, sequences in sequences_by_feature.items():
            if feature_name not in self.ltm:
                self.ltm[feature_name] = []
            self.ltm[feature_name].extend(sequences)

        total = sum(len(v) for v in self.ltm.values())
        print(f"    LTM: {total} sekwencji łącznie ({', '.join(f'{k}:{len(v)}' for k, v in self.ltm.items())})")

    def ltm_size(self) -> dict:
        """Zwraca liczbę sekwencji w LTM per cecha."""
        return {k: len(v) for k, v in self.ltm.items()}