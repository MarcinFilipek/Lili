import random
from collections import deque

STM_CAPACITY = 30
LTM_CAPACITY = 10000  # max sekwencji per cecha (reservoir sampling)


class Hippocampus:
    def __init__(self):
        self.sleeping = True
        self.stm = deque(maxlen=STM_CAPACITY)
        # LTM: dict {feature_name: list of sequences}
        # Sekwencje dodawane przez reservoir sampling — stare losowo zastępowane
        self.ltm = {}
        self._ltm_seen = {}  # licznik widzianych sekwencji per cecha (do reservoir)

    def wake_up(self):
        self.sleeping = False
        total = sum(len(v) for v in self.ltm.values())
        print(f"\n☀️  Waking up — STM cleared, LTM: {total} sekwencji")
        self.stm.clear()

    def go_sleep(self):
        self.sleeping = True
        print("\n🌙  Going to sleep — Consolidation phase")

    def record(self, features, action):
        if not self.sleeping:
            self.stm.append({"features": features, "action": action})

    def extract_sequences(self) -> dict:
        """Buduje trojki (f_t, action, f_t+1) z bufora STM."""
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

    def consolidate_to_ltm(self, sequences_by_feature: dict):
        """
        Dodaje sekwencje ze STM do LTM przez reservoir sampling.
        Gdy LTM jest pełne, każda nowa sekwencja zastępuje losową starą
        z prawdopodobieństwem LTM_CAPACITY / n_seen — gwarantuje
        równomierne próbkowanie całej historii.
        """
        for feature_name, sequences in sequences_by_feature.items():
            if feature_name not in self.ltm:
                self.ltm[feature_name] = []
                self._ltm_seen[feature_name] = 0

            for seq in sequences:
                n = self._ltm_seen[feature_name]
                self._ltm_seen[feature_name] += 1

                if len(self.ltm[feature_name]) < LTM_CAPACITY:
                    self.ltm[feature_name].append(seq)
                else:
                    # Reservoir sampling: zastąp losowy element
                    idx = random.randint(0, n)
                    if idx < LTM_CAPACITY:
                        self.ltm[feature_name][idx] = seq

        total = sum(len(v) for v in self.ltm.values())
        seen  = sum(v for v in self._ltm_seen.values())
        print(f"    LTM: {total} sekwencji w pamięci / {seen} widzianych łącznie"
              f" ({', '.join(f'{k}:{len(v)}' for k, v in self.ltm.items())})")

    def sample_ltm(self, n: int) -> dict:
        """
        Losuje n sekwencji z LTM per cecha (lub mniej jeśli LTM mniejsze).
        Zwraca dict {feature_name: [seq, ...]} gotowy do consolidate().
        """
        result = {}
        for name, seqs in self.ltm.items():
            k = min(n, len(seqs))
            result[name] = random.sample(seqs, k) if k > 0 else []
        return result

    def ltm_size(self) -> dict:
        return {k: len(v) for k, v in self.ltm.items()}