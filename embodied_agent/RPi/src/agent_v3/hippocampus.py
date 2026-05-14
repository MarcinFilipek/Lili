from collections import deque

STM_CAPACITY = 30


class Hippocampus:
    def __init__(self):
        self.sleeping = True
        self.stm = deque(maxlen=STM_CAPACITY)
        self.ltm = []

    def wake_up(self):
        self.sleeping = False
        print("\n☀️  Waking up — STM cleared, LTM intact")
        self.stm.clear()

    def go_sleep(self):
        self.sleeping = True
        print("\n🌙  Going to sleep — Consolidation phase")

    def record(self, features, action):
        if not self.sleeping:
            self.stm.append({"features": features, "action": action})

    def extract_sequences(self):
        sequences_by_feature = {}
        if len(self.stm) < 2:
            return sequences_by_feature

        stm_list = list(self.stm)
        feature_names = stm_list[0]["features"].keys()

        for name in feature_names:
            sequences_by_feature[name] = []
            for i in range(len(stm_list) - 1):
                sequences_by_feature[name].append({
                    "feature_t": stm_list[i]["features"][name],
                    "action": stm_list[i]["action"],
                    "feature_t1": stm_list[i+1]["features"][name]
                })

        return sequences_by_feature
