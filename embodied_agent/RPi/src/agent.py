import cv2
import numpy as np

from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus import Hippocampus
from predictive_system import PredictiveSystem
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


class Agent:
    def __init__(self):
        self.eye = Eye()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = Hippocampus(rhythms=RHYTHMS, brain_state=BRAIN_STATE)
        self.predictive_system = PredictiveSystem()

        self.frustration = 0.0
        self.frustration_threshold = 5.0
        self.frustration_decay = 0.1
        self.error_threshold = 0.1

        self.previous_features = None
        selt._theta_processed = True

        start_global_rhythms()

    def life(self):
        while True:
            frame = self.eye.capture(gray=True, blur=True)
            features = self.vision_cortex.extract_features(frame)

            if self.previous_features is None:
                self.previous_features = features
                self.hippocampus.life(features, self.motor_cortex.action)
                continue

            predictions = self.predictive_system.predict(
                self.previous_features, self.motor_cortex.action
            )

            error, error_breakdown = self.predictive_system.compute_error(
                predictions, features
            )

            if error > self.error_threshold:
                self.frustration += error  # rośnie proporcjonalnie do błędu
                print(f"    ⚠️ Error={error:.3f}, frustration={self.frustration:.2f}")
            else:
                self.frustration = max(0, self.frustration - self.frustration_decay)

            action = self.select_action(features)

            self.hippocampus.life(features, action)

            self.motor_cortex.execute(action)

            val = BRAIN_STATE["theta"]
            if val > 0.95 and not self._theta_processed:
                print("🟣 θ - theta processing")
                self._theta_processed = True
                seq = self.hippocampus.extract_sequences()
                if len(seq) > 0:
                    self.predictive_system.consolidate(seq, learning_rate=0.01)
            elif val < 0:
                self._theta_processed = False

            self.previous_features = features

            if self.frustration > self.frustration_threshold:
                self.hippocampus.go_sleep()
                continue

            # cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # cv2.destroyAllWindows()
        self.eye.close()

    def select_action(self, features):
        return np.array([0, 0, 0, 0, 1])


def main():
    agent = Agent()
    agent.life()


if __name__ == "__main__":
    main()
