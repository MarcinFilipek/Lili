import cv2
import numpy as np

from brain_state import BrainState
from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus import Hippocampus
from predictive_system import PredictiveSystem


class Agent:
    def __init__(self):
        self.eye = Eye()
        self.brain_state = BrainState()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = Hippocampus(rhythms=self.brain_state.rhythms)
        self.predictive_system = PredictiveSystem()

        self.frustration = 0.0
        self.frustration_threshold = 5.0
        self.frustration_decay = 0.1
        self.error_threshold = 0.1

        self.previous_features = None
        self._theta_processed = True

        self.gamma_rhythm = self.brain_state.rhythms["gamma"]
        self.theta_rhythm = self.brain_state.rhythms["theta"]
        self.brain_state.start()

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

            if error > self.error_threshold and self.gamma_rhythm.is_started():
                self.frustration += error  # rośnie proporcjonalnie do błędu
            else:
                self.frustration = max(0, self.frustration - self.frustration_decay)

            action = self.select_action(features)

            self.hippocampus.life(features, action)

            self.motor_cortex.execute(action)

            val = self.theta_rhythm.value
            if val > 0.95 and not self._theta_processed:
                print("🟣 θ - theta processing")
                self._theta_processed = True
                seq = self.hippocampus.extract_sequences()
                lr = 0.0005 if self.gamma_rhythm.is_started() else 0.01
                if len(seq) > 0:
                    self.predictive_system.consolidate(seq, learning_rate=lr)
            elif val < 0:
                self._theta_processed = False

            self.previous_features = features

            if self.frustration > self.frustration_threshold:
                print(f"    ⚠️ Error={error:.3f}, frustration={self.frustration:.2f}")
                self.hippocampus.go_sleep()
                continue

            if not self.gamma_rhythm.is_started() and error < self.error_threshold:
                self.hippocampus.wake_up()
                self.frustration = 0.0
                print(f"    ⚠️ Error={error:.3f}, frustration={self.frustration:.2f}")

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
