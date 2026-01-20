import cv2
import numpy as np

from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus import Hippocampus
from predictive_system import PredictiveSystem


class Agent:
    def __init__(self):
        self.eye = Eye()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = Hippocampus()
        self.predictive_system = PredictiveSystem(self.hippocampus.ltm_sequences)

        self.frustration = 0.0  # akumuluje się przy błędach
        self.frustration_threshold = 5.0  # próg zmęczenia
        self.frustration_decay = 0.1  # maleje gdy error niski
        self.error_threshold = 0.1

        self.previous_features = None

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
                print(f"  ⚠️ Error={error:.3f}, frustration={self.frustration:.2f}")
            else:
                self.frustration = max(0, self.frustration - self.frustration_decay)

            action = self.select_action(features)

            self.hippocampus.life(features, action)

            self.motor_cortex.execute(action)

            print(f"STM: {self.hippocampus.stm_buffers}")
            print(f"Sequences: {self.hippocampus.extract_sequences(verbose=True)}")

            """
            if delta > 0.9:
                sequences = self.hippocampus.get_recent_sequences(n=5)
                total_seqs = sum(len(s) for s in sequences.values())
                
                if total_seqs > 0:
                    # Gentle online learning
                    self.predictive_system.consolidate(
                    sequences, 
                    learning_rate=self.delta_lr_awake
                )
            """

            self.previous_features = features

            if error > self.frustration_threshold:
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
