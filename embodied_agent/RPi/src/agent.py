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

        self.sleep_threshold = 0.1
        self.previous_features = None

    def life(self):
        while True:
            frame = self.eye.capture(gray=True, blur=True)
            features = self.vision_cortex.extract_features(frame)
            print("New features")

            if self.previous_features is not None:
                predictions = self.predictive_system.predict(
                    self.previous_features, self.motor_cortex.action
                )

                error, error_breakdown = self.predictive_system.compute_error(
                    predictions, features
                )
                print(f"Error={error:.4f} | {error_breakdown}")

            action = self.select_action(features)

            self.hippocampus.life(features, action)

            self.motor_cortex.execute(action)

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

            if error > self.sleep_threshold:
                self.hippocampus.go_sleep()
                return

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
