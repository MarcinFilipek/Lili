from enum import Enum
import cv2

from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus import Hippocampus
from predictive_system import PredictiveSystem


class State(Enum):
    AWAKE = "awake"
    SLEEP = "sleep"


class Agent:
    def __init__(self):
        self.eye = Eye()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = Hippocampus()
        self.predictive_system = PredictiveSystem()

        self.state: State = State.AWAKE

    def life(self):
        while True:
            frame = self.eye.capture(gray=True, blur=True)
            features = self.vision_cortex.extract_features(frame)

            predictions = self.predictive_system.predict(
                features, self.motor_cortex.action
            )

            error, error_breakdown = self.predictive_system.compute_error(
                predictions, features
            )

            action = self.select_action(features)

            self.hippocampus.life(features, action)

            self.motor_cortex.execute(action)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.eye.close()

    def select_action(self, features):
        return [0, 0, 0, 0, 1]


def main():
    agent = Agent()
    agent.life()


if __name__ == "__main__":
    main()
