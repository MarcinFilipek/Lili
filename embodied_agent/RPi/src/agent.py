from enum import Enum
import cv2

from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from Eye import Eye


class State(Enum):
    AWAKE = "awake"
    SLEEP = "sleep"


class Agent:
    def __init__(self):
        self.eye = Eye()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()

        self.state: State = State.AWAKE

    def life(self):
        while True:
            frame = self.eye.capture(gray=True, blur=True)
            self.vision_cortex.extract_features(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.eye.close()


def main():
    agent = Agent()
    agent.life()


if __name__ == "__main__":
    main()
