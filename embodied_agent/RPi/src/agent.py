from enum import Enum

from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye


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
            self.vision_cortex.extract_features(self.eye.capture(gray=True, blur=True))
            pass


def main():
    agent = Agent()
    agent.life()


if __name__ == "__main__":
    main()
