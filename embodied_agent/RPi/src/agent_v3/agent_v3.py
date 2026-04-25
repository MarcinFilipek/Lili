from eye import Eye
from motor_cortex import MotorCortex
from hippocampus import Hippocampus
from vision_cortex import VisionCortex


class AgentV3:
    def __init__(self):
        self.eye = Eye()
        self.motor = MotorCortex()
        self.hippocampus = Hippocampus()
        self.vision_cortex = VisionCortex()

    def run(self):
        print("🤖 Agent starting...")

        self.motor.reset()
        self.hippocampus.wake_up()
        prev_features = self._capture_features()

        while True:
            features = self._capture_features()

    def _capture_features(self):
        frame = self.eye.capture(gray=True, blur=True)
        features = self.vision_cortex.extract_features(frame)
        return features["brightness"]


def main():
    agent = AgentV3()
    agent.run()


if __name__ == "__main__":
    main()
