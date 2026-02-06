from brain_state import BrainState
from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus import Hippocampus
from predictive_system import PredictiveSystem


class AgentV2:
    def __init__(self):
        self.eye = Eye()
        self.brain_state = BrainState()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = Hippocampus(rhythms=self.brain_state.rhythms)
        self.predictive_system = PredictiveSystem()

        self.gamma_rhythm = self.brain_state.rhythms["gamma"]
        self.theta_rhythm = self.brain_state.rhythms["theta"]
        self.brain_state.start()

    def life(self):
        while True:
            proprioception = self.motor_cortex.update_proprioception()
            print(f"Proprioception: {proprioception)
            frame = self.eye.capture(gray=True, blur=True)
            features = self.vision_cortex.extract_features(frame)

            self.hippocampus.life(features, self.motor_cortex.action)


def main():
    agent = AgentV2()
    agent.life()


if __name__ == "__main__":
    main()
