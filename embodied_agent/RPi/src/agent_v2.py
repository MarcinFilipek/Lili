from brain_state import BrainState
from motor_cortex import MotorCortex
from vision_cortex import VisionCortex
from eye import Eye
from hippocampus_without_rhythms import HippocampusWithoutRhythms
from predictive_system import PredictiveSystem


class AgentV2:
    def __init__(self):
        self.eye = Eye()
        self.brain_state = BrainState()
        self.motor_cortex = MotorCortex()
        self.vision_cortex = VisionCortex()
        self.hippocampus = HippocampusWithoutRhythms()
        self.predictive_system = PredictiveSystem()

        self.previous_features = None

        self.gamma_rhythm = self.brain_state.rhythms["gamma"]
        self.theta_rhythm = self.brain_state.rhythms["theta"]
        self.brain_state.start()
        self.wake_up()

    def wake_up(self):
        self.hippocampus.wake_up()
        frame = self.eye.capture(gray=True, blur=True)
        features = self.vision_cortex.extract_features(frame)
        self.hippocampus.life(features, self.motor_cortex.action)
        self.previous_features = features

    def life(self):
        while True:
            proprioception = self.motor_cortex.update_proprioception()
            # print(f"Proprioception: {proprioception}")
            frame = self.eye.capture(gray=True, blur=True)
            features = self.vision_cortex.extract_features(frame)

            print(f"Features: {features}")

            predictions = self.predictive_system.predict(
                self.previous_features, self.motor_cortex.action
            )
            error, error_breakdown = self.predictive_system.compute_error(
                predictions, features
            )
            print(f"Error: {error:.3f}, Error breakdown: {error_breakdown}")

            self.hippocampus.life(features, self.motor_cortex.action)

            seq = self.hippocampus.extract_sequences()

            print(f"Stm brightness: {len(self.hippocampus.stm_buffers['brightness'])}")
            print(f"Stm corners: {len(self.hippocampus.stm_buffers['corners'])}")
            print(f"Sequences: {seq}")

            if len(seq) > 0:
                self.predictive_system.consolidate(seq, learning_rate=0.0005)

            self.previous_features = features


def main():
    agent = AgentV2()
    agent.life()


if __name__ == "__main__":
    main()
