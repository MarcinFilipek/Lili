import time
import numpy as np
from eye import Eye
from motor_cortex import MotorCortex
from hippocampus import Hippocampus
from vision_cortex import VisionCortex
from predictive_system import PredictiveSystem


class AgentV3:
    def __init__(self):
        self.eye = Eye()
        self.motor = MotorCortex()
        self.hippocampus = Hippocampus()
        self.vision_cortex = VisionCortex()
        self.predictive_system = PredictiveSystem()

        self.frustration = 0.0
        self.error_threshold = 0.05
        self.frustration_gain = 1.0
        self.frustration_decay = 0.05
        self.frustration_threshold = 10.0
        self.consolidation_lr = 0.01

    def _select_action(self):
        sigma = 3.0  # degrees
        return np.random.normal(0.0, sigma, size=2)

    def run(self):
        print("🤖 Agent starting...")

        self.motor.reset()
        self.hippocampus.wake_up()

        # Wait for mechanical parts to settle
        time.sleep(5.0)

        # Initial perception
        prev_features = self._capture_features()
        prev_action = np.array([0.0, 0.0])

        while True:
            # 1. Proprioception
            actual_pos = self.motor.read_position()

            # 2. Perception
            features = self._capture_features()

            # 3. Prediction & Error
            predictions = self.predictive_system.predict(prev_features, prev_action)
            avg_error, error_breakdown = self.predictive_system.compute_error(
                predictions, features
            )

            print(
                f"Pos:[{actual_pos[0]:.1f}, {actual_pos[1]:.1f}] "
                f"Error: {avg_error:.4f} | Frust: {self.frustration:.2f}"
            )

            # 4. Frustration / Sleep
            if avg_error > self.error_threshold:
                self.frustration += avg_error * self.frustration_gain
            else:
                self.frustration = max(0.0, self.frustration - self.frustration_decay)

            if self.frustration > self.frustration_threshold:
                self.hippocampus.go_sleep()
                sequences = self.hippocampus.extract_sequences()

                print(f"    Consolidating {len(sequences.get('brightness', []))} sequences...")
                self.predictive_system.consolidate(sequences, lr=self.consolidation_lr)

                self.hippocampus.wake_up()
                self.frustration = 0.0
                
                # Small sleep to let mechanical parts settle if any
                time.sleep(0.5)
                
                # Capture fresh state after sleep
                prev_features = self._capture_features()
                prev_action = np.array([0.0, 0.0])
                continue

            # 5. Record to STM
            self.hippocampus.record(prev_features, prev_action)

            # 6. Action selection
            action = self._select_action()

            # 7. Action execution
            self.motor.execute(action)

            # Update history
            prev_features = features
            prev_action = action

            time.sleep(0.1)

    def _capture_features(self):
        frame = self.eye.capture(gray=True, blur=True)
        features = self.vision_cortex.extract_features(frame)
        return features


def main():
    agent = AgentV3()
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        agent.eye.close()


if __name__ == "__main__":
    main()
