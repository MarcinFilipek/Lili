from spi_connection import SPIConnection
import numpy as np


class MotorCortex:
    def __init__(self):
        self.spi = SPIConnection()  # To STM32
        self.action = np.array([0, 0, 0, 0, 1])

        self.actual_position = np.array([0.0, 0.0])
        self.target_position = np.array([0.0, 0.0])

        self.CHANNEL_HEAD = 0
        self.CHANNEL_EYE = 1

    def execute(self, action):
        self.action = action

    def update_proprioception(self):
        """
        Read actual servo positions from STM32.
        Call this every frame to get current state.

        Returns:
            np.array([head_pan, eye_tilt]) - actual positions in [-1, 1]
        """
        try:
            # Read head position
            head_pos = self.spi.read_servo_position(self.CHANNEL_HEAD)

            # Read eye position
            eye_pos = self.spi.read_servo_position(self.CHANNEL_EYE)

            self.actual_position = np.array([head_pos, eye_pos])

        except Exception as e:
            print(f"⚠️  Proprioception read failed: {e}")
            # Fallback: assume we're close to target
            self.actual_position = self.target_position.copy()

        return self.actual_position
