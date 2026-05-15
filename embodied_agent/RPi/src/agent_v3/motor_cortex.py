import numpy as np
from uart_connection import UARTConnection


PAN_MIN = -90.0
PAN_MAX = 90.0
TILT_MIN = -90.0
TILT_MAX = 90.0


class MotorCortex:
    def __init__(self):
        self.uart = UARTConnection()

        self.last_delta = np.array([0.0, 0.0])

        self.actual_position = np.array([0.0, 0.0])

        self.estimated_position = np.array([0.0, 0.0])

    def execute(self, delta):
        self.last_delta = delta.copy()

        target = self.actual_position + delta
        target = self._clamp(target)

        self._send_position(target)
        self.estimated_position = target

    def read_position(self):
        """Reads actual position from STM32"""
        try:
            pan = self.uart.read_servo_position(0)
            tilt = self.uart.read_servo_position(1)
            # pan and tilt are -1.0 to 1.0. Convert to degrees
            pan_deg = pan * 90.0
            tilt_deg = tilt * 90.0
            self.actual_position = np.array([pan_deg, tilt_deg])
            return self.actual_position
        except Exception as e:
            print(f"❌ Error reading position: {e}")
            return self.actual_position

    def reset(self):
        """Move to home position (0, 0)."""
        self.execute(np.array([0.0, 0.0]))

    def _clamp(self, position):
        return np.array(
            [
                np.clip(position[0], PAN_MIN, PAN_MAX),
                np.clip(position[1], TILT_MIN, TILT_MAX),
            ]
        )

    def _degrees_to_byte(self, degrees):
        """Map [-90, 90] → [0, 255]"""
        normalized = (degrees + 90.0) / 180.0  # [0, 1]
        return int(np.clip(normalized * 255, 0, 255))

    def _send_position(self, position_degrees):
        """
        Send absolute target position to STM32.
        (Delta is computed here; STM32 receives absolute target.)
        """
        pan_byte = self._degrees_to_byte(position_degrees[0])
        tilt_byte = self._degrees_to_byte(position_degrees[1])
        print(f"Sending position: {position_degrees} -> {pan_byte}, {tilt_byte}")

        try:
            self.uart.set_servo_position(0, pan_byte)
            self.uart.set_servo_position(1, tilt_byte)
        except Exception as e:
            print(f"❌ Error setting position: {e}")
