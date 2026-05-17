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
            pan  = self.uart.read_servo_position(0)  # już w stopniach [-90, 90]
            tilt = self.uart.read_servo_position(1)
            self.actual_position = np.array([pan, tilt])
            return self.actual_position
        except Exception as e:
            print(f"❌ Error reading position: {e}")
            return self.actual_position

    def reset(self):
        """Move to physical center (0°, 0°) regardless of current state."""
        center = np.array([0.0, 0.0])
        self._send_position(center)
        self.estimated_position = center
        self.actual_position = center

    def _clamp(self, position):
        return np.array(
            [
                np.clip(position[0], PAN_MIN, PAN_MAX),
                np.clip(position[1], TILT_MIN, TILT_MAX),
            ]
        )

    def _degrees_to_byte(self, degrees):
        # [-90, 90] → [0, 180] — STM32 oczekuje stopni bezpośrednio
        return int(np.clip(degrees + 90.0, 0, 180))

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
