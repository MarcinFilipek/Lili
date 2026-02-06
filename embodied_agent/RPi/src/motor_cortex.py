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

    def send_frame(self, cmd_type, channel, data):
        start_byte = self.spi.START_BYTE
        checksum = self._calculate_checksum(start_byte, cmd_type, channel, data)
        frame = [start_byte, cmd_type, channel, data, checksum]
        response = self.spi.send(frame)
        time.sleep(0.05)
        return response

    def _calculate_checksum(self, start, cmd, channel, data):
        return start ^ cmd ^ channel ^ data

    def _read_servo_position(self, channel):
        """
        Request and read servo position from STM32.

        Args:
            channel: 0 (head) or 1 (eye)

        Returns:
            position: float in [-1, 1]
        """
        # Send request frame
        request = [
            self.spi.START_BYTE,
            self.spi.CMD_GET_POSITION,
            channel,
            0x00,
            self._compute_checksum(
                self.spi.START_BYTE, self.spi.CMD_GET_POSITION, channel, 0x00
            ),
        ]

        self.spi.send(request)

        # Read response frame
        response = self.spi.receive(5)

        # Validate response
        if response[0] != self.spi.START_BYTE:
            raise Exception(f"Invalid start byte: {response[0]:#x}")

        if response[2] != channel:
            raise Exception(f"Channel mismatch: expected {channel}, got {response[2]}")

        # Verify checksum
        expected_checksum = self._compute_checksum(
            response[0], response[1], response[2], response[3]
        )
        if response[4] != expected_checksum:
            raise Exception(f"Checksum mismatch")

        # Extract position data
        data_byte = response[3]
        position = self._map_from_byte(data_byte)

        return position

    def update_proprioception(self):
        """
        Read actual servo positions from STM32.
        Call this every frame to get current state.

        Returns:
            np.array([head_pan, eye_tilt]) - actual positions in [-1, 1]
        """
        try:
            # Read head position
            head_pos = self._read_servo_position(self.CHANNEL_HEAD)

            # Read eye position
            eye_pos = self._read_servo_position(self.CHANNEL_EYE)

            self.actual_position = np.array([head_pos, eye_pos])

        except Exception as e:
            print(f"⚠️  Proprioception read failed: {e}")
            # Fallback: assume we're close to target
            self.actual_position = self.target_position.copy()

        return self.actual_position
