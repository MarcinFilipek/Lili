import spidev
import time


class SPIConnection:
    def __init__(self):
        self.spi = spidev.SpiDev()

        self.START_BYTE = 0xAA
        self.CMD_SET_POSITION = 0x01
        self.CMD_GET_POSITION = 0x02

        try:
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 500000
        except Exception as e:
            print(f"Error initializing SPI: {e}")

    def _calculate_checksum(self, start, cmd, channel, data):
        return start ^ cmd ^ channel ^ data

    def _map_from_byte(self, byte_value):
        """
        Convert byte (0-255) to position (-1 to 1).

        Args:
            byte_value: 0-255

        Returns:
            position: -1.0 to 1.0
        """
        return (byte_value / 127.5) - 1.0

    def send(self, data):
        try:
            response = self.spi.xfer2(data)
            return response
        except Exception as e:
            print(f"Error sending data: {e}")

    def _receive(self, num_bytes):
        """
        Receive data via SPI.

        Args:
            num_bytes: number of bytes to read

        Returns:
            list of bytes
        """
        try:
            # Send dummy bytes to clock in response
            dummy = [0x00] * num_bytes
            response = self.spi.xfer2(dummy)
            return response
        except Exception as e:
            print(f"Error receiving data: {e}")
            return [0x00] * num_bytes

    # def send_frame(self, cmd_type, channel, data):
    #     start_byte = self.START_BYTE
    #     checksum = self._calculate_checksum(start_byte, cmd_type, channel, data)
    #     frame = [start_byte, cmd_type, channel, data, checksum]
    #     response = self.send(frame)
    #     time.sleep(0.05)
    #     return response

    def read_servo_position(self, channel):
        """
        Request and read servo position from STM32.

        Args:
            channel: 0 (head) or 1 (eye)

        Returns:
            position: float in [-1, 1]
        """
        # Prepare request frame
        request = [
            self.START_BYTE,
            self.CMD_GET_POSITION,
            channel,
            0x00,
            self._calculate_checksum(
                self.START_BYTE, self.CMD_GET_POSITION, channel, 0x00
            ),
        ]

        # Phase 1: Send request (STM32 receives command, sends old buffer)
        try:
            self.spi.xfer2(request)
        except Exception as e:
            print(f"Error sending request: {e}")
            raise

        # Wait for STM32 to process
        time.sleep(0.02)  # 1ms

        # Phase 2: Send dummy bytes (STM32 loads response into TX buffer)
        dummy = [0x00] * 5
        try:
            self.spi.xfer2(dummy)
        except Exception as e:
            print(f"Error in phase 2: {e}")
            raise

        # Small delay
        time.sleep(0.02)  # 1ms

        # Phase 3: Send dummy bytes AGAIN (NOW we get the real response!)
        try:
            response = self.spi.xfer2(dummy)
        except Exception as e:
            print(f"Error receiving response: {e}")
            raise

        print(f"Response: {[hex(b) for b in response]}")

        # Validate response
        if response[0] != self.START_BYTE:
            raise Exception(
                f"Invalid start byte: {response[0]:#x} (expected {self.START_BYTE:#x})"
            )

        if response[2] != channel:
            raise Exception(f"Channel mismatch: expected {channel}, got {response[2]}")

        # Verify checksum
        expected_checksum = self._calculate_checksum(
            response[0], response[1], response[2], response[3]
        )
        if response[4] != expected_checksum:
            raise Exception(
                f"Checksum mismatch: got {response[4]:#x}, expected {expected_checksum:#x}"
            )

        # Extract position data
        data_byte = response[3]
        position = self._map_from_byte(data_byte)

        return position
