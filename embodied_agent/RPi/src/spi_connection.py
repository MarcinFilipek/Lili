import spidev


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

    def send(self, data):
        try:
            self.spi.xfer2(data)
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
        # Send request frame
        request = [
            self.START_BYTE,
            self.CMD_GET_POSITION,
            channel,
            0x00,
            self._calculate_checksum(
                self.START_BYTE, self.CMD_GET_POSITION, channel, 0x00
            ),
        ]

        self.send(request)

        # Read response frame
        response = self._receive(5)

        # Validate response
        if response[0] != self.START_BYTE:
            raise Exception(f"Invalid start byte: {response[0]:#x}")

        if response[2] != channel:
            raise Exception(f"Channel mismatch: expected {channel}, got {response[2]}")

        # Verify checksum
        expected_checksum = self._calculate_checksum(
            response[0], response[1], response[2], response[3]
        )
        if response[4] != expected_checksum:
            raise Exception(f"Checksum mismatch")

        # Extract position data
        data_byte = response[3]
        position = self._map_from_byte(data_byte)

        return position
