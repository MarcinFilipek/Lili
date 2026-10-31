from . import protocol as p

DEFAULT_PORT = "/dev/ttyAMA0"
DEFAULT_BAUDRATE = 115200


class StmLink:
    """Sterowanie serwami na STM32 przez UART. Kąty w stopniach [-90, 90]."""

    def __init__(self, port=DEFAULT_PORT, baudrate=DEFAULT_BAUDRATE, timeout=0.1, serial_port=None):
        if serial_port is None:
            import serial

            serial_port = serial.Serial(port, baudrate, timeout=timeout)
        self.ser = serial_port

    def set_position(self, channel, degrees):
        """Ustawia serwo i zwraca pozycję potwierdzoną przez STM32."""
        data = self._transact(p.CMD_SET_POSITION, channel, p.degrees_to_byte(degrees))
        return p.byte_to_degrees(data)

    def get_position(self, channel):
        data = self._transact(p.CMD_GET_POSITION, channel, 0x00)
        return p.byte_to_degrees(data)

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _transact(self, cmd, channel, data):
        self.ser.reset_input_buffer()
        self.ser.write(p.build_frame(cmd, channel, data))
        return p.parse_response(self.ser.read(p.FRAME_LEN), channel)
