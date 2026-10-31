import pytest

from lili.hw.stm_link import protocol as p


class FakeSerial:
    """Udaje STM32: odpowiada ACK z echem kanału i bieżącą pozycją."""

    def __init__(self):
        self.positions = {0: 90, 1: 90}
        self.written = []
        self._out = b""
        self.override = None

    def reset_input_buffer(self):
        self._out = b""

    def write(self, frame):
        self.written.append(bytes(frame))
        _, cmd, ch, data, _ = frame
        if cmd == p.CMD_SET_POSITION:
            self.positions[ch] = data
        pos = self.positions[ch]
        self._out = bytes([p.START_BYTE, p.CMD_ACK, ch, pos, p.checksum(p.START_BYTE, p.CMD_ACK, ch, pos)])

    def read(self, n):
        out = self.override if self.override is not None else self._out
        return out[:n]

    def close(self):
        pass


@pytest.fixture
def ser():
    return FakeSerial()
