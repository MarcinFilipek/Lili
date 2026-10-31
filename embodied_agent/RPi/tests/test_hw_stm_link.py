import pytest

from lili.hw.stm_link import StmLink, ProtocolError
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


def test_frames_match_stm_wire_format(ser):
    # Ramki z dotychczasowego stres-testu UART
    StmLink(serial_port=ser).set_position(0, -40.0)
    assert ser.written[-1] == bytes([0xAA, 0x01, 0x00, 0x32, 0xAA ^ 0x01 ^ 0x00 ^ 0x32])


def test_set_then_get_roundtrip(ser):
    link = StmLink(serial_port=ser)
    assert link.set_position(1, 25.0) == 25.0
    assert link.get_position(1) == 25.0


def test_degrees_are_clamped(ser):
    link = StmLink(serial_port=ser)
    assert link.set_position(0, 200.0) == 90.0
    assert link.set_position(0, -200.0) == -90.0


@pytest.mark.parametrize(
    "response",
    [
        b"",  # timeout
        bytes([0xAB, 0xCC, 0, 90, 0xAB ^ 0xCC ^ 0 ^ 90]),  # start byte
        bytes([0xAA, 0xCC, 0, 90, 0x00]),  # checksum
        bytes([0xAA, 0xCC, 1, 90, 0xAA ^ 0xCC ^ 1 ^ 90]),  # zły kanał
        bytes([0xAA, 0x01, 0, 90, 0xAA ^ 0x01 ^ 0 ^ 90]),  # brak ACK
    ],
)
def test_bad_responses_raise(ser, response):
    ser.override = response
    with pytest.raises(ProtocolError):
        StmLink(serial_port=ser).get_position(0)
