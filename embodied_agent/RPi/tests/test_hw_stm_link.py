import pytest

from lili.hw.stm_link import StmLink, ProtocolError


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
