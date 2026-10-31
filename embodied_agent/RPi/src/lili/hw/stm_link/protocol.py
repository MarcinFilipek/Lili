"""Ramki protokołu RPi <-> STM32. Opis: embodied_agent/doc/protocol.md

Frame: [START, CMD, CHANNEL, DATA, XOR_CHECKSUM]
"""

START_BYTE = 0xAA
CMD_SET_POSITION = 0x01
CMD_GET_POSITION = 0x02
CMD_ACK = 0xCC

FRAME_LEN = 5

CHANNEL_PAN = 0
CHANNEL_TILT = 1


class ProtocolError(Exception):
    pass


def checksum(start, cmd, channel, data):
    return start ^ cmd ^ channel ^ data


def build_frame(cmd, channel, data):
    if not 0 <= data <= 0xFF:
        raise ValueError(f"data poza zakresem bajtu: {data}")
    return bytes([START_BYTE, cmd, channel, data, checksum(START_BYTE, cmd, channel, data)])


def parse_response(response, channel):
    """Sprawdza odpowiedź STM32 i zwraca bajt DATA."""
    if len(response) != FRAME_LEN:
        raise ProtocolError(f"Timeout: odebrano {len(response)}/{FRAME_LEN} bajtów")

    start, cmd, resp_channel, data, chk = response

    if start != START_BYTE:
        raise ProtocolError(f"Zły start byte: {start:#x}")
    if chk != checksum(start, cmd, resp_channel, data):
        raise ProtocolError(f"Zła suma kontrolna: {chk:#x}")
    if cmd != CMD_ACK:
        raise ProtocolError(f"Brak ACK, cmd={cmd:#x}")
    if resp_channel != channel:
        raise ProtocolError(f"Odpowiedź dla kanału {resp_channel}, oczekiwano {channel}")

    return data


# Python [-90°, +90°] <-> byte [0, 180]
def degrees_to_byte(degrees):
    return int(round(min(max(degrees, -90.0), 90.0) + 90.0))


def byte_to_degrees(byte_value):
    return float(byte_value) - 90.0
