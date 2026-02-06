import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.mode = 0
spi.max_speed_hz = 500000


def calculate_checksum(start, cmd, channel, data):
    return start ^ cmd ^ channel ^ data


def send_frame(cmd_type, channel, data):
    start_byte = 0xAA
    checksum = calculate_checksum(start_byte, cmd_type, channel, data)
    frame = [start_byte, cmd_type, channel, data, checksum]

    # Phase 1: Send request (STM32 receives command, sends garbage back)
    spi.xfer2(frame)

    # Wait for STM32 to process (150μs measured, use 500μs for safety)
    time.sleep(0.0005)  # 0.5ms

    # Phase 2: Send dummy bytes (STM32 sends REAL response during this)
    dummy = [0x00] * 5
    response = spi.xfer2(dummy)

    return response


# Ustaw servo 0 na 90 stopni
response_from_set = send_frame(0x01, 0, 90)
print(f"SET response: {[hex(b) for b in response_from_set]}")

# Ustaw servo 1 na 90 stopni
response_from_set2 = send_frame(0x01, 1, 90)
print(f"SET response: {[hex(b) for b in response_from_set2]}")

# Odczytaj pozycję servo 1
response = send_frame(0x02, 1, 0x00)
print(f"GET response: {[hex(b) for b in response]}")

# Test połączenia
send_frame(0x05, 1, 0x00)
