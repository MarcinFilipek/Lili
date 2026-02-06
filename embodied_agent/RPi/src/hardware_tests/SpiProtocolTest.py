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

    print(f"\n📤 Sending: {[hex(b) for b in frame]}")

    # Phase 1: Send request (STM32 receives command, sends garbage back)
    phase1_response = spi.xfer2(frame)
    print(
        f"📥 Phase 1 response (should be garbage): {[hex(b) for b in phase1_response]}"
    )

    # Wait for STM32 to process (try longer delay to be safe)
    time.sleep(0.005)  # 5ms - much longer than 150μs measured

    # Phase 2: Send dummy bytes (STM32 sends REAL response during this)
    dummy = [0x00] * 5
    phase2_response = spi.xfer2(dummy)
    print(f"📥 Phase 2 response (should be ACK): {[hex(b) for b in phase2_response]}")

    return phase2_response


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
