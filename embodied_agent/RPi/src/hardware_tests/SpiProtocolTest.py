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

    # Phase 1: Send request (STM32 receives command, sends old buffer)
    phase1_response = spi.xfer2(frame)
    print(f"📥 Phase 1 (old buffer): {[hex(b) for b in phase1_response]}")

    # Wait for STM32 to process
    time.sleep(0.001)  # 1ms

    # Phase 2: Send dummy bytes (STM32 loads response into TX buffer)
    dummy = [0x00] * 5
    phase2_response = spi.xfer2(dummy)
    print(f"📥 Phase 2 (transitioning): {[hex(b) for b in phase2_response]}")

    # Small delay
    time.sleep(0.001)  # 1ms

    # Phase 3: Send dummy bytes AGAIN (NOW we get the real response!)
    phase3_response = spi.xfer2(dummy)
    print(f"📥 Phase 3 (REAL response): {[hex(b) for b in phase3_response]}")

    return phase3_response


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
