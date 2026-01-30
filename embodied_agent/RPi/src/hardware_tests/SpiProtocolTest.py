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
    response = spi.xfer2(frame)
    time.sleep(0.05)
    return response

# Ustaw servo 0 na 90 stopni
send_frame(0x01, 1, 80)

# Odczytaj pozycję servo 0
response = send_frame(0x02, 1, 0x00)

# Test połączenia
send_frame(0x05, 1, 0x00)