import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.mode = 0
spi.max_speed_hz = 10000


def calculate_checksum(start, cmd, channel, data):
    return start ^ cmd ^ channel ^ data


def resync_spi():
    print("🔄 Próba resynchronizacji SPI...")
    for _ in range(5):
        resp = spi.xfer2([0x00] * 5)
        time.sleep(0.01)
        if resp[0] == 0xAA:
            print("✅ SPI zsynchronizowane pomyślnie!")
            return True
        # Send 1 byte to shift the STM32 internal frame counter by 1
        spi.xfer2([0x00])
        time.sleep(0.01)
    print("❌ Resynchronizacja nie powiodła się!")
    return False


def send_frame(cmd_type, channel, data):
    start_byte = 0xAA
    checksum = calculate_checksum(start_byte, cmd_type, channel, data)
    frame = [start_byte, cmd_type, channel, data, checksum]

    print(f"\n📤 Sending: {[hex(b) for b in frame]}")

    # Phase 1: Send request
    spi.xfer2(frame)
    
    # Polling Phase: Wait for real response
    max_retries = 20
    dummy = [0x00] * 5
    
    for i in range(max_retries):
        time.sleep(0.015)  # Wait 15ms before polling to give STM32 time to re-arm SPI
        response = spi.xfer2(dummy)
        
        # Check if we got a real response (starts with 0xAA and is not the dummy 0x00 cmd)
        # 0xCC is ACK, 0xFF is NACK (based on expected values)
        if response and response[0] == 0xAA and response[1] in (0xCC, 0xFF):
            print(f"📥 REAL response (poll {i+1}): {[hex(b) for b in response]}")
            return response
            
        print(f"   Poll {i+1} (transitioning): {[hex(b) for b in response]}")

    print("❌ Timeout waiting for response!")
    resync_spi()
    return [0x00] * 5


EXPECTED = {
    "SET ch0":  [0xaa, 0xcc, 0x0, 0x32, 0x54],
    "SET ch1":  [0xaa, 0xcc, 0x1, 0x3c, 0x5b],
    "GET ch0":  [0xaa, 0xcc, 0x0, 0x32, 0x54],
    "GET ch1":  [0xaa, 0xcc, 0x1, 0x3c, 0x5b],
}

stats = {key: {"ok": 0, "err": 0} for key in EXPECTED}


def check(label, response):
    expected = EXPECTED[label]
    if list(response) == expected:
        stats[label]["ok"] += 1
        status = "✅"
    else:
        stats[label]["err"] += 1
        status = f"❌ (expected {[hex(b) for b in expected]})"
    print(f"  {status} {label}: {[hex(b) for b in response]}")


iterations = 1
for i in range(0, iterations):
    print(f"\n{'='*50} Iteration {i+1}/{iterations} {'='*50}")

    response_from_set = send_frame(0x01, 0, 50)
    check("SET ch0", response_from_set)
    time.sleep(0.02)  # Give STM32 time to re-arm SPI between transactions

    response_from_set2 = send_frame(0x01, 1, 60)
    check("SET ch1", response_from_set2)
    time.sleep(0.02)

    response = send_frame(0x02, 0, 0x00)
    check("GET ch0", response)
    time.sleep(0.02)

    response = send_frame(0x02, 1, 0x00)
    check("GET ch1", response)
    time.sleep(0.02)


# ── Summary ──────────────────────────────────────────────────────────────────
total_ok  = sum(v["ok"]  for v in stats.values())
total_err = sum(v["err"] for v in stats.values())
total     = total_ok + total_err

print("\n" + "="*60)
print("📊  COMMUNICATION SUMMARY")
print("="*60)
print(f"{'Frame':<12} {'OK':>6} {'ERR':>6} {'Total':>7}  {'OK%':>6}")
print("-"*60)
for label, s in stats.items():
    t = s["ok"] + s["err"]
    pct = 100.0 * s["ok"] / t if t else 0.0
    print(f"{label:<12} {s['ok']:>6} {s['err']:>6} {t:>7}  {pct:>5.1f}%")
print("-"*60)
overall_pct = 100.0 * total_ok / total if total else 0.0
print(f"{'TOTAL':<12} {total_ok:>6} {total_err:>6} {total:>7}  {overall_pct:>5.1f}%")
print("="*60)


# Test połączenia
# send_frame(0x05, 1, 0x00)

