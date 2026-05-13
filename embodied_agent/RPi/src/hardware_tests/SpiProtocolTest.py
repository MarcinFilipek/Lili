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
    # print(f"📥 Phase 1 (old buffer): {[hex(b) for b in phase1_response]}")

    # Wait for STM32 to process
    time.sleep(0.03)  # 20ms

    # Phase 2: Send dummy bytes (STM32 loads response into TX buffer)
    dummy = [0x00] * 5
    phase2_response = spi.xfer2(dummy)
    # print(f"📥 Phase 2 (transitioning): {[hex(b) for b in phase2_response]}")

    # Small delay
    time.sleep(0.03)  # 20ms

    # Phase 3: Send dummy bytes AGAIN (NOW we get the real response!)
    phase3_response = spi.xfer2(dummy)
    # print(f"📥 Phase 3 (REAL response): {[hex(b) for b in phase3_response]}")

    return phase3_response


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


iterations = 100
for i in range(0, iterations):
    print(f"\n{'='*50} Iteration {i+1}/{iterations} {'='*50}")

    response_from_set = send_frame(0x01, 0, 50)
    # response should be ['0xaa', '0xcc', '0x0', '0x32', '0x54']
    check("SET ch0", response_from_set)

    response_from_set2 = send_frame(0x01, 1, 60)
    # response should be ['0xaa', '0xcc', '0x1', '0x3c', '0x5b']
    check("SET ch1", response_from_set2)

    response = send_frame(0x02, 0, 0x00)
    # response should be ['0xaa', '0xcc', '0x0', '0x32', '0x54']
    check("GET ch0", response)

    response = send_frame(0x02, 1, 0x00)
    # response should be ['0xaa', '0xcc', '0x1', '0x3c', '0x5b']
    check("GET ch1", response)


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

