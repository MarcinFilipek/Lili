import serial
import time
import sys

# Otwórz port szeregowy
# Domyślny sprzętowy port na RPi to /dev/serial0 (lub /dev/ttyAMA0 / /dev/ttyS0)
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=0.1)
except Exception as e:
    print(f"❌ Nie udało się otworzyć portu UART: {e}")
    print("💡 Upewnij się, że port szeregowy jest włączony w raspi-config!")
    print("   Uruchom: sudo raspi-config -> Interfacing Options -> Serial")
    print("   - Would you like a login shell to be accessible over serial? -> NO")
    print("   - Would you like the serial port hardware to be enabled? -> YES")
    sys.exit(1)

def calculate_checksum(start, cmd, channel, data):
    return start ^ cmd ^ channel ^ data

def send_frame(cmd_type, channel, data):
    start_byte = 0xAA
    checksum = calculate_checksum(start_byte, cmd_type, channel, data)
    frame = bytearray([start_byte, cmd_type, channel, data, checksum])

    # Czyszczenie bufora wejściowego z ewentualnych śmieci
    ser.reset_input_buffer()
    
    # Wyślij zapytanie
    ser.write(frame)
    
    # Czekaj na dokładnie 5 bajtów odpowiedzi (timeout to 0.1s)
    response = ser.read(5)
    
    if len(response) == 5:
        # Sprawdzamy czy odpowiedź ma sens
        if response[0] == 0xAA:
            return list(response)
        else:
            print(f"\n❌ Błędna ramka odpowiedzi: {[hex(b) for b in response]}")
    else:
        print(f"\n❌ Timeout! Odebrano tylko {len(response)} bajtów.")
        
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
    else:
        stats[label]["err"] += 1

iterations = 100
print(f"🚀 Uruchamianie stres-testu UART na {iterations} iteracji...")

for i in range(0, iterations):
    # Print progress
    sys.stdout.write(f"\rPrzetwarzanie iteracji {i+1}/{iterations}...")
    sys.stdout.flush()

    response_from_set = send_frame(0x01, 0, 50)
    check("SET ch0", response_from_set)

    response_from_set2 = send_frame(0x01, 1, 60)
    check("SET ch1", response_from_set2)

    response = send_frame(0x02, 0, 0x00)
    check("GET ch0", response)

    response = send_frame(0x02, 1, 0x00)
    check("GET ch1", response)

print("\n\n" + "="*60)
print("📊  COMMUNICATION SUMMARY (UART)")
print("="*60)
print(f"{'Frame':<12} {'OK':>6} {'ERR':>6} {'Total':>7}  {'OK%':>6}")
print("-"*60)
for label, s in stats.items():
    t = s["ok"] + s["err"]
    pct = 100.0 * s["ok"] / t if t else 0.0
    print(f"{label:<12} {s['ok']:>6} {s['err']:>6} {t:>7}  {pct:>5.1f}%")
print("-"*60)
total_ok  = sum(v["ok"]  for v in stats.values())
total_err = sum(v["err"] for v in stats.values())
total     = total_ok + total_err
overall_pct = 100.0 * total_ok / total if total else 0.0
print(f"{'TOTAL':<12} {total_ok:>6} {total_err:>6} {total:>7}  {overall_pct:>5.1f}%")
print("="*60)
