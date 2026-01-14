# Przykłady użycia - STM32 + Raspberry Pi

## Przykład 1: Podstawowa komunikacja SPI

### Na Raspberry Pi:

```bash
cd embodied_agent/RPi
source venv/bin/activate
python src/SpiTest.py
```

Wprowadź komendy:
```
Enter command (A/B): A
Sending 'A'...
Sent: [65], Received: [0]

Enter command (A/B): B
Sending 'B'...
Sent: [66], Received: [0]
```

### Efekt na STM32:
- Komenda `A` → PWM CH1 (PB4) = 0%
- Komenda `B` → PWM CH1 (PB4) = 100%

---

## Przykład 2: Sterowanie mocą silnika (0-90%)

### Python (RPi):

```python
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000
spi.mode = 0

# Ustaw moc na 50%
spi.xfer2([ord('5')])
time.sleep(0.1)

# Ustaw moc na 90%
spi.xfer2([ord('9')])
time.sleep(0.1)

# Wyłącz
spi.xfer2([ord('A')])

spi.close()
```

---

## Przykład 3: Płynna zmiana mocy

### Python (RPi):

```python
import spidev
import time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000
spi.mode = 0

# Płynne zwiększanie mocy od 0% do 90%
for power in range(10):
    spi.xfer2([ord(str(power))])
    time.sleep(0.5)

# Płynne zmniejszanie mocy od 90% do 0%
for power in range(9, -1, -1):
    spi.xfer2([ord(str(power))])
    time.sleep(0.5)

spi.close()
```

---

## Przykład 4: Rozszerzona funkcjonalność (modyfikacja kodu STM32)

### Dodanie nowych komend

W pliku `src/main.cpp`, w pętli głównej:

```cpp
if (received == 'A') {
    PWM_SetDutyCycle_CH1(0);
}
else if (received == 'B') {
    PWM_SetDutyCycle_CH1(100);
}
else if (received == 'C') {  // NOWA KOMENDA
    PWM_SetDutyCycle_CH1(25);
    PWM_SetDutyCycle_CH2(75);
}
else if (received == 'D') {  // NOWA KOMENDA
    PWM_SetDutyCycle_CH1(75);
    PWM_SetDutyCycle_CH2(25);
}
```

---

## Przykład 5: Dwukierunkowa komunikacja

### Modyfikacja STM32 (src/main.cpp):

Odkomentuj linie 271-272:

```cpp
// Wyślij status z powrotem do RPi
uint8_t response = 0xAA;  // Kod potwierdzenia
SPI_SendData(&response, 1);
```

### Python (RPi):

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000
spi.mode = 0

# Wyślij komendę i odbierz odpowiedź
response = spi.xfer2([ord('B')])
print(f"STM32 odpowiedział: {hex(response[0])}")  # Powinno być 0xAA

spi.close()
```

---

## Przykład 6: Sterowanie dwoma silnikami niezależnie

### Modyfikacja STM32:

```cpp
// W main.cpp, dodaj obsługę dla obu kanałów
if (received == 'A') {
    PWM_SetDutyCycle_CH1(0);
}
else if (received == 'B') {
    PWM_SetDutyCycle_CH1(100);
}
else if (received == 'C') {
    PWM_SetDutyCycle_CH2(0);
}
else if (received == 'D') {
    PWM_SetDutyCycle_CH2(100);
}
else if (received >= '0' && received <= '9') {
    // Cyfry parzyste → CH1, nieparzyste → CH2
    if ((received - '0') % 2 == 0) {
        PWM_SetDutyCycle_CH1((received - '0') * 10);
    } else {
        PWM_SetDutyCycle_CH2((received - '0') * 10);
    }
}
```

---

## Przykład 7: Integracja z kamerą

### Python (RPi) - Sterowanie na podstawie obrazu:

```python
import cv2
from picamera2 import Picamera2
import spidev
import time

# Inicjalizacja SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000
spi.mode = 0

# Inicjalizacja kamery
picam = Picamera2()
cam_conf = picam.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam.configure(cam_conf)
picam.start()

try:
    while True:
        frame = picam.capture_array()
        
        # Przykład: wykrywanie jasności
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = gray.mean()
        
        # Steruj silnikiem na podstawie jasności
        if brightness < 50:
            spi.xfer2([ord('A')])  # Ciemno → wyłącz
        elif brightness < 100:
            spi.xfer2([ord('3')])  # Średnio → 30%
        else:
            spi.xfer2([ord('7')])  # Jasno → 70%
        
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)

finally:
    cv2.destroyAllWindows()
    picam.stop()
    spi.close()
```

---

## Debugowanie

### Sprawdzanie połączenia SPI

```python
import spidev

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 500000
spi.mode = 0

# Test połączenia
for i in range(10):
    response = spi.xfer2([ord('A')])
    print(f"Test {i+1}: Wysłano 'A', Odebrano: {response}")
    time.sleep(0.5)

spi.close()
```

### Pomiar PWM (oscyloskop/multimetr)

- Podłącz oscyloskop do PB4 (CH1) lub PB5 (CH2)
- Wyślij komendę `'5'` (50%)
- Sprawdź:
  - Częstotliwość: ~18 kHz
  - Duty cycle: ~50%
  - Napięcie: 0-3.3V

---

## Wskazówki

1. **Zawsze** sprawdź połączenie GND między RPi a STM32
2. **Nie** podłączaj silników bezpośrednio do STM32 - użyj sterownika
3. **Testuj** najpierw z LED zamiast silników
4. **Dodaj** opóźnienia między komendami SPI (min. 10ms)
5. **Używaj** oscyloskopu do weryfikacji sygnałów PWM
