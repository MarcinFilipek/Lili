# Protokół komunikacji SPI - Dokumentacja

## Struktura ramki

Każda ramka SPI składa się z 5 bajtów:

| Bajt | Nazwa | Opis |
|------|-------|------|
| 0 | START_BYTE | 0xAA - znacznik początku ramki |
| 1 | CMD_TYPE | Typ komendy (0x01-0x06, 0xAA, 0xEE) |
| 2 | CHANNEL | Numer kanału (0-15) |
| 3 | DATA | Dane (kąt, PWM, itp.) |
| 4 | CHECKSUM | Suma kontrolna XOR (bajty 0^1^2^3) |

## Dostępne komendy

### Komendy sterujące (Master → Slave)

- **0x01 - SET_SERVO** - Ustaw kąt serwa
  - CHANNEL: numer serwa (0-15)
  - DATA: kąt w stopniach (0-180)
  - Odpowiedź: ACK z ustawionym kątem

- **0x02 - GET_SERVO** - Odczytaj aktualny kąt serwa
  - CHANNEL: numer serwa (0-15)
  - DATA: 0x00 (ignorowane)
  - Odpowiedź: ACK z aktualnym kątem w polu DATA

- **0x03 - SET_PWM** - Ustaw wypełnienie PWM
  - CHANNEL: numer kanału PWM (1-2)
  - DATA: wypełnienie w % (0-100)
  - Odpowiedź: ACK z ustawionym wypełnieniem

- **0x04 - GET_PWM** - Odczytaj wypełnienie PWM (nie zaimplementowane)
  - CHANNEL: numer kanału PWM (1-2)
  - DATA: 0x00
  - Odpowiedź: ACK z aktualnym wypełnieniem

- **0x05 - PING** - Test połączenia
  - CHANNEL: 0x00 (ignorowane)
  - DATA: 0x00 (ignorowane)
  - Odpowiedź: ACK z sygnaturą 0x55 w polu DATA

- **0x06 - RESET** - Zatrzymaj animację serw
  - CHANNEL: 0x00 (ignorowane)
  - DATA: 0x00 (ignorowane)
  - Odpowiedź: ACK z 0x00 w polu DATA

### Odpowiedzi (Slave → Master)

- **0xAA - ACK** - Potwierdzenie wykonania komendy
  - CHANNEL: echo z komendy
  - DATA: wynik operacji lub wartość odczytana

- **0xEE - NACK** - Błąd wykonania komendy
  - CHANNEL: echo z komendy
  - DATA: 0xFF

## Przykład użycia - Python (Raspberry Pi)

```python
import spidev
import time

class STM32_SPI:
    def __init__(self, bus=0, device=0, speed=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0  # CPOL=0, CPHA=0
        
        self.START_BYTE = 0xAA
        self.CMD_SET_SERVO = 0x01
        self.CMD_GET_SERVO = 0x02
        self.CMD_SET_PWM = 0x03
        self.CMD_GET_PWM = 0x04
        self.CMD_PING = 0x05
        self.CMD_RESET = 0x06
        self.CMD_ACK = 0xAA
        self.CMD_NACK = 0xEE
    
    def calculate_checksum(self, start, cmd, channel, data):
        """Oblicz sumę kontrolną XOR"""
        return start ^ cmd ^ channel ^ data
    
    def send_frame(self, cmd_type, channel, data):
        """Wyślij ramkę do STM32 i odbierz odpowiedź"""
        checksum = self.calculate_checksum(self.START_BYTE, cmd_type, channel, data)
        frame = [self.START_BYTE, cmd_type, channel, data, checksum]
        
        response = self.spi.xfer2(frame)
        time.sleep(0.01)  # Krótkie opóźnienie na przetworzenie
        
        return response
    
    def set_servo_angle(self, channel, angle):
        """
        Ustaw kąt serwa (0-180°)
        
        Args:
            channel: numer serwa (0-15)
            angle: kąt w stopniach (0-180)
        
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        if angle < 0 or angle > 180:
            raise ValueError("Angle must be 0-180")
        
        response = self.send_frame(self.CMD_SET_SERVO, channel, angle)
        
        if len(response) >= 5 and response[1] == self.CMD_ACK:
            return True
        return False
    
    def get_servo_angle(self, channel):
        """
        Odczytaj aktualny kąt serwa
        
        Args:
            channel: numer serwa (0-15)
        
        Returns:
            Aktualny kąt serwa lub None w przypadku błędu
        """
        response = self.send_frame(self.CMD_GET_SERVO, channel, 0x00)
        
        # Odpowiedź STM32 to ramka ACK z kątem w polu DATA
        if len(response) >= 5 and response[1] == self.CMD_ACK:
            return response[3]  # Bajt DATA
        return None
    
    def set_pwm(self, channel, duty_cycle):
        """
        Ustaw wypełnienie PWM (0-100%)
        
        Args:
            channel: numer kanału PWM (1-2)
            duty_cycle: wypełnienie w procentach (0-100)
        
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        if duty_cycle < 0 or duty_cycle > 100:
            raise ValueError("Duty cycle must be 0-100")
        
        response = self.send_frame(self.CMD_SET_PWM, channel, duty_cycle)
        
        if len(response) >= 5 and response[1] == self.CMD_ACK:
            return True
        return False
    
    def ping(self):
        """
        Sprawdź połączenie z STM32
        
        Returns:
            True jeśli STM32 odpowiada, False w przeciwnym razie
        """
        response = self.send_frame(self.CMD_PING, 0, 0x00)
        
        # Sprawdź czy otrzymano ACK z sygnaturą 0x55
        if len(response) >= 5 and response[1] == self.CMD_ACK and response[3] == 0x55:
            return True
        return False
    
    def reset_animation(self):
        """
        Zatrzymaj automatyczną animację serw
        
        Returns:
            True jeśli sukces, False w przeciwnym razie
        """
        response = self.send_frame(self.CMD_RESET, 0, 0x00)
        
        if len(response) >= 5 and response[1] == self.CMD_ACK:
            return True
        return False
    
    def close(self):
        """Zamknij połączenie SPI"""
        self.spi.close()


# Przykład użycia
if __name__ == "__main__":
    stm = STM32_SPI()
    
    try:
        # Test połączenia
        print("Sprawdzam połączenie z STM32...")
        if stm.ping():
            print("✓ STM32 odpowiada!")
        else:
            print("✗ Brak połączenia z STM32")
            exit(1)
        
        # Zatrzymaj animację
        print("\nZatrzymuję animację...")
        stm.reset_animation()
        time.sleep(0.5)
        
        # Ustaw servo 0 na 90°
        print("\nUstawiam servo 0 na 90°")
        stm.set_servo_angle(0, 90)
        time.sleep(0.5)
        
        # Odczytaj pozycję
        angle = stm.get_servo_angle(0)
        print(f"Aktualna pozycja servo 0: {angle}°")
        
        # Ustaw servo 1 na 45°
        print("\nUstawiam servo 1 na 45°")
        stm.set_servo_angle(1, 45)
        time.sleep(0.5)
        
        angle = stm.get_servo_angle(1)
        print(f"Aktualna pozycja servo 1: {angle}°")
        
        # Animacja serwa 0-180°
        print("\nAnimacja serw 0-180°...")
        for angle in range(0, 181, 10):
            stm.set_servo_angle(0, angle)
            stm.set_servo_angle(1, angle)
            print(f"Kąt: {angle}°", end='\r')
            time.sleep(0.1)
        print("\n")
        
        # Ustaw PWM
        print("Ustawiam PWM kanał 1 na 75%")
        stm.set_pwm(1, 75)
        time.sleep(1)
        
        print("Ustawiam PWM kanał 2 na 50%")
        stm.set_pwm(2, 50)
        
        print("\n✓ Test zakończony pomyślnie!")
        
    except KeyboardInterrupt:
        print("\n\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"\n✗ Błąd: {e}")
    finally:
        stm.close()
        print("Zamknięto połączenie SPI")
```

## Przykładowe scenariusze

### 1. Ustawienie pozycji serwa

**Master wysyła:**
```
[0xAA, 0x01, 0x00, 0x5A, 0xFB]
```
- START: 0xAA
- CMD: 0x01 (SET_SERVO)
- CHANNEL: 0x00 (servo 0)
- DATA: 0x5A (90°)
- CHECKSUM: 0xAA ^ 0x01 ^ 0x00 ^ 0x5A = 0xFB

**Slave odpowiada:**
```
[0xAA, 0xAA, 0x00, 0x5A, 0x00]
```
- Potwierdzenie ustawienia serwa 0 na 90°

### 2. Odczyt pozycji serwa

**Master wysyła:**
```
[0xAA, 0x02, 0x01, 0x00, 0xA9]
```
- GET_SERVO dla kanału 1

**Slave odpowiada:**
```
[0xAA, 0xAA, 0x01, 0x2D, 0x86]
```
- Aktualna pozycja serwa 1: 0x2D (45°)

### 3. Test połączenia

**Master wysyła:**
```
[0xAA, 0x05, 0x00, 0x00, 0xAF]
```
- PING

**Slave odpowiada:**
```
[0xAA, 0xAA, 0x00, 0x55, 0xFF]
```
- ACK z sygnaturą 0x55

## Uwagi implementacyjne

1. **Timeout**: Zalecane jest ustawienie timeout na odpowiedź ze Slave (~100ms)
2. **Retry**: W przypadku braku odpowiedzi lub błędnej checksumy, powtórz wysłanie ramki
3. **Opóźnienia**: Po każdej transakcji dodaj krótkie opóźnienie (10ms) na przetworzenie
4. **Walidacja**: Zawsze sprawdzaj checksumę w odpowiedzi od Slave
5. **SPI Mode**: Używaj SPI Mode 0 (CPOL=0, CPHA=0)
6. **Prędkość**: Zalecana prędkość do 1 MHz

## Rozszerzenia protokołu

Protokół można łatwo rozszerzyć o dodatkowe komendy:

- **GET_STATUS** - Odczyt statusu systemu
- **SET_LED** - Sterowanie LED
- **GET_SENSORS** - Odczyt czujników
- **SET_MODE** - Zmiana trybu pracy

Wystarczy dodać nowe wartości do enum `SPI_CommandType` i obsługę w funkcji `ProcessSPIFrame()`.
