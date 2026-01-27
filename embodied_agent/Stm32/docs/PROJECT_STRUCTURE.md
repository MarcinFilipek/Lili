# Struktura projektu STM32

## Przegląd

Projekt został zrefaktoryzowany w celu zwiększenia modularności, czytelności i łatwości utrzymania kodu.

## Struktura katalogów

```
Stm32/
├── include/              # Pliki nagłówkowe (.h)
│   ├── pca9685.h        # Sterownik PCA9685 (16-kanałowy PWM)
│   ├── pwm_driver.h     # Obsługa PWM (TIM3)
│   ├── spi_comm.h       # Komunikacja SPI z Raspberry Pi
│   ├── servo_control.h  # Kontroler serwomechanizmów
│   └── system_config.h  # Konfiguracja systemu (zegar, I2C, GPIO)
│
├── src/                 # Pliki źródłowe (.c/.cpp)
│   ├── main.cpp         # Główna aplikacja
│   ├── it.c             # Procedury obsługi przerwań
│   ├── pca9685.c        # Implementacja sterownika PCA9685
│   ├── pwm_driver.c     # Implementacja PWM
│   ├── spi_comm.c       # Implementacja SPI
│   ├── servo_control.c  # Implementacja kontrolera serw
│   └── system_config.c  # Implementacja konfiguracji systemu
│
├── docs/                # Dokumentacja
│   ├── EXAMPLES.md
│   ├── PCA9685.md
│   └── PROJECT_STRUCTURE.md (ten plik)
│
└── platformio.ini       # Konfiguracja PlatformIO
```

## Moduły projektu

### 1. system_config (system_config.h/c)

**Odpowiedzialność:** Konfiguracja podstawowych zasobów systemowych

**Funkcje:**
- `SystemClock_Config()` - konfiguracja zegara systemowego (48 MHz)
- `System_I2C_Init()` - inicjalizacja I2C1 dla PCA9685
- `System_GPIO_Init()` - inicjalizacja GPIO (LED na PB3)
- `System_ErrorHandler()` - obsługa błędów (miganie LED)
- `HAL_I2C_MspInit()` - callback HAL dla I2C (konfiguracja pinów)

**Eksportowane zasoby:**
- `hi2c1` - handle I2C1

---

### 2. pwm_driver (pwm_driver.h/c)

**Odpowiedzialność:** Obsługa sprzętowego PWM (TIM3)

**Funkcje:**
- `PWM_Init()` - inicjalizacja TIM3 na częstotliwość 18 kHz
- `PWM_SetDutyCycle_CH1(uint8_t duty)` - ustawienie duty cycle (0-100%) na kanale 1 (PB4)
- `PWM_SetDutyCycle_CH2(uint8_t duty)` - ustawienie duty cycle (0-100%) na kanale 2 (PB5)

**Eksportowane zasoby:**
- `htim3` - handle timera TIM3

**Parametry PWM:**
- Częstotliwość: 18 kHz
- Rozdzielczość: 2666 kroków
- Kanały: TIM3_CH1 (PB4), TIM3_CH2 (PB5)

---

### 3. spi_comm (spi_comm.h/c)

**Odpowiedzialność:** Komunikacja SPI z Raspberry Pi (tryb Slave)

**Funkcje:**
- `SPI_Init()` - inicjalizacja SPI1 jako Slave
- `SPI_SendData(uint8_t *data, uint16_t size)` - wysyłanie danych
- `SPI_GetLastReceivedByte()` - pobranie ostatniego odebranego bajtu
- `HAL_SPI_RxCpltCallback()` - callback po odebraniu danych
- `HAL_SPI_TxCpltCallback()` - callback po wysłaniu danych
- `SPI1_IRQHandler()` - procedura obsługi przerwania SPI1

**Eksportowane zasoby:**
- `hspi1` - handle SPI1
- `spi_rx_buffer[64]` - bufor odbiorczy
- `spi_tx_buffer[64]` - bufor nadawczy
- `spi_rx_index` - indeks odbiorczy
- `spi_data_ready` - flaga gotowości danych

**Konfiguracja SPI:**
- Tryb: Slave
- Częstotliwość: sterowana przez Master
- Piny: PA4 (NSS), PA5 (SCK), PA6 (MISO), PA7 (MOSI)
- Tryb komunikacji: Mode 0 (CPOL=0, CPHA=0)

---

### 4. servo_control (servo_control.h/c)

**Odpowiedzialność:** Zaawansowane sterowanie serwomechanizmami

**Struktury:**
```c
typedef struct {
    uint8_t channel;      // Kanał PCA9685 (0-15)
    uint16_t min_us;      // Minimalny impuls (µs)
    uint16_t max_us;      // Maksymalny impuls (µs)
} ServoConfig_t;

typedef struct {
    uint8_t current_angle;  // Aktualny kąt
    uint8_t target_angle;   // Kąt docelowy
    uint8_t step;          // Krok zmiany
    uint8_t active;        // Flaga aktywności
} ServoAnimation_t;
```

**Funkcje:**
- `Servo_Init()` - inicjalizacja serwa z początkowym kątem
- `Servo_SetAngleWithConfig()` - ustawienie kąta z wykorzystaniem konfiguracji
- `Servo_StartSweep()` - uruchomienie animacji przesuwu
- `Servo_UpdateSweep()` - aktualizacja animacji (wywołuj w pętli)

**Przykład użycia:**
```cpp
ServoConfig_t myServo = {0, 500, 2500};  // Kanał 0, 500-2500µs
ServoAnimation_t animation;

Servo_Init(&hi2c1, &myServo, 90);         // Startuj z 90°
Servo_StartSweep(&animation, 0, 180, 1);  // Animacja 0-180° co 1°

// W pętli:
uint8_t angle = Servo_UpdateSweep(&animation);
Servo_SetAngleWithConfig(&hi2c1, &myServo, angle);
```

---

### 5. pca9685 (pca9685.h/c)

**Odpowiedzialność:** Sterownik dla 16-kanałowego kontrolera PWM PCA9685

**Funkcje:**
- `PCA9685_Init()` - inicjalizacja układu
- `PCA9685_SetPWMFreq()` - ustawienie częstotliwości PWM
- `PCA9685_SetPWM()` - surowe ustawienie PWM (0-4095)
- `PCA9685_SetServoAngle()` - ustawienie kąta serwa (0-180°)
- `PCA9685_SetServoPulse()` - ustawienie długości impulsu (µs)

**Konfiguracja:**
- Adres I2C: 0x40
- Częstotliwość PWM: 50 Hz (typowa dla serw)
- Rozdzielczość: 12-bit (0-4095)
- Kanały: 0-15

---

## main.cpp - Główna aplikacja

Plik główny zawiera tylko logikę aplikacji:

```cpp
// Deklaracje globalnych konfiguracji
ServoConfig_t servoSG5010, servo1201MG;
ServoAnimation_t servoAnimation;

int main() {
    // 1. Inicjalizacja systemu
    HAL_Init();
    SystemClock_Config();
    
    // 2. Konfiguracja serwomechanizmów
    servoSG5010.channel = 0;
    servoSG5010.min_us = 500;
    servoSG5010.max_us = 2500;
    
    // 3. Inicjalizacja peryferiów
    PWM_Init();
    System_I2C_Init();
    System_GPIO_Init();
    
    // 4. Inicjalizacja PCA9685
    PCA9685_Init(&hi2c1);
    
    // 5. Inicjalizacja serwomechanizmów
    Servo_Init(&hi2c1, &servoSG5010, 90);
    
    // 6. Inicjalizacja SPI
    SPI_Init();
    
    // 7. Główna pętla
    while(1) {
        // Obsługa LED, serw, SPI
    }
}
```

---

## Mapowanie pinów

### GPIO
- **PB3**: LED (status)
- **PB4**: TIM3_CH1 (PWM)
- **PB5**: TIM3_CH2 (PWM)

### I2C1 (PCA9685)
- **PF0**: I2C1_SDA
- **PF1**: I2C1_SCL

### SPI1 (Raspberry Pi)
- **PA4**: SPI1_NSS (Chip Select)
- **PA5**: SPI1_SCK (Clock)
- **PA6**: SPI1_MISO (Slave Out)
- **PA7**: SPI1_MOSI (Slave In)

---

## Przepływ danych

```
Raspberry Pi (Master)
        ↓ SPI
    STM32F042K6
        ↓ I2C
      PCA9685
        ↓ PWM
   Serwomechanizmy
```

---

## Zużycie zasobów

Po kompilacji:
- **RAM**: 7.3% (448/6144 bajtów)
- **Flash**: 45.6% (14940/32768 bajtów)

---

## Kompilacja i wgrywanie

```bash
# Kompilacja
platformio run

# Wgranie do płytki
platformio run --target upload

# Czyszczenie
platformio run --target clean
```

---

## Rozszerzanie projektu

### Dodanie nowego modułu:

1. Utwórz `include/nowy_modul.h`
2. Utwórz `src/nowy_modul.c`
3. Dodaj `#include "nowy_modul.h"` w `main.cpp`
4. Wywołaj funkcje inicjalizacyjne w `main()`

### Przykład - moduł UART:

```c
// include/uart_comm.h
#ifndef UART_COMM_H
#define UART_COMM_H
#include "stm32f0xx_hal.h"

void UART_Init(void);
void UART_SendString(const char *str);
#endif

// src/uart_comm.c
#include "uart_comm.h"

UART_HandleTypeDef huart;

void UART_Init(void) {
    // Implementacja...
}
```

---

## Debugowanie

### LED jako wskaźnik błędów:
- **Wolne miganie (500ms)**: Normalna praca
- **Szybkie miganie (100ms)**: Błąd inicjalizacji PCA9685

### Najczęstsze problemy:

1. **PCA9685 nie odpowiada**:
   - Sprawdź połączenia I2C (SDA, SCL)
   - Sprawdź rezystory pull-up (2.2-10kΩ)
   - Sprawdź zasilanie PCA9685

2. **SPI nie działa**:
   - Sprawdź poziomy napięć (3.3V)
   - Sprawdź ustawienia SPI na RPi (mode 0)
   - Sprawdź połączenie NSS (CS)

3. **Serwa drżą**:
   - Oddziel zasilanie serw od STM32
   - Dodaj kondensatory wygładzające
   - Sprawdź wartości min_us/max_us

---

## Autorzy

Projekt STM32 dla platformy robotycznej Lili

## Licencja

MIT License
