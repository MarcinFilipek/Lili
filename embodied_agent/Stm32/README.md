# Lili — STM32

Firmware dla **Nucleo-F042K6**: odbiera komendy z Raspberry Pi 5 przez UART i steruje dwoma serwami
(pan/tilt) przez sterownik PCA9685 na I2C.

Protokół komunikacji: [../doc/protocol.md](../doc/protocol.md)

## Sprzęt

| Parametr | Wartość |
|----------|---------|
| Mikrokontroler | STM32F042K6T6 (Cortex-M0, 48 MHz z HSI + PLL) |
| Framework | STM32Cube HAL (STM32CubeF0 1.11.5) |
| Build | CMake, STM32CubeIDE for VS Code |
| Serwa | PCA9685, I2C, adres 0x40, 50 Hz |

## Pinout

### UART (USART1) — do RPi 5

| Pin STM32 | Funkcja | RPi 5 |
|-----------|---------|-------|
| PA9 (D1) | TX | GPIO 15 RXD (pin 10) |
| PA10 (D0) | RX | GPIO 14 TXD (pin 8) |
| GND | masa | GND |

115200 8N1, odbiór na przerwaniach, kolejka 16 ramek.

### I2C1 — do PCA9685

| Pin STM32 | Funkcja |
|-----------|---------|
| PF0 | SDA |
| PF1 | SCL |

### Pozostałe

| Pin STM32 | Funkcja |
|-----------|---------|
| PB3 | LED: szybkie miganie (co 25 ms) = praca, wolniejsze (co 100 ms) = błąd |
| PB4 | TIM3_CH1, PWM 18 kHz (komenda SET_PWM, kanał 1) |
| PB5 | TIM3_CH2, PWM 18 kHz (komenda SET_PWM, kanał 2) |

## Serwa

| Kanał protokołu | Kanał PCA9685 | Serwo | Impuls dla 0–180 |
|---|---|---|---|
| 0 (pan) | 0 | 1201MG | 500–2500 µs |
| 1 (tilt) | 1 | SG5010 | 1000–1600 µs |

Po starcie oba serwa ustawiają się na 90. Konfiguracja impulsów jest w `main()` w [src/main.cpp](src/main.cpp).

## Budowanie i wgrywanie

1. Otwórz folder `embodied_agent/Stm32` w VS Code jako osobny folder roboczy
2. Na pytanie rozszerzenia ST o skonfigurowanie projektu CMake odpowiedz **Yes**
   (zainstaluje CMake, Ninja i GNU Tools for STM32)
3. Wybierz preset `Debug` lub `Release` i zbuduj (`F7` / *CMake: Build*)
4. Wgranie i debug: **Run and Debug → STM32Cube: Launch ST-Link GDB Server** (`F5`)

Z linii poleceń (z `arm-none-eabi-gcc` w `PATH`):

```bash
cmake --preset Debug
cmake --build --preset Debug    # wynik: build/Debug/Stm32.elf (+ .hex, .bin)
```

Nowy plik `.c` trzeba dopisać do `target_sources` w [CMakeLists.txt](CMakeLists.txt).

## Struktura

```
Stm32/
├── src/                 # kod aplikacji
│   ├── main.cpp         # inicjalizacja, pętla główna, obsługa komend
│   ├── uart_comm.c      # UART: odbiór ramek, kolejka, odpowiedzi
│   ├── servo_control.c  # serwa: kąt -> impuls, zapamiętany kąt
│   ├── pca9685.c        # sterownik PCA9685
│   ├── pwm_driver.c     # TIM3 PWM
│   ├── system_config.c  # zegar, I2C, GPIO
│   └── it.c             # SysTick
├── include/             # nagłówki do src/
├── Core/                # startup, system_stm32f0xx.c, stm32f0xx_hal_conf.h
├── Drivers/             # HAL i CMSIS
├── cmake/               # toolchain i biblioteka HAL dla CMake
├── docs/PCA9685.md      # notatki o sterowniku PCA9685
└── STM32F042K6TX_FLASH.ld
```

## Test z Raspberry Pi

```bash
cd embodied_agent/RPi
source venv/bin/activate
python tools/uart_stress.py -n 100
```

## Rozwiązywanie problemów

- **LED PB3 miga wolniej (co 100 ms):** firmware stoi w `System_ErrorHandler`, najczęściej
  dlatego, że PCA9685 nie odpowiada na I2C. Sprawdź PF0/PF1 i zasilanie modułu.
- **LED PB3 nie miga:** program nie wystartował albo zawiesił się na inicjalizacji (np. UART).
- **Timeout po stronie RPi:** sprawdź skrzyżowanie TX/RX, wspólną masę i włączony port szeregowy
  w `raspi-config`. Ramki ze złą sumą kontrolną STM32 odrzuca bez odpowiedzi.
- **Serwo drży lub nie rusza się:** osobne zasilanie V+ dla serw w PCA9685, patrz [docs/PCA9685.md](docs/PCA9685.md).
