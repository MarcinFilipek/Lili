# STM32 Embodied Agent - Motor Control via SPI

Projekt STM32 dla mikrokontrolera **Nucleo F042K6** z komunikacją SPI (slave) i sterowaniem PWM. Współpracuje z Raspberry Pi 5 jako master.

## 📋 Spis treści

- [Przegląd](#przegląd)
- [Specyfikacja sprzętowa](#specyfikacja-sprzętowa)
- [Pinout](#pinout)
- [Funkcjonalność](#funkcjonalność)
- [Kompilacja i wgrywanie](#kompilacja-i-wgrywanie)
- [Protokół komunikacji](#protokół-komunikacji)
- [Schemat połączeń](#schemat-połączeń)

## Przegląd

Ten projekt implementuje:
- **SPI Slave** - odbiera komendy z Raspberry Pi 5
- **PWM (2 kanały)** - sterowanie silnikami/LED @ 18 kHz
- **Interrupt-driven SPI** - efektywna komunikacja bez blokowania CPU
- **Dynamiczne sterowanie** - zmiana duty cycle w czasie rzeczywistym

## Specyfikacja sprzętowa

| Parametr | Wartość |
|----------|---------|
| **Mikrokontroler** | STM32F042K6T6 |
| **Płytka** | Nucleo-F042K6 |
| **Taktowanie** | 48 MHz (HSI + PLL) |
| **Framework** | STM32Cube HAL |
| **IDE/Build** | STM32CubeIDE for VS Code (CMake) lub PlatformIO |
| **Częstotliwość PWM** | 18 kHz |
| **Rozdzielczość PWM** | 2666 kroków (0-100%) |

## Pinout

### SPI1 (Slave Mode)

| Pin STM32 | Funkcja | Połączenie RPi 5 | Opis |
|-----------|---------|------------------|------|
| **PA4** | NSS (CS) | GPIO 8 (CE0) | Chip Select |
| **PA5** | SCK | GPIO 11 (SCLK) | Clock |
| **PA6** | MISO | GPIO 9 (MISO) | Master In Slave Out |
| **PA7** | MOSI | GPIO 10 (MOSI) | Master Out Slave In |
| **GND** | Ground | GND | Wspólna masa |

> [!IMPORTANT]
> **Wszystkie 5 połączeń są wymagane** dla poprawnej komunikacji SPI!

### PWM (TIM3)

| Pin STM32 | Kanał | Częstotliwość | Zastosowanie |
|-----------|-------|---------------|--------------|
| **PB4** | TIM3_CH1 | 18 kHz | Silnik 1 / LED 1 |
| **PB5** | TIM3_CH2 | 18 kHz | Silnik 2 / LED 2 |

### GPIO

| Pin STM32 | Funkcja | Opis |
|-----------|---------|------|
| **PB3** | LED | Wskaźnik aktywności (miganie co 300ms) |

## Funkcjonalność

### Komendy SPI

Mikrokontroler nasłuchuje komend z Raspberry Pi:

| Komenda | Akcja | PWM CH1 |
|---------|-------|---------|
| `'A'` (0x41) | Wyłącz | 0% |
| `'B'` (0x42) | Włącz pełną moc | 100% |
| `'0'` - `'9'` | Ustaw moc | 0% - 90% (co 10%) |

**Przykład:**
- Wysłanie `'A'` → PWM CH1 = 0%
- Wysłanie `'B'` → PWM CH1 = 100%
- Wysłanie `'5'` → PWM CH1 = 50%

### Obsługa PWM

Funkcje do sterowania duty cycle:

```cpp
PWM_SetDutyCycle_CH1(50);  // 50% na kanale 1 (PB4)
PWM_SetDutyCycle_CH2(75);  // 75% na kanale 2 (PB5)
```

Zakres: **0-100%**

## Kompilacja i wgrywanie

### STM32CubeIDE for Visual Studio Code (CMake)

Projekt ma konfigurację CMake zgodną z rozszerzeniem ST (`CMakeLists.txt`, `CMakePresets.json`,
`cmake/`). HAL i CMSIS (STM32CubeF0 1.11.5) są w katalogu `Drivers/`, startup, `system_stm32f0xx.c`
i `stm32f0xx_hal_conf.h` w `Core/`, linker script to `STM32F042K6TX_FLASH.ld`.

1. Otwórz folder `embodied_agent/Stm32` w VS Code (jako osobny folder roboczy)
2. Na pytanie rozszerzenia o skonfigurowanie wykrytego projektu CMake odpowiedz **Yes**
   (zainstaluje CMake, Ninja i toolchain GNU Tools for STM32)
3. Wybierz preset `Debug` lub `Release` i zbuduj (`F7` / *CMake: Build*)
4. Debug/wgranie: **Run and Debug → STM32Cube: Launch ST-Link GDB Server** (`F5`)

Z linii poleceń (z `arm-none-eabi-gcc` w `PATH`):

```bash
cmake --preset Debug
cmake --build --preset Debug    # wynik: build/Debug/Stm32.elf (+ .hex, .bin)
```

### PlatformIO

#### Wymagania

- **PlatformIO** (VSCode extension lub CLI)
- **ST-Link** (wbudowany w płytkę Nucleo)
- **USB kabel** (Micro-USB)

#### Kompilacja

```bash
# W katalogu projektu
cd embodied_agent/Stm32

# Kompilacja
pio run

# Kompilacja + wgranie
pio run --target upload

# Monitor szeregowy (opcjonalnie)
pio device monitor
```

#### VSCode

1. Otwórz folder `embodied_agent/Stm32` w VSCode
2. PlatformIO automatycznie wykryje projekt
3. Użyj przycisków w dolnym pasku:
   - ✓ Build
   - → Upload
   - 🔌 Serial Monitor

## Protokół komunikacji

### Konfiguracja SPI

| Parametr | Wartość |
|----------|---------|
| **Mode** | Slave |
| **Clock Polarity** | CPOL = 0 (Low) |
| **Clock Phase** | CPHA = 0 (1st Edge) |
| **Data Size** | 8 bit |
| **Bit Order** | MSB First |
| **NSS** | Hardware (PA4) |

> [!NOTE]
> Raspberry Pi musi być skonfigurowany w **SPI Mode 0** (CPOL=0, CPHA=0)

### Format danych

```
Master (RPi) → Slave (STM32): 1 bajt (komenda)
Slave (STM32) → Master (RPi): opcjonalnie status (zakomentowane)
```

Aktualnie STM32 tylko odbiera dane. Możesz odkomentować w `main.cpp`:

```cpp
// Linie 271-272
uint8_t response = 0xAA;
SPI_SendData(&response, 1);
```

## Schemat połączeń

### Diagram połączeń

![RPi 5 ↔ STM32 Connection Diagram](docs/connection_diagram.png)

### Połączenie RPi ↔ STM32

```
┌─────────────────────────┐         SPI          ┌──────────────────────┐
│   Raspberry Pi 5        │                       │   STM32 Nucleo F042  │
│                         │                       │                      │
│  GPIO 8  (CE0)  ────────┼──────────────────────►│ PA4 (NSS)            │
│  GPIO 11 (SCLK) ────────┼──────────────────────►│ PA5 (SCK)            │
│  GPIO 9  (MISO) ◄───────┼───────────────────────│ PA6 (MISO)           │
│  GPIO 10 (MOSI) ────────┼──────────────────────►│ PA7 (MOSI)           │
│  GND            ────────┼──────────────────────►│ GND                  │
│                         │                       │                      │
│                         │                       │  PB4 (PWM CH1) ──┐   │
│                         │                       │  PB5 (PWM CH2) ──┼─► │ Motor Driver
│                         │                       │  PB3 (LED)       │   │
└─────────────────────────┘                       └──────────────────────┘
```

### Przykładowy układ z silnikami

```
STM32 PB4 (PWM CH1) ──► [Motor Driver] ──► Motor 1
STM32 PB5 (PWM CH2) ──► [Motor Driver] ──► Motor 2
```

> [!WARNING]
> **Nie podłączaj silników bezpośrednio do STM32!** Użyj sterownika silników (np. L298N, TB6612FNG) z odpowiednim zasilaniem.

## Struktura projektu

```
embodied_agent/Stm32/
├── .pio/                  # Build artifacts (gitignored)
├── .vscode/              # VSCode config
├── include/              # Header files
│   └── README           # Include directory info
├── lib/                  # Project libraries
├── src/
│   ├── main.cpp         # Main application
│   └── it.c             # Interrupt handlers
├── test/                # Unit tests
├── .gitignore           # Git ignore rules
├── platformio.ini       # PlatformIO configuration
└── README.md           # This file
```

## Testowanie

### Test z Raspberry Pi

Na Raspberry Pi uruchom:

```bash
cd embodied_agent/RPi
source venv/bin/activate
python src/SpiTest.py
```

Wprowadź komendy:
- `A` → STM32 ustawi PWM CH1 na 0%
- `B` → STM32 ustawi PWM CH1 na 100%
- `5` → STM32 ustawi PWM CH1 na 50%

### Weryfikacja

1. **LED PB3** powinien migać co 300ms (wskaźnik działania)
2. **PWM PB4/PB5** można zmierzyć oscyloskopem lub podłączyć LED z rezystorem
3. **Monitor szeregowy** - możesz dodać `printf` do debugowania

## Rozwiązywanie problemów

### STM32 nie odpowiada na komendy SPI

- ✓ Sprawdź wszystkie 5 połączeń (NSS, SCK, MISO, MOSI, GND)
- ✓ Upewnij się, że RPi używa SPI Mode 0
- ✓ Sprawdź, czy LED PB3 miga (oznacza działający program)
- ✓ Zmierz napięcia - STM32 pracuje na 3.3V

### Błąd kompilacji

```bash
# Wyczyść build i spróbuj ponownie
pio run --target clean
pio run
```

### Nie można wgrać programu

- ✓ Sprawdź połączenie USB
- ✓ Upewnij się, że ST-Link jest rozpoznany
- ✓ Spróbuj innego portu USB

## Dalszy rozwój

Możliwe rozszerzenia:

- [ ] Dwukierunkowa komunikacja SPI (wysyłanie statusu do RPi)
- [ ] Enkodery silników (odczyt prędkości)
- [ ] PID control dla silników
- [ ] Dodatkowe czujniki (I2C, UART)
- [ ] Watchdog timer dla bezpieczeństwa
- [ ] EEPROM do zapisywania konfiguracji

## Licencja

Projekt edukacyjny - użyj według potrzeb.

## Autor

Marcin Filipek - Embodied Agent Project
