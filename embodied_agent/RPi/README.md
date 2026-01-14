# Raspberry Pi Embodied Agent Project

Projekt zawierający kod do komunikacji SPI oraz obsługi kamery na Raspberry Pi 5.

## Struktura projektu

```
embodied_agent/RPi/
├── venv/                 # Wirtualne środowisko Python (nie w repo)
├── src/                  # Kod źródłowy
│   ├── SpiTest.py       # Test komunikacji SPI
│   └── CameraTest.py    # Test kamery Picamera2
├── tests/               # Testy jednostkowe
├── config/              # Pliki konfiguracyjne
├── requirements.txt     # Zależności Python
└── README.md           # Ten plik
```

## Instalacja

### 1. Aktywacja wirtualnego środowiska

```bash
cd embodied_agent/RPi
source venv/bin/activate
```

### 2. Instalacja zależności

```bash
pip install -r requirements.txt
```

**Uwaga:** Na Raspberry Pi może być konieczne zainstalowanie dodatkowych pakietów systemowych:

```bash
# Dla picamera2
sudo apt update
sudo apt install -y python3-picamera2

# Dla OpenCV
sudo apt install -y python3-opencv libopencv-dev
```

## Użycie

### SPI Test

Test komunikacji SPI - wysyłanie wartości 'A' lub 'B':

```bash
python src/SpiTest.py
```

**Wymagane połączenia (Raspberry Pi 5):**
- MOSI → GPIO 10 (Pin 19)
- MISO → GPIO 9 (Pin 21)
- SCLK → GPIO 11 (Pin 23)
- CE0 → GPIO 8 (Pin 24)
- GND → Ground

### Camera Test

Test kamery Picamera2 z podglądem OpenCV:

```bash
python src/CameraTest.py
```

Naciśnij `q`, aby zakończyć podgląd.

## Konfiguracja SPI na Raspberry Pi 5

Upewnij się, że SPI jest włączony:

```bash
sudo raspi-config
# Interface Options → SPI → Enable
```

Lub edytuj `/boot/config.txt`:

```
dtparam=spi=on
```

Następnie zrestartuj:

```bash
sudo reboot
```

## Deaktywacja środowiska

```bash
deactivate
```

## Wymagania sprzętowe

- Raspberry Pi 5 (lub nowszy)
- Picamera2 kompatybilna kamera
- Urządzenie SPI slave (opcjonalne dla testów)

## Notatki

- **SPI bez slave:** Jeśli nie ma podłączonego urządzenia slave, `SpiTest.py` będzie odbierać wartości z pływającej linii MISO (zazwyczaj 0x00 lub 0xFF)
- **Uprawnienia:** Może być wymagane dodanie użytkownika do grupy `spi` i `video`:
  ```bash
  sudo usermod -a -G spi,video $USER
  ```
