# Lili — RPi

Kod agenta na Raspberry Pi 5. Sterowanie serwami przez STM32 (UART) i kamera Picamera2.
Poprzednia wersja (agent v1–v3): tag `rpi-v3-final`.

## Struktura

```
RPi/
├── src/lili/
│   ├── stm_link/      # komunikacja z STM32 (protokół: ../doc/protocol.md)
│   └── camera.py      # Picamera2
├── tools/             # skrypty sprzętowe (stres-test UART, podgląd kamery)
└── tests/             # testy bez sprzętu (fałszywy port szeregowy)
```

## Instalacja na RPi

```bash
sudo apt install -y python3-picamera2 python3-opencv
cd embodied_agent/RPi
python3 -m venv --system-site-packages venv   # picamera2 jest z apt
source venv/bin/activate
pip install -e ".[dev]"
```

UART: `sudo raspi-config` → Interface Options → Serial Port
- login shell przez serial: **No**
- serial port hardware: **Yes**

Uprawnienia: `sudo usermod -a -G dialout,video $USER`

## Instalacja lokalnie (Mac, bez sprzętu)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Użycie

```python
from lili.stm_link import StmLink, CHANNEL_PAN

with StmLink() as link:
    link.set_position(CHANNEL_PAN, 30.0)
    print(link.get_position(CHANNEL_PAN))
```

```bash
python tools/uart_stress.py -n 100   # stres-test UART
python tools/camera_preview.py       # podgląd kamery, 'q' kończy
```

## Połączenia RPi 5 ↔ STM32

| RPi | STM32 |
|---|---|
| GPIO 14 TXD (pin 8) | PA10 RX (D0) |
| GPIO 15 RXD (pin 10) | PA9 TX (D1) |
| GND | GND |
