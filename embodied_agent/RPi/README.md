# Lili — RPi

Kod agenta Lili v4 na Raspberry Pi 5. Projekt: [../doc/v4_design.md](../doc/v4_design.md).
Poprzednia wersja (agent v1–v3): tagi `rpi-v3-final`, `rpi-v3-latest`.

## Struktura

```
RPi/
├── src/lili/
│   ├── hw/            # jedyne miejsce dotykające sprzętu
│   │   ├── stm_link/  # komunikacja z STM32 (protokół: ../doc/protocol.md)
│   │   └── camera.py  # Picamera2
│   ├── body/          # model ciała: geometria, kamera, serwa, kopia eferentna
│   ├── record/        # zapis i odczyt sesji
│   ├── motor/         # programy ruchu (peering, ruch-stop-patrz)
│   └── perception/    # przepływ, derotacja, paralaksa, krawędzie zasłaniania
├── tools/             # skrypty CLI
├── calib/             # wyniki kalibracji
├── data/              # nagrania sesji (poza repo)
└── tests/             # testy bez sprzętu
```

Nagrania z RPi na Maca:

```bash
rsync -av uzume@raspberrypi:~/Documents/PythonWorkspace/AudioAgent/embodied_agent/RPi/data/ data/
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
from lili.hw.stm_link import StmLink, CHANNEL_PAN

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
