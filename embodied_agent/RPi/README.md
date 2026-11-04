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
python tools/camera_preview.py       # podgląd kamery na monitorze RPi, 'q' kończy
```

Podgląd na żywo przez SSH (przeglądarka na Macu):

```bash
python tools/camera_stream.py                           # RPi, --color dla koloru
ssh -N -L 8000:localhost:8000 uzume@raspberrypi         # Mac, drugi terminal
```

Potem http://localhost:8000. Krzyż (biały w szarości, żółty z --color) wyznacza środek kadru. Kamera może mieć jednego
użytkownika naraz, więc przed `record_session.py` zatrzymaj podgląd (Ctrl+C).

### Nagrywanie sesji (RPi)

```bash
python tools/record_session.py peering kubek_na_stole          # peering ±10°, 3 cykle
python tools/record_session.py peering kubek --amplitude 15 --settle 0.8
python tools/session_info.py                                   # podsumowanie najnowszej sesji
```

Sesje trafiają do `data/sessions/<data>_<etykieta>/`, format w [../doc/v4_design.md](../doc/v4_design.md).

### Kalibracja kamery

```bash
python tools/capture_calib.py                  # RPi: ruszaj tablicą, zbiera 25 ujęć
python tools/calibrate_camera.py --square 25   # zmierzony bok pola w mm -> calib/camera_320x240.json
```

### Kalibracja serw

Kamera skierowana na daleką, szczegółową scenę (min. ok. 3 m), bo ramię 8 cm daje paralaksę.

```bash
python tools/record_session.py sweep pan_kal --axis pan --limit 30 --step 5
python tools/calibrate_servo.py                # -> calib/servo.json [pan]
python tools/record_session.py sweep tilt_kal --axis tilt --limit 75 --step 15
python tools/calibrate_servo.py                # -> calib/servo.json [tilt]
```

### Ręczna translacja (M4)

Głowa na podstawce, pan 0, listwa przyłożona do boku podstawki, prostopadle do osi optycznej.
Program prowadzi komunikatami: po każdym przesunięciu zabierz ręce i naciśnij Enter.

```bash
python tools/record_session.py manual stol_listwa --offsets 0 2 4            # RPi, przesuw w prawo
python tools/record_session.py manual stol_listwa --offsets 0 2 4 --direction lewo
```

### Percepcja

```bash
python tools/view_session.py                   # Mac: najnowsza sesja, postoje 0 i 1, okno
python tools/view_session.py --pair 0 2        # para postojów 0 i 2 (przy manual: 4 cm)
python tools/view_session.py --save wynik.png  # RPi bez monitora: zapis mozaiki do pliku
python tools/bench_perception.py               # czas derotacji, DIS i paralaksy w 320x240 i 160x120
```

## Połączenia RPi 5 ↔ STM32

| RPi | STM32 |
|---|---|
| GPIO 14 TXD (pin 8) | PA10 RX (D0) |
| GPIO 15 RXD (pin 10) | PA9 TX (D1) |
| GND | GND |
