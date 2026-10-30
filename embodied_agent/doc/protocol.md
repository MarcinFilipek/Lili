# Protokół RPi ↔ STM32 (UART)

Źródło prawdy dla obu stron: `RPi/src/lili/stm_link/protocol.py` i firmware STM32.

## Warstwa fizyczna

| Parametr | Wartość |
|---|---|
| Port RPi | `/dev/ttyAMA0` |
| Baudrate | 115200, 8N1 |
| Model | RPi = master, wysyła żądanie i czeka na 5 bajtów odpowiedzi (timeout 100 ms) |

## Ramka (5 bajtów, żądanie i odpowiedź)

```
[START=0xAA, CMD, CHANNEL, DATA, CHECKSUM]
CHECKSUM = START ^ CMD ^ CHANNEL ^ DATA
```

Żądania (RPi → STM):

| CMD | DATA w żądaniu | DATA w ACK | Obsługa w RPi |
|---|---|---|---|
| `0x01` SET_SERVO | kąt 0–180 | echo żądanego kąta | tak |
| `0x02` GET_SERVO | `0x00` | ostatnio **zadany** kąt (to nie pomiar, serwa nie mają sprzężenia zwrotnego) | tak |
| `0x03` SET_PWM | wypełnienie % (kanał 1 → PB4, kanał 2 → PB5) | echo | nie |
| `0x04` PING | `0x00` | `0x55` | nie |
| `0x05` RESET | `0x00` | `0x00` (zatrzymuje animację) | nie |

Odpowiedzi (STM → RPi): `0xCC` ACK albo `0xFF` NACK (kanał ≥ 16 albo nieznana komenda).
CHANNEL w odpowiedzi to zawsze echo z żądania.

Uwagi o firmware (`Stm32/src/main.cpp`, `uart_comm.c`):
- Ramka ze złą sumą kontrolną jest po cichu odrzucana, więc RPi dostaje timeout zamiast NACK.
- SET_SERVO na kanałach 2–15 dostaje ACK, ale nic nie rusza.
- Pętla główna ma `HAL_Delay(25)`, więc pojedyncza transakcja może trwać do ok. 25 ms.
- Serwa są sterowane przez PCA9685 (I2C, adres 0x40).

## Kanały

| CHANNEL | Serwo |
|---|---|
| 0 | pan (głowa lewo-prawo) |
| 1 | tilt (oko góra-dół) |

## Mapowanie kątów

Python `[-90°, +90°]` → bajt `degrees + 90` → STM32 ustawia serwo `[0°, 180°]`.
Odpowiedź: bajt `- 90` → stopnie.

## Przykład

SET kanał 0 na bajt 50 (−40°): `AA 01 00 32 99`, odpowiedź: `AA CC 00 32 54`.
