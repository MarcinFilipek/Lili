# Agent v3 — Dokumentacja projektu

## Cel

Agent v3 to bioinspirowany, ucieleśniony agent działający na Raspberry Pi 5.  
Jego zadaniem jest **zakotwiczenie się w jednej pozycji** i eksploracja sceny przez wykonywanie
małych, delikatnych ruchów serwami (pan/tilt). Agent buduje wewnętrzny model świata i stara się
minimalizować błąd predykcji — uczy się online, bez etykiet, bez nagród zewnętrznych.

---

## Filozofia — Predictive Coding

Agent zakłada, że świat jest przewidywalny. Przy każdym kroku:

1. Przewiduje jak będzie wyglądać scena **po wykonaniu akcji**
2. Porównuje przewidywanie z tym co faktycznie zobaczył
3. Uczy się zmniejszać tę różnicę (błąd predykcji)

Im dłużej obserwuje scenę, tym dokładniej model opisuje lokalną okolicę.

> Nie ma zewnętrznej nagrody. Jedynym sygnałem uczącym jest własna zaskoczoność agenta.

---

## Sprzęt

| Komponent | Opis |
|-----------|------|
| **Raspberry Pi 5** | Komputer główny, uruchamia agenta |
| **Picamera2** | Kamera — 640×480 RGB, ~30 fps |
| **STM32** | Slave SPI — steruje 2 serwami, odsyła aktualną pozycję |
| **Serwo pan** | Obrót głowy lewo-prawo `[−90°, +90°]` |
| **Serwo tilt** | Pochylenie oka góra-dół `[−90°, +90°]` |

### Pinout SPI (Raspberry Pi 5)

```
MOSI → GPIO 10 (Pin 19)
MISO → GPIO 9  (Pin 21)
SCLK → GPIO 11 (Pin 23)
CE0  → GPIO 8  (Pin 24)
GND  → Ground
```

---

## Architektura

```
┌──────────────────────────────────────────────────────────┐
│                        AgentV3                           │
│                                                          │
│  Picamera2                                               │
│     │ 640×480 RGB                                        │
│     ▼                                                    │
│  [ Eye ]                                                 │
│     │ szarość + blur (pyrDown×2 + pyrUp×2)               │
│     ▼                                                    │
│  [ VisionCortex ]                                        │
│     │ brightness: np.array(9,)  ← siatka 3×3            │
│     ▼                                                    │
│  [ Hippocampus ] ←──────────────────────────┐           │
│     │ STM: deque(30) par (features, action)  │           │
│     │ LTM: []                                │           │
│     │ sekwencje (f_t, action, f_t+1)         │           │
│     ▼                                        │           │
│  [ PredictiveSystem ]                        │           │
│     │ W: tensor [9 × 2 × 9]                 │           │
│     │ predict() / compute_error() (MSE)      │           │
│     │ consolidate() ← Hebbian learning        │           │
│     │ błąd MSE                               │           │
│     ▼                                        │           │
│  [ _select_action() ] ──── action ───────────┘           │
│     │ (Δpan, Δtilt) w stopniach                          │
│     ▼                                                    │
│  [ MotorCortex ]                                         │
│     │ delta → pozycja absolutna → clamp [−90°, +90°]    │
│     │ read_position() ← faktyczna pozycja z STM32        │
│     ▼                                                    │
│  [ SPIConnection ] ──── protokół 3-fazowy ──► STM32     │
└──────────────────────────────────────────────────────────┘
```

---

## Pełna pętla życia (`run()`)

```
START
  │
  ├─ motor.reset()          → serwa do pozycji (0°, 0°)
  ├─ hippocampus.wake_up()  → STM wyczyszczony
  │
  └─ PĘTLA (while True):
       │
       ├─ [1] PROPRIOCEPCJA
       │      actual_pos = motor.read_position()   ← STM32
       │
       ├─ [2] PERCEPCJA
       │      frame    = eye.capture()
       │      features = vision_cortex.extract(frame)
       │      # → {"brightness": array(9,)}
       │
       ├─ [3] PREDYKCJA
       │      predicted = predictive_system.predict(prev_features, prev_action)
       │      error     = MSE(predicted, features)
       │
       ├─ [4] FRUSTRACJA / SEN
       │      jeśli error > error_threshold:
       │          frustration += error * frustration_gain
       │      w przeciwnym razie:
       │          frustration = max(0, frustration - frustration_decay)
       │
       │      jeśli frustration > frustration_threshold:
       │          hippocampus.go_sleep()
       │          sequences = hippocampus.extract_sequences()
       │          predictive_system.consolidate(sequences, lr=0.01)
       │          hippocampus.wake_up()   ← czyści STM
       │          frustration = 0
       │          continue
       │
       ├─ [5] ZAPIS DO STM
       │      hippocampus.record(features, prev_action)
       │
       ├─ [6] WYBÓR AKCJI
       │      action = _select_action()
       │      # δ ~ N(0, σ=3°) dla pan i tilt
       │
       ├─ [7] WYKONANIE AKCJI
       │      motor.execute(action)
       │
       └─ prev_features = features
          prev_action   = action
```

---

## Moduły — odpowiedzialności

### `Eye`
- Wrapper na Picamera2 (640×480, RGB888)
- `capture(gray, blur)` → szarość + rozmycie piramidalne (`pyrDown×2 + pyrUp×2`)

### `VisionCortex`
- `extract_features(frame)` → `{"brightness": array(9,)}`
- Siatka 3×3: każda komórka = `mean(piksele) / 255.0` → zakres `[0, 1]`
- Celowo mała reprezentacja — agent rozumie strukturę sceny, nie zapamiętuje pikseli

### `Hippocampus`
- `record(features, action)` → dopisuje parę do `deque(maxlen=30)`
- `extract_sequences()` → lista trojek `(f_t, action, f_t+1)` z bufora STM
- `wake_up()` → czyści STM, gotowy do nowej eksploracji
- `go_sleep()` → sygnalizuje fazę konsolidacji

### `PredictiveSystem`
- Tensor `W[9, 2, 9]` dla cechy `brightness`
- `predict(features, action)` → `einsum("ijk,i,j->k", W, f, a)`
- `compute_error(predicted, actual)` → MSE
- `consolidate(sequences, lr)` → `W += lr * einsum("i,j,k->ijk", f_t, a, f_t1)`

### `MotorCortex`
- `execute(delta)` → `actual_pos + delta`, clamp `[−90°, +90°]`, wysyła przez SPI
- `read_position()` → odczyt **faktycznej** pozycji z STM32 (zamknięta pętla)
- `reset()` → wraca do `(0°, 0°)`

### `SPIConnection`
- Protokół 3-fazowy: request → 30 ms → dummy → 30 ms → odczyt odpowiedzi
- Frame: `[0xAA, CMD, CHANNEL, DATA, XOR_CHECKSUM]`
- `0x01` = SET_POSITION, `0x02` = GET_POSITION

---

## Przestrzeń cech i akcji

| | Opis | Wymiar |
|--|------|--------|
| **Cechy wejściowe** | Jasność siatki 3×3 | 9 |
| **Akcja** | (Δpan, Δtilt) w stopniach | 2 |
| **Tensor predykcji W** | feature × action × feature | 9 × 2 × 9 = **162 wag** |

---

## Mechanizm snu i frustracji

| Parametr | Wartość | Opis |
|----------|---------|------|
| `error_threshold` | 0.05 | Poniżej — frustracja spada |
| `frustration_gain` | 1.0 | Wzrost frustracji na jednostkę błędu |
| `frustration_decay` | 0.05 | Spadek frustracji gdy błąd mały |
| `frustration_threshold` | 10.0 | Próg wyzwalający sen i konsolidację |
| `consolidation_lr` | 0.01 | Learning rate podczas konsolidacji |

```
błąd > threshold  →  frustration += błąd × gain
błąd ≤ threshold  →  frustration  = max(0, f − decay)

frustration > frustration_threshold:
    go_sleep()
    extract_sequences() ze STM
    consolidate(lr=0.01)       ← Hebb
    wake_up()                  ← STM wyczyszczony
    frustration = 0
```

---

## Wybór akcji — eksploracja

Agent eksploruje **losowo, ale delikatnie** — małe ruchy wokół zakotwiczonej pozycji:

```python
def _select_action(self) -> np.ndarray:
    sigma = 3.0  # stopni
    return np.random.normal(0.0, sigma, size=2)
```

> **Przyszłość:** `_select_action()` może zostać zastąpiony przez gradient minimalizujący
> błąd predykcji w przestrzeni akcji (*active inference / active perception*).

---

## Czego agent **nie** robi (świadome ograniczenia v3)

- ❌ Brak rytmów mózgowych — tracą sens w systemach cyfrowych
- ❌ Brak corners, edges, motion — tylko `brightness 3×3` (na razie)
- ❌ Brak supervised learning, backpropagation, głębokich sieci
- ❌ Brak globalnej nawigacji — agent siedzi w jednym miejscu
- ❌ Brak komunikacji sieciowej w tej wersji

---

## Uruchomienie

```bash
cd embodied_agent/RPi
source venv/bin/activate
cd src/agent_v3
python agent_v3.py
```

**Wymagania:**

```
picamera2  >= 0.3.12
opencv-python >= 4.8.0
numpy >= 1.24.0
spidev >= 3.6
```

---

## Plan implementacji

- [ ] `hippocampus.py` — `record()`, `extract_sequences()`, `go_sleep()`
- [ ] `motor_cortex.py` — `read_position()`, podłączyć `_send_position()` do SPI
- [ ] `agent_v3.py` — pełna pętla `run()` z predykcją, frustracją i snem
- [ ] `predictive_system.py` + `predictive_matrix.py` — dodać do katalogu `agent_v3/`
- [ ] Testy end-to-end z kamerą i serwami
- [ ] Logowanie błędu predykcji i frustracji (CSV + wykresy)
