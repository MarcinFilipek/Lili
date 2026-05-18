# Agent v3 — Dokumentacja projektu

## Cel

Agent v3 to bioinspirowany, ucieleśniony agent działający na Raspberry Pi 5.
Eksploruje scenę przez małe ruchy serwomechanizmami (pan/tilt), buduje wewnętrzny
model świata i minimalizuje błąd predykcji — uczy się online, bez etykiet, bez nagród zewnętrznych.

> Nie ma zewnętrznej nagrody. Jedynym sygnałem uczącym jest własna zaskoczoność agenta.

---

## Filozofia — Predictive Coding

Agent zakłada, że świat jest przewidywalny. Przy każdym kroku:

1. Przewiduje jak będzie wyglądać scena **po wykonaniu akcji**
2. Porównuje przewidywanie z tym co faktycznie zobaczył
3. Uczy się zmniejszać tę różnicę (błąd predykcji)

Im dłużej obserwuje scenę, tym dokładniej model opisuje lokalną okolicę.

### Kiedy agent "zna" scenę?

Agent zna scenę gdy jego model predykcyjny potrafi przewidzieć co zobaczy
**po dowolnym ruchu w obrębie kotwicy** — nie tylko następną klatkę, ale całą
geometrię i fotometrię lokalnego otoczenia. Operacyjnie: błąd predykcji jest
**stabilnie niski** przez wiele kroków eksploracji.

### Dlaczego brak rytmów mózgowych?

Rytmy (gamma, theta, delta) pełnią w biologii funkcję synchronizacji neuronów
rozsianych przestrzennie. W układzie cyfrowym synchronizacja jest za darmo —
globalny zegar, wspólna pamięć, deterministyczna kolejność. Sztuczne rytmy
tylko zwalniają agenta bez dodatkowej wartości.

Zastępujemy je **jawną hierarchią czasową**:
- Szybka pętla: percepcja → predykcja → błąd → update
- Wolna pętla: monitorowanie frustracji → decyzja o śnie / skoku

---

## Sprzęt

| Komponent | Opis |
|-----------|------|
| **Raspberry Pi 5** | Komputer główny, uruchamia agenta |
| **Picamera2** | Kamera — 640×480 RGB, ~30 fps |
| **STM32** | Slave UART — steruje 2 serwami, odsyła aktualną pozycję |
| **Serwo pan** | Obrót głowy lewo-prawo `[−90°, +90°]` |
| **Serwo tilt** | Pochylenie oka góra-dół `[−90°, +90°]` |

### Protokół UART (STM32)

```
Frame: [0xAA, CMD, CHANNEL, DATA, XOR_CHECKSUM]
0x01 = SET_POSITION  (data = stopnie 0–180)
0x02 = GET_POSITION  (data = 0x00, odpowiedź = stopnie 0–180)
```

### Mapowanie kątów

```
Python [-90°, +90°] → byte [0, 180] → STM32 ustawia serwo [0°, 180°]
STM32 zwraca [0, 180] → Python odejmuje 90 → [-90°, +90°]
```

---

## Architektura v3 (aktualna)

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
│  [ Hippocampus ]                                         │
│     │ STM: deque(30) par (features, action)              │
│     │ LTM: [] — konsolidowane podczas snu               │
│     │ sekwencje (f_t, action, f_t+1)                    │
│     ▼                                                    │
│  [ PredictiveSystem ]                                    │
│     │ W: tensor [9 × 2 × 9]                             │
│     │ predict() / compute_error() (MSE)                  │
│     │ consolidate() ← delta rule + weight decay          │
│     ▼                                                    │
│  [ _select_action() ]                                    │
│     │ δ ~ N(0, σ=3°), odbicie przy ANCHOR_RADIUS        │
│     ▼                                                    │
│  [ MotorCortex ]                                         │
│     │ delta → estimated_position + delta → clamp        │
│     │ read_position() ← faktyczna pozycja z STM32        │
│     ▼                                                    │
│  [ UARTConnection ] ────────────────────► STM32         │
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
       │          motor.execute(anchor - estimated_position)  ← powrót do kotwicy
       │          frustration = 0
       │          continue
       │
       ├─ [5] ZAPIS DO STM
       │      hippocampus.record(features, prev_action)
       │
       ├─ [6] WYBÓR AKCJI (z zakotwiczeniem)
       │      action = _select_action()
       │      # δ ~ N(0, σ=3°), odbicie gdy wychodzi poza ANCHOR_RADIUS=15°
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

### `Hippocampus`
- `record(features, action)` → dopisuje parę do `deque(maxlen=30)`
- `extract_sequences()` → lista trojek `(f_t, action, f_t+1)` z bufora STM
- `wake_up()` → czyści STM, LTM nienaruszone
- `go_sleep()` → sygnalizuje fazę konsolidacji

### `PredictiveSystem`
- Tensor `W[9, 2, 9]` dla cechy `brightness`
- `predict(features, action)` → `einsum("ijk,i,j->k", W, f, a)`
- `compute_error(predicted, actual)` → MSE
- `consolidate(sequences, lr)` → delta rule + weight decay `0.999`

### `MotorCortex`
- `execute(delta)` → `estimated_position + delta`, clamp `[−90°, +90°]`
- Używa `estimated_position` (nie `actual_position`) jako bazy do delta
- `read_position()` → odczyt faktycznej pozycji z STM32 (tylko monitoring)
- `reset()` → wysyła bezpośrednio centrum `90°` bez polegania na actual_position

---

## Przestrzeń cech i akcji (v3)

| | Opis | Wymiar |
|--|------|--------|
| **Cechy wejściowe** | Jasność siatki 3×3 | 9 |
| **Akcja** | (Δpan, Δtilt) w stopniach | 2 |
| **Tensor predykcji W** | feature × action × feature | 9 × 2 × 9 = **162 wagi** |

---

## Mechanizm snu i frustracji

| Parametr | Wartość | Opis |
|----------|---------|------|
| `error_threshold` | 0.05 | Poniżej — frustracja spada |
| `frustration_gain` | 1.0 | Wzrost frustracji na jednostkę błędu |
| `frustration_decay` | 0.05 | Spadek frustracji gdy błąd mały |
| `frustration_threshold` | 10.0 | Próg wyzwalający sen i konsolidację |
| `consolidation_lr` | 0.01 | Learning rate podczas konsolidacji |

---

## Zakotwiczenie (anchor)

Agent eksploruje lokalnie wokół punktu kotwiczącego. Biologicznie sensowne —
oko nie wędruje po całym polu widzenia losowo, fiksuje się i eksploruje lokalnie.

```python
ANCHOR_RADIUS = 15.0  # stopni

# Gdy nowa pozycja wykracza poza zasięg — odbicie w stronę kotwicy
if distance(estimated_next, anchor) > ANCHOR_RADIUS:
    delta = direction_to_anchor * sigma
```

Po każdym śnie agent wraca do kotwicy przed wznowieniem eksploracji.
W przyszłości: gdy błąd predykcji stabilnie niski w całym zasięgu kotwicy
→ agent przesuwa kotwicę w nowe miejsce i zaczyna poznawać nowe otoczenie.

---

## STM i LTM — rola i różnica

### STM (Short-Term Memory)

- `deque(maxlen=30)` — rolling buffer ostatnich obserwacji
- Czyszczony po każdym śnie
- Służy **wyłącznie** do potwierdzenia znajomości sceny w obrębie kotwicy
- Gdy błąd STM stabilnie niski → agent zna lokalne otoczenie → może skoczyć dalej
- Gdy błąd STM nie spada → frustracja → sen → konsolidacja do LTM

### LTM (Long-Term Memory)

- Rośnie przez całe życie agenta, nigdy nie jest czyszczona
- Skompilowana wiedza ze wszystkich odwiedzonych miejsc
- Reprezentuje **wiedzę transferowalną** — nie "znam to miejsce"
  ale "znam ten typ miejsca"
- Błąd LTM ma **wyższy priorytet** w obliczaniu frustracji niż błąd STM

### Przepływ wiedzy

```
Eksploracja lokalna → STM (tymczasowe)
        │
        ▼ (podczas snu)
       LTM (trwałe, generalizuje)
        │
        ▼ (gdy agent jedzie w nowe miejsce)
Transfer do nowego kontekstu
```

### Priorytet błędu

```python
# STM — lokalna eksploracja, wysoki błąd jest naturalny i oczekiwany
# LTM — skompilowana wiedza, błąd oznacza coś naprawdę nieznanego
frustration += error_stm * 0.3 + error_ltm * 1.0
```

Uzasadnienie: gdy agent wjeżdża w nowe miejsce na gąsienicach, STM nie zna
go jeszcze — to oczekiwane. Ale jeśli LTM też się myli, agent trafił na coś
naprawdę nieznanego i powinien zwolnić eksplorację żeby się nauczyć.

---

## Architektura LTM — opcje implementacji

### Opcja A — Predyktor LTM jako sieć generalizująca (v4)

Osobna sieć MLP trenowana podczas każdego snu na **wszystkich** historycznych
sekwencjach. Nie zapamiętuje konkretnych miejsc — uczy się reguł które działają wszędzie.

```python
# Wejście: [brightness(9), delta_pan, delta_tilt]
# Wyjście: brightness_next(9)
ltm_predictor = MLP(input=11, hidden=64, output=9)

# Podczas snu — uczy się na całej historii
ltm_predictor.train(all_historical_sequences, lr=0.001)
```

Przykład: agent widział 50 różnych miejsc, w każdym "mały ruch w prawo przesuwa
jasną plamę lekko w lewo". LTM uczy się tej reguły jako generalnej zasady geometrii
kamery. Gdy agent wjeżdża w nowe miejsce — LTM zna zasadę ruchu kamery → niski błąd.

**Zalety:** prosta implementacja, generalizuje naturalnie, błąd predykcji
jest gotowym sygnałem podobieństwa.

**Wada:** uśrednia reguły — może nie rozróżniać między różnymi typami miejsc.

---

### Opcja B — Episodic Memory z kluczami (v5)

Inspirowana biologicznym hipokampem. LTM jako baza epizodów:

```python
episode = {
    "key":       brightness_vector,  # co widziałem
    "sequences": [...]               # czego się nauczyłem
}
```

Gdy agent jest w nowym miejscu, szuka **najbliższego epizodu** i używa jego
sekwencji jako punkt startowy — startuje nie od zera ale od wiedzy z podobnego miejsca.

```python
class EpisodicLTM:
    def __init__(self, similarity_threshold=0.85):
        self.episodes = []

    def store(self, brightness, sequences):
        for episode in self.episodes:
            sim = cosine_similarity(brightness, episode["key"])
            if sim > self.similarity_threshold:
                episode["sequences"].extend(sequences)  # wzbogać istniejący
                return
        self.episodes.append({"key": brightness, "sequences": sequences})

    def retrieve(self, brightness):
        return max(self.episodes,
                   key=lambda ep: cosine_similarity(brightness, ep["key"]))
```

To jest podobne do **few-shot learning** — agent potrzebuje tylko kilku obserwacji
w nowym miejscu bo ma dobry punkt startowy z podobnego epizodu.

**Zalety:** agent rozróżnia typy miejsc, transfer precyzyjny, biologicznie
najbardziej wierny hipokampowi.

**Wada:** wymaga mechanizmu łączenia podobnych epizodów żeby baza nie rosła bez końca.

---

### Opcja C — Contrastive Learning (przyszłość)

Agent uczy się przestrzeni embeddingów gdzie podobne sceny są blisko siebie,
a różne sceny daleko. Podobne do CLIP — ale zamiast obrazu i tekstu, porównujemy
sceny między sobą.

Pary treningowe generowane automatycznie przez eksplorację:
- **Pozytywne** (podobne): dwie klatki z tej samej lub bardzo bliskiej pozycji
- **Negatywne** (różne): klatki z odległych pozycji (>30°) lub różnych sesji

```python
def contrastive_loss(emb_a, emb_b, label, margin=1.0):
    distance = norm(emb_a - emb_b)
    if label == "similar":
        return distance ** 2                    # przyciągaj
    else:
        return max(0, margin - distance) ** 2   # odpychaj
```

Po treningu embeddingi grupują typy miejsc:
```
ściana biała    → [0.8, 0.1, 0.3, ...]
ściana szara    → [0.7, 0.2, 0.3, ...]  ← blisko białej
ciemny korytarz → [0.1, 0.9, 0.1, ...]  ← daleko
okno            → [0.9, 0.1, 0.8, ...]  ← blisko białej ale inne
```

**Zalety:** najbardziej eleganckie, embeddingi są bogatą reprezentacją,
skaluje się do większych wektorów cech (edges, corners, motion).

**Wada:** najtrudniejsze do implementacji online, wymaga dużych batchy,
niestabilne przy małych danych.

---

### Porównanie opcji LTM

| | Opcja A | Opcja B | Opcja C |
|--|---------|---------|---------|
| **Złożoność impl.** | niska | średnia | wysoka |
| **Generalizacja** | uśredniona | per typ miejsca | strukturalna |
| **Rozróżnia typy miejsc** | słabo | tak | tak |
| **Skaluje się** | dobrze | średnio | bardzo dobrze |
| **Online learning** | łatwe | średnie | trudne |
| **Wersja docelowa** | v4 | v5 | przyszłość |

---

## Decyzje architektoniczne — plan v4

### 1. Dwie oddzielne macierze predykcji

Obecny tensor `W[9, 2, 9]` uczy się jednej uśrednionej reguły bez kontekstu
pozycji — nie wie gdzie jest kamera. W v4 rozdzielamy na dwie macierze:

**Macierz ruchu:**
```
[brightness(9), delta_pan, delta_tilt] → brightness_next(9)
```
"Jak ruch transformuje obraz" — wiedza o dynamice, niezależna od miejsca.

**Macierz miejsca:**
```
[brightness(9), pan, tilt] → brightness_next(9)
```
"Czego spodziewam się w tej pozycji" — wiedza przestrzenna, kontekst lokalizacji.

Uzasadnienie separacji: to dwa różne rodzaje wiedzy. Model ruchu uczy się
transformacji — jak akcja deformuje obraz. Model miejsca uczy się kontekstu —
co jest charakterystyczne dla tej lokalizacji. Biologicznie odpowiada to
separacji "where pathway" i "what pathway" w korze wzrokowej.

### 2. LTM jako osobny predyktor (Opcja A)

STM i LTM używają osobnych sieci z różnymi learning rate:

```python
stm_predictor = MLP(input=11, hidden=32, output=9)   # szybkie, lokalne
ltm_predictor = MLP(input=11, hidden=64, output=9)   # wolne, generalizuje
```

STM uczy się online przy każdym kroku. LTM uczy się podczas snu na całej historii.

### 3. Skalowalność — agent na gąsienicach

Agent będzie się poruszał w przestrzeni `[x, y, orientacja, pan, tilt]`.
Globalna mapa nie skaluje się — agent działa **lokalnie**:

- Nie buduje globalnej mapy świata
- Uczy się **kategorii miejsc** a nie konkretnych lokalizacji
- Wiedza z LTM transferowalna — agent który zna "jak wygląda ściana"
  rozpozna ją w nowym miejscu bez wcześniejszej wizyty

---

## Czego agent nie robi (świadome ograniczenia v3)

- ❌ Pozycja absolutna nie jest wejściem do tensora — model nie wie gdzie jest kamera
- ❌ LTM istnieje jako lista ale nie jest używana jako osobny predyktor
- ❌ Brak separacji model ruchu / model miejsca
- ❌ Brak różnicowania priorytetu błędu STM vs LTM
- ❌ Brak zakotwiczenia — agent dryfuje po całej scenie
- ❌ Tylko brightness 3×3 — brak edges, corners, motion

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
picamera2        >= 0.3.12
opencv-python    >= 4.8.0
numpy            >= 1.24.0
pyserial         >= 3.5
```

---

## Roadmap

**v3 (aktualna)**
- [x] Działający agent z kamerą i UART
- [x] STM + LTM w hippocampusie
- [x] Delta rule + weight decay
- [x] Frustracja i sen
- [ ] Zakotwiczenie (anchor)
- [ ] Logowanie błędu do CSV

**v4**
- [ ] Dwie macierze predykcji: model ruchu + model miejsca
- [ ] LTM jako osobny predyktor (Opcja A)
- [ ] Priorytet błędu LTM > STM w frustracji
- [ ] Zakotwiczenie z powrotem do anchor po śnie

**v5**
- [ ] Episodic Memory z kluczami (Opcja B)
- [ ] Agent mobilny na gąsienicach
- [ ] Transfer wiedzy między miejscami

**Przyszłość**
- [ ] Contrastive Learning / embeddingi (Opcja C)
- [ ] Rozszerzenie VisionCortex: edges, corners, optical flow
- [ ] Active inference — agent wybiera akcje które minimalizują błąd predykcji
