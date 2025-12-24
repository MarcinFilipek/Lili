# BioAgent Evolution Plan: Od Modelu Reaktywnego do Predykcyjnego

## 📋 Spis Treści
1. [Stan Obecny](#stan-obecny)
2. [Wizja Docelowa](#wizja-docelowa)
3. [Scenariusze Rozwoju](#scenariusze-rozwoju)
4. [Plan Implementacji](#plan-implementacji)
5. [Metryki Sukcesu](#metryki-sukcesu)

---

## Stan Obecny

### Architektura BioAgent3

Obecna implementacja ([BioAgent3.py](file:///Users/marcinfilipek/Documents/python_workspace/Lili/src/BioAgent3.py)) zawiera:

#### Komponenty
- **32 sensory audio** - analiza pasm częstotliwości (FFT + logarytmiczne skalowanie)
- **Eligibility Traces** (`self.T`) - ślady niedawnej aktywności
- **Hebbian Learning** - macierz wag `W` aktualizowana przez `np.outer(self.T, S)`
- **Adaptacyjne progi** (`self.theta`) - normalizacja sygnału

#### Parametry
```python
n_sensors = 32          # Liczba pasm częstotliwości
decay_rate = 0.2        # Zanik śladów
lr = 0.01              # Learning rate
FS = 16000             # Sample rate
HOP_MS = 80            # Okno czasowe (80ms)
```

#### Ograniczenia
- ✅ **Reaktywny model** - agent reaguje tylko na bieżące bodźce
- ❌ **Brak predykcji** - nie przewiduje przyszłych stanów
- ❌ **Brak pamięci długoterminowej** - wszystkie zmiany są natychmiastowe
- ❌ **Brak planowania** - nie symuluje konsekwencji akcji
- ❌ **Brak mechanizmu uwagi** - nie wie, co jest ważne do zapamiętania

---

## Wizja Docelowa

### Agent z Pamięcią i Planowaniem

> **Cel**: Przekształcić agenta reaktywnego w agenta predykcyjnego, który:
> - Przewiduje przyszłe stany sensoryczne
> - Buduje trwałą reprezentację "modelu świata"
> - Planuje akcje poprzez symulację mentalną
> - Konsoliduje ważne doświadczenia w pamięci długoterminowej

### Inspiracje Biologiczne

| Mechanizm Biologiczny | Implementacja w Agencie |
|----------------------|-------------------------|
| Hipokamp (pamięć epizodyczna) | Bufor zdarzeń z wysokim prediction error |
| Konsolidacja podczas snu | Przenoszenie wzorców: short-term → long-term |
| Komórki miejsca (place cells) | Chunking powtarzających się sekwencji |
| Korteks przedczołowy (planowanie) | Mental playback przez Forward Model |

---

## Scenariusze Rozwoju

### 🎯 Scenariusz 1: Model Świata (Predykcja Sensoryczna)

#### Założenia
Agent uczy się przewidywać następny stan sensorów na podstawie obecnego stanu i akcji.

#### Komponenty do Dodania
1. **Forward Model** - sieć predykcyjna
   ```python
   W_pred: (n_sensors, n_sensors)  # Przewiduje S_t+1 z S_t
   ```

2. **Prediction Error** - metryka zaskoczenia
   ```python
   PE_t = ||S_t+1_actual - S_t+1_predicted||²
   ```

3. **Attention Signal** - sygnał uwagi
   ```python
   if PE_t > threshold:
       save_to_longterm_memory(S_t, S_t+1)
   ```

#### Eksperyment Walidacyjny
- **Setup**: Odtwarzaj powtarzalną sekwencję dźwięków (np. metronomu)
- **Oczekiwanie**: Prediction error powinien spadać z czasem
- **Sukces**: PE < 0.1 po 100 iteracjach dla znanego wzorca

#### Wskaźniki
- `mean_prediction_error` - średni błąd predykcji
- `surprise_events` - liczba zdarzeń z PE > threshold
- `learning_curve` - PE w funkcji czasu

---

### 🧩 Scenariusz 2: Hierarchiczna Pamięć (Chunking)

#### Założenia
Agent grupuje powtarzające się sekwencje w "koncepty" wyższego poziomu.

#### Komponenty do Dodania
1. **Dual-Layer Architecture**
   ```python
   # Warstwa dolna (szybka, 80ms)
   S_low: (n_sensors,)
   
   # Warstwa górna (wolna, 2s)
   S_high: (n_concepts,)
   ```

2. **Temporal Chunking**
   - Wykrywanie powtarzających się sekwencji
   - Kompresja sekwencji do pojedynczego symbolu
   - Hierarchiczna reprezentacja czasu

3. **Sequence Memory**
   ```python
   sequence_buffer = deque(maxlen=25)  # 2s przy 80ms hop
   chunk_detector = detect_repeating_patterns(sequence_buffer)
   ```

#### Eksperyment Walidacyjny
- **Setup**: Odtwarzaj sekwencję A→B→C wielokrotnie
- **Oczekiwanie**: Agent tworzy jeden neuron reprezentujący całą sekwencję
- **Sukces**: Aktywacja neuronu konceptu przy wykryciu początku sekwencji

#### Wskaźniki
- `n_chunks_learned` - liczba wykrytych wzorców
- `compression_ratio` - stosunek długości sekwencji do reprezentacji
- `chunk_stability` - jak często ten sam wzorzec aktywuje ten sam neuron

---

### 🎮 Scenariusz 3: Planowanie przez Wyobraźnię

#### Założenia
Agent symuluje konsekwencje akcji przed ich wykonaniem.

#### Komponenty do Dodania
1. **Mental Playback Engine**
   ```python
   def imagine(state, action, n_steps=5):
       trajectory = []
       s = state
       for _ in range(n_steps):
           s_next = forward_model.predict(s, action)
           reward = reward_model.predict(s_next)
           trajectory.append((s_next, reward))
           s = s_next
       return trajectory
   ```

2. **Action Selection**
   ```python
   best_action = argmax([
       sum(r for _, r in imagine(S_t, a))
       for a in possible_actions
   ])
   ```

3. **Delayed Reward Learning**
   - Eligibility traces rozszerzone w czasie
   - Credit assignment problem
   - TD-learning lub podobny mechanizm

#### Eksperyment Walidacyjny
- **Setup**: Nagroda pojawia się 5s po specyficznej sekwencji akcji
- **Oczekiwanie**: Agent nauczy się wykonywać sekwencję mimo opóźnienia
- **Sukces**: Konsystentne wykonywanie sekwencji prowadzącej do nagrody

#### Wskaźniki
- `planning_depth` - ile kroków w przód agent symuluje
- `action_quality` - jak często wybiera optymalną akcję
- `credit_assignment_accuracy` - czy łączy przyczynę ze skutkiem

---

## Plan Implementacji

### Faza 1: Fundament Predykcyjny (2-3 dni)

#### Krok 1.1: Dual Memory System
Rozdziel pamięć na operacyjną i trwałą.

```python
class BioAgent4:
    def __init__(self):
        # Pamięć operacyjna (szybka)
        self.W_short = np.random.normal(0, 0.2, (n_sensors, n_sensors))
        self.lr_short = 0.01
        
        # Pamięć trwała (wolna)
        self.W_long = np.zeros((n_sensors, n_sensors))
        self.lr_long = 0.0001
        
        # Konsolidacja
        self.consolidation_threshold = 0.5
```

**Plik**: `src/BioAgent4.py`

#### Krok 1.2: Forward Model
Dodaj mechanizm predykcji.

```python
def predict_next(self, S_current):
    """Przewiduje następny stan sensorów"""
    S_predicted = np.tanh(self.W_pred @ S_current)
    return S_predicted

def compute_prediction_error(self, S_actual, S_predicted):
    """Oblicza błąd predykcji (zaskoczenie)"""
    PE = np.mean((S_actual - S_predicted)**2)
    return PE
```

**Plik**: `src/BioAgent4.py`

#### Krok 1.3: Attention Mechanism
Zapisuj tylko zaskakujące zdarzenia.

```python
def process_with_attention(self, frame):
    S_current = self._hear(frame)
    S_predicted = self.predict_next(self.S_prev)
    PE = self.compute_prediction_error(S_current, S_predicted)
    
    # Aktualizuj pamięć operacyjną zawsze
    self.W_short += self.lr_short * np.outer(self.T, S_current)
    
    # Aktualizuj pamięć trwałą tylko przy wysokim PE
    if PE > self.consolidation_threshold:
        self.W_long += self.lr_long * np.outer(self.T, S_current)
        self.surprise_buffer.append((self.S_prev, S_current, PE))
    
    # Aktualizuj model predykcyjny
    self.W_pred += 0.001 * np.outer(S_current - S_predicted, self.S_prev)
    
    self.S_prev = S_current
    return S_current, PE
```

**Plik**: `src/BioAgent4.py`

---

### Faza 2: Hierarchiczna Reprezentacja (3-4 dni)

#### Krok 2.1: Sequence Buffer
Bufor przechowujący ostatnie N stanów.

```python
from collections import deque

class BioAgent5(BioAgent4):
    def __init__(self):
        super().__init__()
        self.sequence_length = 25  # 2s przy 80ms
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.n_concepts = 16
        self.concept_layer = np.zeros(self.n_concepts)
```

**Plik**: `src/BioAgent5.py`

#### Krok 2.2: Pattern Detection
Wykrywanie powtarzających się wzorców.

```python
def detect_chunks(self):
    """Szuka powtarzających się sekwencji w buforze"""
    if len(self.sequence_buffer) < self.sequence_length:
        return None
    
    # Konwertuj bufor na macierz
    sequence_matrix = np.array(self.sequence_buffer)
    
    # Użyj PCA lub autoenkodera do kompresji
    compressed = self.compressor.transform(sequence_matrix)
    
    # Klasteryzacja - czy ta sekwencja jest znana?
    cluster_id = self.cluster_model.predict(compressed)
    
    return cluster_id
```

**Plik**: `src/BioAgent5.py`

#### Krok 2.3: Concept Neurons
Neurony reprezentujące całe sekwencje.

```python
def update_concepts(self, chunk_id):
    """Aktywuje neuron konceptu dla rozpoznanej sekwencji"""
    if chunk_id is not None:
        self.concept_layer[chunk_id] = 1.0
    else:
        self.concept_layer *= 0.9  # Decay
```

**Plik**: `src/BioAgent5.py`

---

### Faza 3: Mental Playback (4-5 dni)

#### Krok 3.1: Imagination Engine
Symulacja przyszłych stanów.

```python
class BioAgent6(BioAgent5):
    def imagine_trajectory(self, initial_state, action, n_steps=5):
        """Symuluje n kroków w przód"""
        trajectory = []
        state = initial_state.copy()
        
        for step in range(n_steps):
            # Przewiduj następny stan
            next_state = self.forward_model(state, action)
            
            # Przewiduj nagrodę
            reward = self.reward_model(next_state)
            
            trajectory.append({
                'state': next_state,
                'reward': reward,
                'step': step
            })
            
            state = next_state
        
        return trajectory
```

**Plik**: `src/BioAgent6.py`

#### Krok 3.2: Action Selection
Wybór akcji na podstawie symulacji.

```python
def plan_action(self, possible_actions):
    """Wybiera najlepszą akcję przez mental playback"""
    action_values = {}
    
    for action in possible_actions:
        trajectory = self.imagine_trajectory(self.S, action)
        total_reward = sum(t['reward'] for t in trajectory)
        action_values[action] = total_reward
    
    best_action = max(action_values, key=action_values.get)
    return best_action, action_values
```

**Plik**: `src/BioAgent6.py`

#### Krok 3.3: Reward Model
Model przewidujący nagrody.

```python
def train_reward_model(self, state, actual_reward):
    """Uczy się przewidywać nagrody"""
    predicted_reward = self.W_reward @ state
    error = actual_reward - predicted_reward
    self.W_reward += 0.001 * error * state
```

**Plik**: `src/BioAgent6.py`

---

## Metryki Sukcesu

### Faza 1: Forward Model

| Metryka | Cel | Metoda Pomiaru |
|---------|-----|----------------|
| Mean Prediction Error | < 0.1 po 100 iteracjach | `np.mean(PE_history[-100:])` |
| Surprise Events | < 5% wszystkich zdarzeń | `len(surprise_buffer) / total_frames` |
| Learning Speed | PE spada o 50% w 50 iteracjach | Nachylenie krzywej uczenia |

### Faza 2: Chunking

| Metryka | Cel | Metoda Pomiaru |
|---------|-----|----------------|
| Chunks Learned | > 10 stabilnych wzorców | Liczba klastrów z >20 przykładami |
| Compression Ratio | > 5:1 | `sequence_length / n_active_concepts` |
| Chunk Stability | > 90% konsystencji | Procent powtórzeń tego samego wzorca |

### Faza 3: Planning

| Metryka | Cel | Metoda Pomiaru |
|---------|-----|----------------|
| Planning Depth | 5-10 kroków | Liczba symulowanych stanów |
| Action Quality | > 80% optymalnych wyborów | Porównanie z ground truth |
| Delayed Reward Learning | Sukces w 5s opóźnieniu | Czy agent wykonuje poprawną sekwencję |

---

## Narzędzia Diagnostyczne

### Wizualizacje do Implementacji

1. **Prediction Error Dashboard**
   ```python
   # Real-time plot
   - PE w czasie
   - Histogram PE
   - Mapa cieplna: które sensory są najtrudniejsze do przewidzenia
   ```

2. **Memory Consolidation Viewer**
   ```python
   # Wizualizacja wag
   - W_short vs W_long (heatmap)
   - Tempo konsolidacji
   - Zdarzenia w surprise_buffer
   ```

3. **Chunk Explorer**
   ```python
   # Analiza wzorców
   - Dendogram klastrów
   - Przykładowe sekwencje dla każdego chunka
   - Częstość występowania
   ```

4. **Mental Playback Tracer**
   ```python
   # Śledzenie symulacji
   - Wizualizacja trajektorii wyobrażonych
   - Porównanie: wyobrażone vs rzeczywiste
   - Wartości akcji
   ```

---

## Eksperymenty Walidacyjne

### Eksperyment 1: Metronomu (Faza 1)
**Cel**: Sprawdzić, czy agent przewiduje regularne dźwięki

**Setup**:
```python
# Odtwarzaj klik co 1s
metronome_signal = generate_metronome(bpm=60, duration=60)
```

**Oczekiwany wynik**:
- PE wysokie przez pierwsze 10 kliknięć
- PE < 0.05 po 20 kliknięciach
- Agent "przewiduje" moment kliknięcia

**Plik**: `experiments/exp1_metronome.py`

---

### Eksperyment 2: Sekwencja ABC (Faza 2)
**Cel**: Sprawdzić chunking

**Setup**:
```python
# Odtwarzaj A(440Hz) → B(880Hz) → C(1320Hz) w pętli
sequence = ['A', 'B', 'C'] * 50
```

**Oczekiwany wynik**:
- Agent tworzy jeden neuron konceptu dla ABC
- Po usłyszeniu A, aktywuje się przewidywanie B i C
- Concept neuron aktywuje się na początku sekwencji

**Plik**: `experiments/exp2_abc_sequence.py`

---

### Eksperyment 3: Opóźniona Nagroda (Faza 3)
**Cel**: Sprawdzić planowanie

**Setup**:
```python
# Agent musi wykonać sekwencję: LEFT → LEFT → RIGHT
# Nagroda pojawia się 5s po RIGHT
delayed_reward_task(sequence=['LEFT', 'LEFT', 'RIGHT'], delay=5.0)
```

**Oczekiwany wynik**:
- Agent nauczy się sekwencji mimo opóźnienia
- Mental playback pokazuje przewidywaną nagrodę
- > 80% poprawnych wykonań po 100 próbach

**Plik**: `experiments/exp3_delayed_reward.py`

---

## Harmonogram

### Tydzień 1: Fundament
- [ ] Implementacja BioAgent4 z dual memory
- [ ] Forward model + prediction error
- [ ] Attention mechanism
- [ ] Eksperyment 1: Metronomu

### Tydzień 2: Hierarchia
- [ ] Implementacja BioAgent5 z sequence buffer
- [ ] Pattern detection (PCA/clustering)
- [ ] Concept neurons
- [ ] Eksperyment 2: Sekwencja ABC

### Tydzień 3: Planowanie
- [ ] Implementacja BioAgent6 z imagination
- [ ] Reward model
- [ ] Action selection
- [ ] Eksperyment 3: Opóźniona nagroda

### Tydzień 4: Integracja i Testy
- [ ] Dashboard wizualizacyjny
- [ ] Testy wydajnościowe
- [ ] Dokumentacja
- [ ] Publikacja wyników

---

## Pytania Otwarte

### Techniczne
1. **Rozmiar sieci**: Czy 32 sensory wystarczą, czy potrzebujemy więcej dla chunking?
2. **Algorytm klasteryzacji**: PCA + K-means, czy może SOM (Self-Organizing Map)?
3. **Akcje**: Jakie akcje agent może wykonywać? (Tylko słuchanie, czy też generowanie dźwięków?)

### Badawcze
1. **Konsolidacja**: Czy robić ją online, czy w trybie offline ("sen")?
2. **Reward signal**: Skąd agent dostaje nagrody? (Zewnętrzne, czy wewnętrzne - np. minimalizacja PE?)
3. **Scaling**: Jak zachowa się system przy 1000+ sensorach?

---

## Referencje i Inspiracje

### Artykuły Naukowe
- **Predictive Coding**: Rao & Ballard (1999) - "Predictive coding in the visual cortex"
- **Eligibility Traces**: Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
- **Chunking**: Graybiel (2008) - "Habits, rituals, and the evaluative brain"
- **Mental Replay**: Foster & Wilson (2006) - "Reverse replay of behavioural sequences"

### Implementacje
- **Reservoir Computing**: Echo State Networks (Jaeger, 2001)
- **Temporal Difference Learning**: TD(λ) algorithm
- **Hierarchical RL**: Options framework (Sutton et al., 1999)

---

## Kontakt i Współpraca

**Autor**: Marcin Filipek  
**Projekt**: BioAgent - Biologicznie Inspirowany Agent Audio  
**Repozytorium**: `/Users/marcinfilipek/Documents/python_workspace/Lili`

---

> **Następny krok**: Wybierz fazę do rozpoczęcia implementacji lub zaproponuj modyfikacje planu.
