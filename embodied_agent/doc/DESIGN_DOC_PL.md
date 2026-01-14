# Design Document: Embodied Agent
## Bio-inspirowany agent predykcyjny z cyklami snu-czuwania

---

## 1. Przegląd projektu

### 1.1 Cel
Stworzenie embodied agenta opartego na biologicznie inspirowanych mechanizmach uczenia, który:
- Uczy się przewidywać konsekwencje swoich akcji
- Wykorzystuje rytmy oscylacyjne (theta, delta) do zarządzania pamięcią
- Implementuje cykle snu-czuwania do konsolidacji wiedzy
- Rozwija się etapami (developmental approach)

### 1.2 Inspiracje biologiczne
- **Predictive coding**: Mózg ciągle przewiduje przyszłe stany sensoryczne
- **Active inference**: Minimalizacja błędu predykcji przez akcję
- **Sleep consolidation**: Offline replay doświadczeń podczas snu
- **Oscillatory dynamics**: Rytmy theta/delta do gatowania pamięci
- **Developmental learning**: Stopniowe zwiększanie złożoności percepcji

### 1.3 Kluczowa innowacja
W przeciwieństwie do standardowego deep RL, agent:
- Nie maksymalizuje reward, ale minimalizuje prediction error
- Samodzielnie reguluje ekspozycję na środowisko (sleep when overwhelmed)
- Uczy się z własnej perspektywy (first-person embodied experience)
- Używa bio-realistic timing mechanisms (oscillations)

---

## 2. Architektura systemu

### 2.1 Hardware
```
┌─────────────────────────────────────────┐
│  Raspberry Pi 5 (2GB)                   │
│  - Główny procesor                       │
│  - Kamera (percepcja)                    │
│  - Komunikacja SPI z STM32              │
└─────────────────┬───────────────────────┘
                  │ SPI
┌─────────────────▼───────────────────────┐
│  STM32                                   │
│  - Motor control (napęd gąsienicowy)    │
│  - Real-time execution                   │
└──────────────────────────────────────────┘
```

**Katalogi projektu:**
- `RPi/` - Kod Python na Raspberry Pi (percepcja, predykcja, uczenie)
- `Stm32/` - Kod C na STM32 (kontrola motorów, low-level execution)

### 2.2 Software Architecture (RPi)

```
┌───────────────────────────────────────────────────────┐
│                    BioAgent                            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ VisualCortex │  │ Hippocampus  │  │ MotorCortex │ │
│  │              │  │              │  │             │ │
│  │ - Feature    │  │ - Oscillatory│  │ - SPI comm  │ │
│  │   extraction │  │   Memory     │  │ - Action    │ │
│  │ - Multi-     │  │ - STM buffer │  │   execution │ │
│  │   feature    │  │ - Theta/Delta│  │             │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                  │         │
│         └────────┬────────┴──────────┬───────┘         │
│                  │                   │                 │
│         ┌────────▼─────────┐  ┌──────▼──────────┐     │
│         │ PredictiveSystem │  │ SleepWakeController│  │
│         │ - 3D tensors     │  │ - Error monitor │     │
│         │ - Multiple models│  │ - State machine │     │
│         └──────────────────┘  └─────────────────┘     │
└───────────────────────────────────────────────────────┘
```

---

## 3. Kluczowe komponenty

### 3.1 Multi-Feature Predictive System

Agent nie pracuje na surowych pikselach, ale na **multiple feature spaces**:

```python
features = {
    'edges': EdgeDetector(),      # Krawędzie (Sobel)
    'corners': CornerDetector(),  # Punkty charakterystyczne (Harris)
    'motion': OpticalFlow(),      # Ruch w obrazie
    'brightness': Brightness(),   # Ogólna jasność
}
```

**Każda właściwość ma własną macierz predykcji:**
- Stan świata: `edges → akcja → predykcja edges`
- Stan świata: `corners → akcja → predykcja corners`
- etc.

**Format macierzy: 3D tensor**
```
W[i, j, k] = [feature_dim, action_dim, next_feature_dim]

Predykcja:
next_feature[k] = Σ W[i,j,k] * current_feature[i] * action[j]
```

### 3.2 Oscillatory Memory (Theta/Delta rhythms)

**Theta rhythm (4Hz)** - Working memory, gatuje co jest zapisywane:
- **Peak (θ ≈ 1.0)**: Zapisz feature vector
- **Trough (θ ≈ 0.0)**: Zapisz action vector

**Short-term Memory Buffer:**
```python
STM = [
    {'type': 'feature', 'data': [0.2, 0.8, ...], 'theta': 0.95},
    {'type': 'action', 'data': [1, 0, 0, 0], 'theta': 0.05},
    {'type': 'feature', 'data': [0.3, 0.7, ...], 'theta': 0.92},
    ...
]
```

**Ekstrakcja sekwencji:**
Pattern matching: `feature → action → next_feature`

**Delta rhythm (1Hz)** - Long-term memory consolidation:
- Transfer stabilnych sekwencji z theta do delta
- Powolna konsolidacja episodic memory

### 3.3 Sleep-Wake Cycle

**Stan: AWAKE**
```python
1. Perceive: Extract features from camera
2. Predict: Use 3D matrices to predict next features
3. Compare: Compute prediction error
4. Decide: 
   - If error < threshold: Continue, execute action
   - If error > threshold: Enter SLEEP
```

**Stan: SLEEP**
```python
1. Disconnect sensory input (camera OFF)
2. Replay recent experiences from STM buffer
3. Extract sequences: feature → action → next_feature
4. Consolidate: Strengthen 3D tensor associations
   W[i,j,k] += lr * feature[i] * action[j] * next_feature[k]
5. Check convergence:
   - If all predictors converged: WAKE UP
   - Else: Continue dreaming
```

**Kluczowy mechanizm:**
Agent sam decyduje kiedy potrzebuje "odpoczynku" bazując na prediction error jako stress signal.

### 3.4 Developmental Stages

Agent nie startuje z pełnymi capabilities. Progressive unlocking:

**Stage 1: Sensorimotor (Age 0-50)**
- Obraz: Mocno zblurowany (σ=5)
- Actions: Tylko `do_nothing`
- Goal: Nauczyć się baseline predictions

**Stage 2: Early motion (Age 50-150)**
- Obraz: Średnio zblurowany (σ=3)
- Actions: `do_nothing`, `forward_small`
- Features: Tylko `brightness`
- Goal: Podstawy ruchu

**Stage 3: Spatial awareness (Age 150-300)**
- Obraz: Lekko zblurowany (σ=1)
- Actions: `forward`, `rotate_left`, `rotate_right`
- Features: `brightness`, `edges`
- Goal: Nawigacja przestrzenna

**Stage 4: Object permanence (Age 300+)**
- Obraz: Pełna rozdzielczość
- Actions: Full motor control
- Features: All features active
- Goal: Object tracking, memory, planning

---

## 4. Object Permanence Task (Główny eksperyment)

### 4.1 Opis zadania
1. Agent widzi obiekt (np. kolorowa piłka)
2. Obiekt znika za przeszkodą
3. Agent oczekuje że będzie widział piłkę (prediction)
4. Błąd predykcji jest wysoki (piłka zniknęła!)
5. Agent wykonuje akcje aby znaleźć piłkę
6. Cel: Obejść przeszkodę i odnaleźć obiekt

### 4.2 Wymagane capabilities
- **Spatial memory**: Gdzie była piłka ostatnio
- **Planning**: Jak obejść przeszkodę
- **Temporal reasoning**: Obiekt nadal tam jest mimo że go nie widzę
- **Persistence**: Ile długo szukać zanim się poddać

### 4.3 Mechanizm persistence

**Confidence decay:**
```python
confidence = 1.0
while searching:
    error = |prediction - observation|
    confidence *= (1 - error * decay_rate)
    
    if confidence < threshold:
        abandon_search()
```

**Memory trace decay (theta):**
- Theta trace naturalnie maleje z czasem
- Gdy trace zbyt słaby → przestań szukać

**Adaptive abandonment:**
- Agent uczy się jak długo zwykle znajduje obiekty
- Jeśli obecna próba trwa znacznie dłużej → coś jest nie tak

---

## 5. Implementacja

### 5.1 Hierarchia klas (RPi)

```python
# Main agent
class BioAgent:
    def __init__(self):
        self.visual_cortex = VisualCortex()
        self.motor_cortex = MotorCortex()  # SPI communication
        self.hippocampus = OscillatoryMemory()
        self.predictive_system = MultiFeaturePredictiveSystem()
        self.sleep_wake = SleepWakeController()
        self.age = 0
        
    def step(self):
        if self.sleep_wake.state == 'awake':
            return self.awake_step()
        else:
            return self.sleep_step()

# Vision processing
class VisualCortex:
    def extract_features(self, image, age):
        # Progressive blur based on age
        blur_sigma = max(0, 5 - age/100)
        image = gaussian_blur(image, blur_sigma)
        
        features = {}
        for name, extractor in self.extractors.items():
            if self.is_active(name, age):
                features[name] = extractor.process(image)
        return features

# Motor control
class MotorCortex:
    def __init__(self):
        self.spi = SPIConnection()  # To STM32
        
    def execute(self, action):
        command = self.encode_action(action)
        self.spi.send(command)

# Oscillatory memory
class OscillatoryMemory:
    def __init__(self):
        self.predictors = {
            'edges': OscillatoryPredictiveMatrix(...),
            'corners': OscillatoryPredictiveMatrix(...),
        }
        
    def step(self, features, action):
        for name, predictor in self.predictors.items():
            theta = predictor.theta_gate()
            if theta > 0.8:
                predictor.encode_feature(features[name])
            elif theta < 0.2:
                predictor.encode_action(action)

# 3D Tensor predictor
class OscillatoryPredictiveMatrix:
    def __init__(self, feature_dim, action_dim):
        self.W = np.random.randn(feature_dim, action_dim, feature_dim) * 0.01
        self.stm_buffer = []
        self.theta_phase = 0.0
        
    def predict(self, feature, action):
        return np.einsum('ijk,i,j->k', self.W, feature, action)
    
    def consolidate(self):
        sequences = self.extract_sequences()
        for seq in sequences:
            # Hebbian learning
            self.W += lr * outer_product_3d(
                seq['feature_t'],
                seq['action'],
                seq['feature_t+1']
            )
```

### 5.2 Communication Protocol (RPi ↔ STM32)

**SPI Protocol:**
```
Message structure: [CMD][PARAM1][PARAM2][CHECKSUM]

Commands from RPi to STM32:
0x01: MOVE_FORWARD   [speed]
0x02: MOVE_BACKWARD  [speed]
0x03: ROTATE_LEFT    [angle]
0x04: ROTATE_RIGHT   [angle]
0x05: STOP
0xFF: EMERGENCY_STOP

Response from STM32 to RPi:
0xA0: ACK
0xA1: EXECUTING
0xA2: COMPLETED
0xE0: ERROR
```

### 5.3 Main Loop

```python
def main():
    agent = BioAgent()
    
    while True:
        if agent.sleep_wake.state == 'awake':
            # Perceive
            image = agent.camera.read()
            features = agent.visual_cortex.extract_features(image, agent.age)
            
            # Predict
            predictions = agent.predictive_system.predict(features, agent.last_action)
            
            # Error
            error = compute_error(predictions, features)
            
            # Decision
            if error > agent.sleep_wake.sleep_threshold:
                agent.sleep_wake.enter_sleep()
                continue
            
            # Act
            action = agent.select_action(features)
            agent.motor_cortex.execute(action)
            
            # Update oscillatory memory
            agent.hippocampus.step(features, action)
            
            agent.age += 1
            
        else:  # SLEEP
            converged = agent.predictive_system.dream_consolidation()
            if converged:
                agent.sleep_wake.wake_up()
        
        time.sleep(0.01)  # 100 Hz main loop
```

---

## 6. Metryki i ewaluacja

### 6.1 Developmental metrics
- Age at which each feature becomes active
- Sleep frequency over time (should decrease)
- Prediction error over time (should decrease)
- Average sleep duration (should decrease as learning stabilizes)

### 6.2 Task-specific metrics (Object Permanence)
- Success rate: % of trials where object is found
- Search time: How long to find occluded object
- Persistence: How long agent searches before giving up
- Path efficiency: Direct vs actual path length

### 6.3 Neural metrics
- 3D tensor sparsity (how many connections are strong)
- STM buffer utilization
- Theta-delta synchronization quality
- Sequence extraction success rate

---

## 7. Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Hardware setup (RPi + STM32 + camera + motors)
- [ ] Basic SPI communication working
- [ ] Simple motor control (forward, rotate, stop)
- [ ] Camera capture working

### Phase 2: Perception (Weeks 3-4)
- [ ] Feature extractors (edges, corners, brightness)
- [ ] Progressive blur mechanism
- [ ] Multi-feature integration

### Phase 3: Prediction (Weeks 5-6)
- [ ] 3D tensor implementation
- [ ] Prediction pipeline
- [ ] Error computation

### Phase 4: Memory (Weeks 7-8)
- [ ] Theta oscillation implementation
- [ ] STM buffer
- [ ] Sequence extraction
- [ ] Delta consolidation

### Phase 5: Sleep-Wake (Weeks 9-10)
- [ ] State machine
- [ ] Dream replay mechanism
- [ ] Convergence detection
- [ ] Adaptive thresholds

### Phase 6: Development (Weeks 11-12)
- [ ] Staged unlocking
- [ ] Age-based gating
- [ ] Curriculum setup

### Phase 7: Object Permanence (Weeks 13-16)
- [ ] Object detection
- [ ] Occlusion scenarios
- [ ] Search behavior
- [ ] Persistence mechanisms
- [ ] Evaluation and tuning

---

## 8. Wyzwania i rozwiązania

### 8.1 Computational constraints (RPi 2GB)
**Problem:** 3D tensors + oscillations + real-time = heavy
**Rozwiązanie:**
- Start z małymi feature dimensions (32-64)
- Sparse representations gdzie możliwe
- Offline consolidation during sleep (less time pressure)
- Może numpy C backend dla critical paths

### 8.2 Timing synchronization
**Problem:** Theta oscillations vs async sensor/motor
**Rozwiązanie:**
- Free-running theta clock
- Sample at appropriate phases
- Log all events with timestamps for post-analysis

### 8.3 Catastrophic beginning
**Problem:** Random weights → total chaos at start
**Rozwiązanie:**
- Implemented! Sleep-wake cycle handles this
- Developmental stages limit initial exposure
- Progressive blur reduces sensory overload

### 8.4 Hyperparameter tuning
**Problem:** Wiele parametrów (learning rates, thresholds, frequencies)
**Rozwiązanie:**
- Start z bio-realistic values (theta=4Hz, delta=1Hz)
- Log everything, analyze offline
- Może meta-learning later

---

## 9. Dokumentacja i dissemination

### 9.1 Code documentation
- Docstrings (Google style)
- README w każdym katalogu
- Architecture diagrams
- API reference

### 9.2 Research documentation
- Weekly progress logs
- Experimental results (with videos)
- Failure analysis (co nie zadziałało i dlaczego)
- Lessons learned

### 9.3 Potential publications
- Embodied predictive coding with sleep consolidation
- Oscillatory memory for robot learning
- Developmental robotics without reward signals

---

## 10. Długoterminowa wizja

Ten projekt to fundament dla:
- **Genuine embodied intelligence** (not pattern matching)
- **Self-regulating learning systems** (know when to rest)
- **Bio-realistic AI** (timing matters!)
- **Compositional understanding** (not just associations)

Kolejne kroki po object permanence:
- Causal reasoning (A→B, B→C, więc A→C?)
- Tool use (jak niemowlę odkrywa że patyk rozszerza reach)
- Social learning (obserwacja innych agentów)
- Language grounding (słowa jako predictions)

---

## Autorzy i kontakt

**Autor:** Marcin
**Projekt:** embodied_agent
**Status:** In development
**Licencja:** TBD

---

*This is a living document. Updated as the project evolves.*
