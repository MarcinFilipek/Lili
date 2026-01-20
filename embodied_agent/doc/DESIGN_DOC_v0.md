# Design Document: Embodied Agent
## Bio-inspired Predictive Agent with Sleep-Wake Cycles

---

## 1. Project Overview

### 1.1 Objective
Create an embodied agent based on biologically-inspired learning mechanisms that:
- Learns to predict consequences of its actions
- Uses oscillatory rhythms (theta, delta) for memory management
- Implements sleep-wake cycles for knowledge consolidation
- Develops in stages (developmental approach)

### 1.2 Biological Inspiration
- **Predictive coding**: The brain constantly predicts future sensory states
- **Active inference**: Minimizing prediction error through action
- **Sleep consolidation**: Offline replay of experiences during sleep
- **Oscillatory dynamics**: Theta/delta rhythms for memory gating
- **Developmental learning**: Gradual increase in perceptual complexity

### 1.3 Key Innovation
Unlike standard deep RL, the agent:
- Doesn't maximize reward, but minimizes prediction error
- Self-regulates environmental exposure (sleeps when overwhelmed)
- Learns from its own perspective (first-person embodied experience)
- Uses bio-realistic timing mechanisms (oscillations)

---

## 2. System Architecture

### 2.1 Hardware
```
┌─────────────────────────────────────────┐
│  Raspberry Pi 5 (2GB)                   │
│  - Main processor                        │
│  - Camera (perception)                   │
│  - SPI communication with STM32         │
└─────────────────┬───────────────────────┘
                  │ SPI
┌─────────────────▼───────────────────────┐
│  STM32                                   │
│  - Motor control (tank treads)          │
│  - Real-time execution                   │
└──────────────────────────────────────────┘
```

**Project Directories:**
- `RPi/` - Python code on Raspberry Pi (perception, prediction, learning)
- `Stm32/` - C code on STM32 (motor control, low-level execution)

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

## 3. Key Components

### 3.1 Multi-Feature Predictive System

The agent doesn't work with raw pixels, but with **multiple feature spaces**:

```python
features = {
    'edges': EdgeDetector(),      # Edges (Sobel)
    'corners': CornerDetector(),  # Corner features (Harris)
    'motion': OpticalFlow(),      # Motion in image
    'brightness': Brightness(),   # Overall brightness
}
```

**Each feature has its own prediction matrix:**
- World state: `edges → action → predicted edges`
- World state: `corners → action → predicted corners`
- etc.

**Matrix format: 3D tensor**
```
W[i, j, k] = [feature_dim, action_dim, next_feature_dim]

Prediction:
next_feature[k] = Σ W[i,j,k] * current_feature[i] * action[j]
```

### 3.2 Oscillatory Memory (Theta/Delta rhythms)

**Theta rhythm (4Hz)** - Working memory, gates what gets encoded:
- **Peak (θ ≈ 1.0)**: Encode feature vector
- **Trough (θ ≈ 0.0)**: Encode action vector

**Short-term Memory Buffer:**
```python
STM = [
    {'type': 'feature', 'data': [0.2, 0.8, ...], 'theta': 0.95},
    {'type': 'action', 'data': [1, 0, 0, 0], 'theta': 0.05},
    {'type': 'feature', 'data': [0.3, 0.7, ...], 'theta': 0.92},
    ...
]
```

**Sequence extraction:**
Pattern matching: `feature → action → next_feature`

**Delta rhythm (1Hz)** - Long-term memory consolidation:
- Transfer stable sequences from theta to delta
- Slow consolidation of episodic memory

### 3.3 Sleep-Wake Cycle

**State: AWAKE**
```python
1. Perceive: Extract features from camera
2. Predict: Use 3D matrices to predict next features
3. Compare: Compute prediction error
4. Decide: 
   - If error < threshold: Continue, execute action
   - If error > threshold: Enter SLEEP
```

**State: SLEEP**
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

**Key mechanism:**
The agent self-determines when it needs "rest" based on prediction error as a stress signal.

### 3.4 Developmental Stages

The agent doesn't start with full capabilities. Progressive unlocking:

**Stage 1: Sensorimotor (Age 0-50)**
- Image: Heavily blurred (σ=5)
- Actions: Only `do_nothing`
- Goal: Learn baseline predictions

**Stage 2: Early motion (Age 50-150)**
- Image: Moderately blurred (σ=3)
- Actions: `do_nothing`, `forward_small`
- Features: Only `brightness`
- Goal: Basic motion

**Stage 3: Spatial awareness (Age 150-300)**
- Image: Lightly blurred (σ=1)
- Actions: `forward`, `rotate_left`, `rotate_right`
- Features: `brightness`, `edges`
- Goal: Spatial navigation

**Stage 4: Object permanence (Age 300+)**
- Image: Full resolution
- Actions: Full motor control
- Features: All features active
- Goal: Object tracking, memory, planning

---

## 4. Object Permanence Task (Main Experiment)

### 4.1 Task Description
1. Agent sees an object (e.g., colored ball)
2. Object disappears behind obstacle
3. Agent expects to see ball (prediction)
4. Prediction error is high (ball disappeared!)
5. Agent executes actions to find ball
6. Goal: Navigate around obstacle and find object

### 4.2 Required Capabilities
- **Spatial memory**: Where was the ball last seen
- **Planning**: How to navigate around obstacle
- **Temporal reasoning**: Object still exists even when not visible
- **Persistence**: How long to search before giving up

### 4.3 Persistence Mechanism

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
- Theta trace naturally decays over time
- When trace too weak → stop searching

**Adaptive abandonment:**
- Agent learns how long it typically takes to find objects
- If current attempt takes significantly longer → something is wrong

---

## 5. Implementation

### 5.1 Class Hierarchy (RPi)

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

## 6. Metrics and Evaluation

### 6.1 Developmental Metrics
- Age at which each feature becomes active
- Sleep frequency over time (should decrease)
- Prediction error over time (should decrease)
- Average sleep duration (should decrease as learning stabilizes)

### 6.2 Task-Specific Metrics (Object Permanence)
- Success rate: % of trials where object is found
- Search time: How long to find occluded object
- Persistence: How long agent searches before giving up
- Path efficiency: Direct vs actual path length

### 6.3 Neural Metrics
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

## 8. Challenges and Solutions

### 8.1 Computational Constraints (RPi 2GB)
**Problem:** 3D tensors + oscillations + real-time = heavy
**Solution:**
- Start with small feature dimensions (32-64)
- Sparse representations where possible
- Offline consolidation during sleep (less time pressure)
- Possibly numpy C backend for critical paths

### 8.2 Timing Synchronization
**Problem:** Theta oscillations vs async sensor/motor
**Solution:**
- Free-running theta clock
- Sample at appropriate phases
- Log all events with timestamps for post-analysis

### 8.3 Catastrophic Beginning
**Problem:** Random weights → total chaos at start
**Solution:**
- Implemented! Sleep-wake cycle handles this
- Developmental stages limit initial exposure
- Progressive blur reduces sensory overload

### 8.4 Hyperparameter Tuning
**Problem:** Many parameters (learning rates, thresholds, frequencies)
**Solution:**
- Start with bio-realistic values (theta=4Hz, delta=1Hz)
- Log everything, analyze offline
- Possibly meta-learning later

---

## 9. Documentation and Dissemination

### 9.1 Code Documentation
- Docstrings (Google style)
- README in each directory
- Architecture diagrams
- API reference

### 9.2 Research Documentation
- Weekly progress logs
- Experimental results (with videos)
- Failure analysis (what didn't work and why)
- Lessons learned

### 9.3 Potential Publications
- Embodied predictive coding with sleep consolidation
- Oscillatory memory for robot learning
- Developmental robotics without reward signals

---

## 10. Long-Term Vision

This project is a foundation for:
- **Genuine embodied intelligence** (not pattern matching)
- **Self-regulating learning systems** (know when to rest)
- **Bio-realistic AI** (timing matters!)
- **Compositional understanding** (not just associations)

Next steps after object permanence:
- Causal reasoning (A→B, B→C, therefore A→C?)
- Tool use (like infants discovering sticks extend reach)
- Social learning (observing other agents)
- Language grounding (words as predictions)

---

## Authors and Contact

**Author:** Marcin
**Project:** embodied_agent
**Status:** In development
**License:** TBD

---

*This is a living document. Updated as the project evolves.*
