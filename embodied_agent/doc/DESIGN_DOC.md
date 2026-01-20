# Bio-Inspired Embodied Agent - Design Document

**Project:** Developmental AI with Oscillatory Memory System  
**Date:** January 2026  
**Version:** 1.0 - Working Prototype  
**Status:** ✅ Core learning loop functional

---

## Table of Contents

1. [Philosophy & Vision](#philosophy--vision)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Learning Dynamics](#learning-dynamics)
5. [Sleep-Wake Cycles](#sleep-wake-cycles)
6. [Current Capabilities](#current-capabilities)
7. [Known Issues & Solutions](#known-issues--solutions)
8. [Future Roadmap](#future-roadmap)
9. [Implementation Notes](#implementation-notes)

---

## Philosophy & Vision

### Bio-Inspired Principles

This project implements a **bio-realistic** embodied AI agent that learns through experience, mimicking biological neural mechanisms:

1. **Oscillatory Memory** - Theta (4Hz) and Gamma (40Hz) brain rhythms gate encoding and consolidation
2. **Frustration-Driven Sleep** - Emotional regulation triggers sleep when prediction errors accumulate
3. **Developmental Progression** - Infant-like learning progressing to adult-level capabilities
4. **Embodied Intelligence** - Learning through sensorimotor interaction, not passive pattern matching

### Core Hypothesis

**Intelligence emerges from the interaction between:**
- Predictive models (world understanding)
- Sensorimotor loops (embodiment)
- Oscillatory timing (biological rhythms)
- Sleep-wake cycles (consolidation)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│          GLOBAL BRAIN RHYTHMS (Threading @ 1ms)         │
│                                                          │
│  Gamma (40 Hz)  - Encoding gate                         │
│  Theta (4 Hz)   - Consolidation rhythm                  │
│  Beta (13 Hz)   - Reserved for attention (future)       │
│  Alpha (8 Hz)   - Reserved for state transitions        │
│  Delta (1 Hz)   - Reserved for deep sleep (future)      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    AGENT (Orchestrator)                  │
│                                                          │
│  - Frustration accumulation & decay                     │
│  - Sleep/wake state decisions                           │
│  - Learning rate control (dual LR)                      │
│  - Perception-action-consolidation loop                 │
└───┬─────────┬──────────┬──────────────┬─────────────────┘
    │         │          │              │
    ▼         ▼          ▼              ▼
┌────────┐ ┌──────────┐ ┌────────────┐ ┌───────────────┐
│  Eye   │ │  Vision  │ │ Hippocampus│ │  Predictive   │
│        │ │  Cortex  │ │            │ │    System     │
│ Camera │ │ Features │ │ STM Buffer │ │ World Model   │
│ RPi5   │ │Brightness│ │Theta/Gamma │ │ 3D Tensors    │
└────────┘ └──────────┘ └────────────┘ └───────────────┘
    │                          │              │
    └──────────────┬───────────┴──────────────┘
                   ▼
            ┌──────────────┐
            │ Motor Cortex │
            │   Actions    │
            │    (STM32)   │
            └──────────────┘
```

---

## Core Components

### 1. Agent (Orchestrator)

**Location:** `agent.py`  
**Role:** Main control loop, decision-making, state management

**Key Responsibilities:**
- Run perception-action loop
- Monitor prediction error
- Accumulate/decay frustration
- Trigger sleep/wake transitions
- Control learning rates
- Coordinate all subsystems

**State Variables:**
```python
frustration: float                 # Accumulates during high error
frustration_threshold: 5.0         # Sleep trigger
error_threshold: 0.1               # Error significance threshold
previous_features: dict            # For prediction
```

**Main Loop:**
1. Perceive (capture image → extract features)
2. Predict (world model forward pass)
3. Compute error (MSE: predicted vs actual)
4. Update frustration (accumulate or decay)
5. Select action
6. Hippocampus encoding (gamma-gated)
7. Theta consolidation (if theta peak)
8. Execute action
9. Check sleep/wake conditions

---

### 2. Brain Rhythms (Timing Control)

**Location:** `brain_rythm.py`  
**Role:** Generate oscillatory signals for timing control

**Implementation:**
- Global threading loop @ 1ms update frequency
- Sinusoidal oscillations: `value = sin(2π × freq × t)`
- Phase accumulation with wraparound

**Active Rhythms:**

| Rhythm | Frequency | Role | Status |
|--------|-----------|------|--------|
| Gamma (γ) | 40 Hz | STM encoding gate | ✅ Active |
| Theta (θ) | 4 Hz | Consolidation rhythm | ✅ Active |
| Beta (β) | 13 Hz | Reserved (attention) | 🔵 Placeholder |
| Alpha (α) | 8 Hz | Reserved (state shift) | 🔵 Placeholder |
| Delta (δ) | 1 Hz | Reserved (deep sleep) | 🔵 Placeholder |

**Control Interface:**
```python
rhythm.start(freq)   # Begin oscillation at frequency
rhythm.stop()        # Freeze oscillation
rhythm.is_started()  # Check if active
rhythm.value         # Current phase [-1, 1]
```

---

### 3. Hippocampus (Memory System)

**Location:** `hippocampus.py`  
**Role:** Oscillatory memory encoding, STM buffer management

**Architecture:**
```
Gamma (40 Hz) → Encoding gate
  ├─ Peak (value > 0.95) → Append [feature, action] to STM
  └─ STM Buffer (max 20 items) → FIFO queue

Theta (4 Hz) → Consolidation signal
  └─ Peak (value > 0.95) → Trigger in Agent
```

**STM Buffer Structure:**
```python
stm_buffers = {
    'brightness': [
        {'type': 'feature', 'data': np.array([...]), 'gamma': 0.98},
        {'type': 'action', 'data': np.array([...]), 'gamma': 0.97},
        {'type': 'feature', 'data': np.array([...]), 'gamma': 0.96},
        ...
    ]
}
```

**Sequence Extraction:**
Pattern matching: `[feature, action, feature]` → training sequence

```python
sequence = {
    'feature_t': brightness at time t,
    'action': action executed,
    'feature_t1': brightness at time t+1
}
```

**Sleep/Wake Control:**
- `go_sleep()` - Stops gamma (no new encoding)
- `wake_up()` - Resumes gamma (encoding active)

---

### 4. Predictive System (World Model)

**Location:** `predictive_system.py`, `predictive_matrix.py`  
**Role:** Learn and predict state transitions

**Architecture:**

Each feature type has its own 3D tensor predictor:
```
PredictiveMatrix: W[i, j, k]
  i: current feature dimension (16 for brightness)
  j: action dimension (5 actions)
  k: next feature dimension (16)
```

**Forward Model (Prediction):**
```python
next_feature[k] = Σᵢ Σⱼ W[i,j,k] × current_feature[i] × action[j]

# Einstein summation notation:
prediction = np.einsum('ijk,i,j->k', W, feature, action)
```

**Learning (Hebbian Update):**
```python
ΔW[i,j,k] = lr × feature_t[i] × action[j] × feature_t+1[k]

# Strengthens 3-way associations
# "When I see X and do Y, I observe Z"
```

**Convergence Detection:**
```python
avg_error < 0.01 → converged = True
```

---

### 5. Vision Cortex (Feature Extraction)

**Location:** `vision_cortex.py`  
**Role:** Extract fixed-size feature vectors from images

**Current Features:**

| Feature | Dimensions | Grid | Description |
|---------|------------|------|-------------|
| Brightness | 16 | 4×4 | Average luminance per cell |
| Edges | 64 | 8×8 | Sobel edge strength (future) |
| Motion | 32 | 4×4 | Optical flow vectors (future) |
| Corners | 36 | 6×6 | Harris corners (future) |

**Brightness Extraction:**
```python
1. Divide grayscale image into 4×4 grid
2. Compute mean brightness per cell
3. Normalize to [0, 1]
4. Flatten to 16-dimensional vector
```

**Developmental Blur (Eye):**
```python
# Infant vision simulation
frame = cv2.pyrDown(cv2.pyrDown(frame))  # Downsample 2×
frame = cv2.pyrUp(cv2.pyrUp(frame))      # Upsample 2×
# → Pixelated blur effect (5-7ms, bio-realistic)
```

---

### 6. Eye (Camera Interface)

**Location:** `eye.py`  
**Hardware:** Raspberry Pi Camera v2/v3  
**Resolution:** 640×480 (downscaled for processing)

**Features:**
- Picamera2 interface
- Grayscale conversion
- Developmental blur (pyrDown/Up pyramid)
- ~10ms capture + processing

---

### 7. Motor Cortex (Action Execution)

**Location:** `motor_cortex.py`  
**Hardware:** STM32 via SPI (future)  
**Current:** Simulation mode

**Action Space:**
```python
actions = [
    [1,0,0,0,0],  # forward
    [0,1,0,0,0],  # backward
    [0,0,1,0,0],  # rotate_left
    [0,0,0,1,0],  # rotate_right
    [0,0,0,0,1],  # do_nothing
]
```

**Current State:** Agent uses `do_nothing` only (static learning)

---

## Learning Dynamics

### Dual Learning Rate System

**Key Innovation:** Different learning rates for awake vs sleep

```python
# AWAKE (gentle online learning)
lr_awake = 0.0005  # or adaptive: error × 0.01

# SLEEP (strong offline consolidation)
lr_sleep = 0.01    # 20× boost!
```

**Rationale:**

| Phase | Learning Rate | Why |
|-------|---------------|-----|
| Awake | Low (0.0005) | New data streaming in, avoid interference |
| Sleep | High (0.01) | Frozen data, safe to make large updates |

### Hebbian Learning (3D Tensor)

**Update Rule:**
```python
for each sequence (feature_t, action, feature_t+1):
    for i in range(feature_dim):
        for j in range(action_dim):
            for k in range(feature_dim):
                W[i,j,k] += lr × feature_t[i] × action[j] × feature_t+1[k]
```

**Biological Inspiration:**
- "Neurons that fire together, wire together"
- Strengthens co-activation patterns
- No backpropagation, purely local learning

### Error Computation

**Mean Squared Error (MSE):**
```python
error = mean((prediction - actual)²)

# Per feature:
error_brightness = mean((pred_brightness - actual_brightness)²)

# Overall:
error_total = mean(all_feature_errors)
```

**Typical Error Ranges:**
- Untrained: 0.20 - 0.30
- Partially trained: 0.05 - 0.15
- Well trained: 0.001 - 0.01
- Overtrained: 0.01+ (rising)

---

## Sleep-Wake Cycles

### Frustration Accumulation

**Mechanism:**
```python
# Every timestep:
if error > error_threshold (0.1) and awake:
    frustration += error           # Accumulate
else:
    frustration -= decay (0.1)      # Decay

# Sleep trigger:
if frustration > threshold (5.0):
    enter_sleep()
```

**Biological Analogy:**
- High error = "I don't understand!" = stress
- Accumulating stress = fatigue
- Threshold reached = sleep need

### Sleep Phase

**Entry:**
```python
hippocampus.go_sleep()
  └─ gamma.stop()  # No new encoding
  └─ theta continues (consolidation)
```

**During Sleep:**
1. Gamma frozen (STM buffer unchanging)
2. Theta peaks trigger consolidation
3. Learning rate boosted (lr = 0.01)
4. Error decreases as model improves
5. Agent monitors error every timestep

**Exit (Wake-up) Conditions:**
```python
# Condition 1: Error convergence
if error < error_threshold (0.1):
    wake_up()

# Condition 2: Timeout (future)
if sleep_duration > max_sleep_steps:
    wake_up()
```

**Wake-up:**
```python
hippocampus.wake_up()
  └─ gamma.start(40)  # Resume encoding
frustration = 0.0      # Reset
```

### Timeline Example

```
Age 0-20:   Awake, error ~0.25, frustration accumulating
Age 21:     frustration > 5.0 → SLEEP
            
Sleep:      Consolidate... error 0.25 → 0.10
            
Age 21:     error < 0.1 → WAKE UP
Age 22-50:  Awake, error ~0.09, stable
```

---

## Current Capabilities

### ✅ Working Features

1. **Static World Learning**
   - Agent learns to predict brightness when stationary
   - Error: 0.25 → 0.001 after one sleep cycle
   - Convergence detection works

2. **Sleep-Wake Cycles**
   - Frustration-based sleep entry
   - Error-based wake-up
   - Frustration reset on wake

3. **Oscillatory Memory**
   - Gamma (40 Hz) gates STM encoding
   - Theta (4 Hz) triggers consolidation
   - Phase-based timing control

4. **Dual Learning Rates**
   - Gentle awake learning (lr=0.0005)
   - Strong sleep learning (lr=0.01)
   - 20× boost during consolidation

5. **Hebbian Learning**
   - 3D tensor update (feature × action × next_feature)
   - No backpropagation
   - Biologically plausible

---

## Known Issues & Solutions

### Issue 1: Overlearning in Static Environment ⚠️

**Symptom:**
```
Error: 0.088 → 0.004 → 0.025 (rises again!)
```

**Cause:**
Agent continues learning from same data (do_nothing only) → weights oscillate

**Solution:**
```python
# Stop learning when error low
if error > 0.01 and len(sequences) > 0:  # ← Add threshold check
    consolidate(sequences, lr)
else:
    # Skip consolidation (model good enough)
```

**Status:** 🟡 Identified, fix ready to implement

---

### Issue 2: Lack of Exploration

**Symptom:**
Agent only does `do_nothing` → limited learning

**Solution (Future):**
```python
# Boredom-driven exploration
if error < 0.01:
    boredom += 0.1
if boredom > threshold:
    action = random_action()  # Explore!
```

**Status:** 🔵 Planned for camera movement phase

---

### Issue 3: No Timeout for Sleep

**Symptom:**
If convergence fails, agent sleeps forever

**Solution:**
```python
sleep_steps = 0
while sleeping:
    sleep_steps += 1
    if sleep_steps > max_sleep_steps:
        wake_up()  # Timeout
```

**Status:** 🔵 Planned enhancement

---

## Future Roadmap

### Phase 1: Camera Head Movement (Next) 🎯

**Goal:** Agent controls camera pan/tilt servos

**Actions:**
```python
[pan_left_5deg, pan_right_5deg, tilt_up_5deg, tilt_down_5deg, do_nothing]
```

**Learning Objective:**
- Predict brightness changes from camera movement
- Understand ego-motion vs object motion
- Foundation for spatial navigation

**Hardware:**
- 2× servos (pan/tilt)
- PWM control from RPi GPIO

---

### Phase 2: Multi-Feature Learning

**Add Features:**
1. **Edges (8×8)** - Age 50+
   - Spatial structure detection
   - Object boundaries
   
2. **Motion (4×4)** - Age 150+
   - Optical flow
   - Ego-motion compensation
   
3. **Corners (6×6)** - Age 250+ (optional)
   - Landmarks
   - Object recognition

**Developmental Blur:**
```python
Age 0-50:   Heavy blur (5× pyrDown/Up)
Age 50-150: Medium blur (2× pyrDown/Up)
Age 150+:   Light blur (1× pyrDown/Up)
```

---

### Phase 3: Locomotion

**Full Robot Movement:**
- Tank treads (STM32 motor control)
- Actions: [forward, backward, rotate_left, rotate_right, stop]
- Navigate simple environments

---

### Phase 4: Object Permanence Task

**The Test:**
1. Agent sees colored ball
2. Ball hidden behind obstacle
3. Agent must navigate to find it

**Required Capabilities:**
- Spatial memory
- Planning (multi-step actions)
- Persistence (don't give up immediately)
- Object tracking

**Success Criteria:**
- Agent relocates hidden object >80% of trials

---

### Phase 5: Advanced Features

**Potential Additions:**
1. **REM Sleep** - Replay for causal reasoning
2. **Delta Rhythm** - Deep sleep, LTM consolidation
3. **Attention (Beta)** - Focus on salient features
4. **Tool Use** - Extend reach with objects
5. **Social Learning** - Observe other agents

---

## Implementation Notes

### Performance Considerations

**Target:** 100 Hz main loop (10ms per iteration)

**Timing Budget:**
```
Camera capture:        ~3ms
Feature extraction:    ~5ms
Prediction:            <1ms
Consolidation:         ~1ms (theta peak only)
Total:                 ~10ms ✓
```

**Optimizations:**
- Pyramid blur (pyrDown/Up) vs Gaussian: 12-15× faster
- Fixed-size feature vectors (no dynamic allocation)
- Einstein summation (optimized tensor ops)
- Gamma @ 40 Hz = every 25ms (not every loop iteration)

---

### Hardware Requirements

**Current Setup:**
- Raspberry Pi 5 (2GB recommended)
- Raspberry Pi Camera v2/v3
- (Future) STM32 for motor control
- (Future) 2× servos for camera pan/tilt

**Software Dependencies:**
```
picamera2
opencv-python (cv2)
numpy
threading (built-in)
```

---

### Code Organization

```
embodied_agent/
├── RPi/                    # Raspberry Pi code
│   ├── agent.py            # Main orchestrator
│   ├── brain_rythm.py      # Oscillation generator
│   ├── hippocampus.py      # Memory system
│   ├── predictive_system.py # World model
│   ├── predictive_matrix.py # 3D tensor learning
│   ├── vision_cortex.py    # Feature extraction
│   ├── eye.py              # Camera interface
│   └── motor_cortex.py     # Action execution
│
└── STM32/                  # Motor control (future)
    └── motor_control.c
```

---

### Key Design Decisions

1. **Why global brain rhythms?**
   - Bio-realistic timing
   - Decoupled from main loop (threading)
   - Multiple subsystems can sync to same rhythms

2. **Why 3D tensors?**
   - Captures (state, action, next_state) relationships
   - Hebbian learning is local (bio-plausible)
   - No backpropagation needed

3. **Why frustration-based sleep?**
   - Self-regulating (no manual sleep schedule)
   - Emotionally grounded (stress → rest)
   - Adapts to task difficulty

4. **Why dual learning rates?**
   - Awake: gentle (avoid interference)
   - Sleep: strong (safe consolidation)
   - Biological precedent (replay during sleep)

5. **Why developmental progression?**
   - Curriculum learning (simple → complex)
   - Bio-realistic (infant → adult)
   - Prevents overwhelming early learning

---

## Theoretical Foundations

### Active Inference Framework

Agent minimizes **prediction error** through:
1. **Perception** - Update beliefs about world
2. **Action** - Change world to match predictions

**Free Energy Minimization:**
```
F = Prediction Error + Complexity

Minimize F by:
- Learning better models (reduce error)
- Acting to create expected outcomes
```

### Predictive Coding

**Hierarchical prediction:**
```
Top-down: Generate predictions
Bottom-up: Compute errors
Update:    Minimize prediction error
```

Our implementation (simplified):
```
Predict: next_brightness = f(current_brightness, action)
Error:   |predicted - observed|²
Learn:   Update f to reduce error
```

---

## References & Inspiration

**Neuroscience:**
- Hippocampal theta rhythms (4-8 Hz) during exploration
- Gamma (30-100 Hz) for feature binding
- Sharp-wave ripples during sleep (memory replay)
- Complementary Learning Systems theory (McClelland et al.)

**AI/ML:**
- Predictive coding networks
- World models (Ha & Schmidhuber)
- Hebbian learning
- Developmental robotics

**Philosophy:**
- Embodied cognition
- Sensorimotor contingency theory
- Enactivism

---

## Conclusion

This project demonstrates a **working bio-inspired learning system** with:

✅ Oscillatory memory (gamma/theta rhythms)  
✅ Frustration-driven sleep-wake cycles  
✅ Hebbian learning (3D tensor updates)  
✅ Dual learning rates (awake/sleep)  
✅ Convergence-based wake-up  

**Next milestone:** Camera head movement → spatial learning

**Long-term vision:** Autonomous embodied agent capable of:
- Exploration
- Object permanence
- Planning
- Tool use
- Social learning

---

**Status:** Foundation complete, ready for expansion 🚀

**Contact:** Marcin (researcher)  
**Date:** January 20, 2026
