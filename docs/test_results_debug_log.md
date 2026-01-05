# Test Results & Debugging Log
## Oscillatory Memory System - Implementation Phase
**Period:** January 3-5, 2026

---

## Table of Contents
1. [Test Session Timeline](#test-session-timeline)
2. [Key Discoveries](#key-discoveries)
3. [Failed Approaches](#failed-approaches)
4. [Successful Solutions](#successful-solutions)
5. [Performance Data](#performance-data)
6. [Debug Traces](#debug-traces)
7. [Parameter Exploration](#parameter-exploration)

---

## Test Session Timeline

### Day 1: January 3, 2026 - Initial Testing

**Goal:** Validate basic compositional learning (A-B-C + X-B-Z → X-B-C)

**Setup:**
```python
Frequencies: A=300, B=1000, C=2000, X=600, Z=2500 Hz
Training: 3x A-B-C, 1x X-B-Z
FFT: 25ms window (400 samples)
Rhythms: Theta 4Hz, Delta 1Hz
```

**Result:** ❌ **FALSE POSITIVE**

```
Step 2:
  vs C: 0.631
  vs Z: 0.628
  Difference: 0.003
```

**Diagnosis:** System "passing" due to exploration noise (0.2), not learning!

**Key Insight:** All vectors ended up in same dominant band (band 2) → Poor frequency resolution.

---

### Day 2: January 4, 2026 - FFT Resolution Investigation

#### Test 2.1: Increase Dimensionality (128 bands)

**Hypothesis:** More bands = better frequency resolution

**Result:** ❌ **Worse performance**

```
B similarity: 0.99 → 0.95 (got worse!)
C-Z similarity: Still ~0.90
```

**Diagnosis:** Curse of dimensionality - vectors too sparse, Hebbian connections weaker.

---

#### Test 2.2: Mel-Scale + Sparse Coding

**Changes:**
- Mel filterbank (64 bands, 100-3000 Hz)
- Top-K sparse coding (k=10)

**Result:** ❌ **Still poor separation**

```
X and Z in same dominant band (band 4)
C-Z similarity: 0.900
```

---

#### Test 2.3: Wide Frequency Separation

**Changes:**
```python
A=300, B=1000, C=2000, X=600, Z=2500 Hz
(Wider spacing)
```

**Result:** ⚠️ **Partial improvement**

Generated vectors activated 13 bands [0-12] (too spread out).

**Issue:** Hebbian query returns blend of all associations, not discrete answer.

---

#### Test 2.4: Winner-Take-All Memory

**Innovation:** Store exact patterns during learning, retrieve closest match (not blend).

**Result:** ⚠️ **Crisp vectors but still C-Z similar**

```
C-Z similarity: 0.900 (unchanged)
```

**Conclusion:** Problem is in representation, not retrieval.

---

#### Test 2.5: Peak Detection Sparse Coding

**Innovation:** Detect local maxima in spectrum instead of global top-K.

**Result:** ⚠️ **Better sparsity but timing issues**

```
Delta timing: length 2,2,2 (should be 3,3,3)
```

Consolidation triggered too early!

---

### ROOT CAUSE FOUND: FFT Window Size

**Analysis:**

```
Current:
  Frame: 400 samples (25ms)
  FFT: n=2048
  Frequency resolution: 16000/2048 = 7.8 Hz/bin

Problem:
  300Hz vs 600Hz difference: 300Hz
  Resolution: 300/7.8 = 38 bins
  After binning to 32 bands (64 bins/band):
    Both in bands 0-1! NO SEPARATION!
```

**Heisenberg Principle:**
```
Δf · Δt ≥ 1

Short window (25ms): Δf = 40 Hz (poor)
Long window (500ms): Δf = 2 Hz (good!)
```

**Solution:** Use FULL echoic buffer (500ms) for FFT in theta sampling!

---

### Day 3: January 5, 2026 - Multi-Process Architecture

#### Test 3.1: Simple Self-Contained Agent

**Goal:** Remove dependency on external rhythm threads.

**Implementation:**
```python
class BioAgent6:
    def __init__(self):
        self.elapsed_time = 0.0
        self.theta_phase = 0.0
        self.delta_phase = 0.0
    
    def process(self, hop):
        # Update time
        self.elapsed_time += 0.025
        
        # Update phases
        self.theta_phase += 2*pi*4.0*0.025
        self.delta_phase += 2*pi*1.0*0.025
        
        # Check peaks
        if sin(theta_phase) > 0.95:
            # Theta processing
```

**Result:** ✅ **Rhythms work but single-process**

---

#### Test 3.2: Multi-Process with manager.dict()

**Goal:** Separate generator and agent processes.

**Architecture:**
```python
# PROCESS 1: Rhythm updater
def rhythm_updater(rhythms_dict):
    rhythms_dict['theta'].update()

# PROCESS 2: Agent
def agent_process(rhythms_dict):
    agent = BioAgent6(rhythms=rhythms_dict)
```

**Result:** ❌ **Rhythms don't synchronize!**

```
Theta triggers: 0
Delta triggers: 0
```

**Diagnosis:** `manager.dict()` doesn't synchronize object methods! Calling `.update()` modifies local copy, not shared object.

---

#### Test 3.3: Rhythms in Agent Process (Threading)

**Key Insight:** Thread must be in SAME process space as agent!

**Solution:**
```python
def agent_process(audio_queue):
    # Create LOCAL rhythms
    RHYTHMS = {"theta": BrainRhythm(0), "delta": BrainRhythm(0)}
    
    # Start thread IN THIS PROCESS
    start_global_rhythms(RHYTHMS)  # Thread shares memory
    
    # Agent uses local rhythms
    agent = BioAgent6(rhythms=RHYTHMS)
```

**Result:** ✅ **WORKS!**

```
Theta triggers: 33
Delta triggers: 8
W norm: 0.307
```

---

### Day 3 Evening: Snapshot Debugging

#### Issue: Snapshots Have Wrong Number of Vectors

**Observation:**
```
A-B-C snapshot: 1 vector (expected 3)
X-B-Z snapshot: 1 vector (expected 3)
```

**Diagnosis:** Snapshot captured AFTER delta cleared working memory!

```python
# WRONG:
agent.process_delta_frequency()  # Clears WM
wm_snapshot = [wm[i] for i in range(7)]  # Already empty!
```

**Solution:**
```python
# Capture BEFORE processing
if delta_peak_detected:
    wm_snapshot = [wm[i].copy() for i in range(7)]  # BEFORE clear

agent.process_delta_frequency()  # Now clear
```

**Result:** ✅ **Snapshots now have 3 vectors!**

---

#### Issue: Using Wrong Snapshots for Analysis

**Problem:** Code assumed last 2 snapshots are A-B-C #3 and X-B-Z.

**Reality:**
```
Snapshot 1 @ 1.00s: 3 vectors  ← A-B-C #1
Snapshot 2 @ 2.25s: 1 vector   ← Spurious
Snapshot 3 @ 3.50s: 2 vectors  ← Incomplete
...
Snapshot 6 @ 7.00s: 3 vectors  ← A-B-C #3
Snapshot 8 @ 9.05s: 3 vectors  ← X-B-Z
```

**Solution:** Filter for snapshots with ≥3 vectors, use last 2 of those.

**Result:** ✅ **Correct sequences analyzed!**

---

### Current Status: Composition Marginally Working

**Latest Test (7:1 training ratio):**

```
======================================================================
COMPOSITIONAL ANALYSIS
======================================================================

Processed 380 hops (9.50s)
Theta triggers: 33
Delta triggers: 8
LTM weight norm: 0.291

✓ LTM learned!

--- Vector Signatures ---
  A (300Hz): bands [1, 2, 3], DOM=1
  B (1000Hz): bands [5, 6, 7], DOM=6
  C (2500Hz): bands [14, 15, 16], DOM=15
  X (600Hz): bands [3, 4, 5], DOM=4
  Z (3500Hz): bands [20, 21, 22], DOM=21

--- B Recognition ---
B similarity: 0.976 ✓

--- Distinctiveness Check ---
  A-C: 0.234 ✓ distinct
  A-X: 0.641 ✗ similar
  C-Z: 0.184 ✓ distinct
  X-Z: 0.211 ✓ distinct

  Average: 0.318 ✓ distinct

--- Evaluation ---
Step 1 vs B: 0.976 ✓

Step 2:
  vs C: 0.580
  vs Z: 0.550
  Difference: 0.030

? Ambiguous (threshold: 0.15)
```

**Interpretation:**
- ✅ B recognition perfect (0.976)
- ✅ Vectors are distinct (avg 0.318)
- ⚠️ Composition leans toward C but not significantly
- ❌ Need >0.15 difference for reliable discrimination

---

## Key Discoveries

### 1. FFT Window Size is Critical

**Finding:** 25ms window gives 7.8 Hz resolution - insufficient for tone discrimination.

**Evidence:**
```
300Hz and 600Hz (300Hz apart):
  25ms window: 38 bins apart → same band after binning
  500ms window: 154 bins apart → distinct bands
```

**Solution:** Use full echoic buffer (500ms) for theta FFT.

**Trade-off:** Lose temporal precision, gain frequency precision.

---

### 2. Multiprocessing + Threading Requires Care

**Finding:** `multiprocessing.Manager().dict()` does NOT synchronize object methods.

**Evidence:**
```python
# This doesn't work:
rhythms_dict = manager.dict()
rhythms_dict['theta'] = BrainRhythm(0)

# In another process:
rhythms_dict['theta'].update()  # Modifies LOCAL copy!
```

**Solution:** Keep rhythm thread in same process as agent.

---

### 3. Snapshot Timing is Delicate

**Finding:** Must capture working memory BEFORE delta clears it.

**Evidence:**
```
[CONSOLIDATE] Sequence length: 3  ← WM has 3 vectors
[AGENT] ⚠ WM already empty!       ← Captured too late
```

**Solution:** Check for delta peak, capture, THEN call process_delta.

---

### 4. Training Ratio Matters More Than Expected

**Finding:** 3:1 ratio insufficient, 7:1 marginally sufficient.

**Evidence:**
```
3:1 ratio: C=0.335, Z=0.806 (Z dominates - recency bias)
7:1 ratio: C=0.580, Z=0.550 (ambiguous but leaning C)
```

**Hypothesis:** Need 10:1 or higher for reliable composition.

---

### 5. Hebbian Decay Eats Learning

**Finding:** 10% decay per update is too aggressive for few-shot learning.

**Math:**
```
With decay = 0.9:
  After 1 update: W = 0.1
  After 2 updates: W = 0.1*0.9 + 0.1 = 0.19
  After 7 updates: W ≈ 0.47 (not 0.7!)
```

**Solution:** Reduce to 5% decay (0.95) or lower.

---

### 6. Context Can Help OR Hurt

**Finding:** Context in `generate()` may reinforce recent associations.

**Evidence:**
```python
# During generation step 2:
context = [X, B]

# If X→Z was learned (from X-B-Z):
context_boost = W @ X + W @ B
# W @ X might point to Z!
```

**Needs testing:** Direct query vs context-based generation.

---

## Failed Approaches

### ❌ Approach 1: Increase Dimensionality (128 bands)

**Rationale:** More bands → better frequency resolution

**Implementation:** 32 → 128 frequency bands

**Result:** Performance DEGRADED
- B similarity: 0.99 → 0.95
- Vectors too sparse
- Hebbian connections weaker

**Lesson:** High-dimensional sparse vectors hurt associative memory.

---

### ❌ Approach 2: Mel Scale + Top-K Sparse

**Rationale:** Perceptually-motivated frequency scale + sparsity

**Implementation:**
- 64 mel-scaled bands (100-3000 Hz)
- Top-10 activation (k=10)

**Result:** X and Z still in same dominant band

**Lesson:** Mel scale doesn't solve fundamental FFT resolution problem.

---

### ❌ Approach 3: manager.dict() for Shared Rhythms

**Rationale:** Share rhythms between processes cleanly

**Implementation:**
```python
manager = mp.Manager()
rhythms_dict = manager.dict()
```

**Result:** Rhythms never trigger (value stays 0)

**Lesson:** Object methods don't synchronize across processes.

---

### ❌ Approach 4: Pause-Based Delta (Without Rhythm)

**Rationale:** Use silence detection instead of delta rhythm

**Implementation:**
```python
if energy < threshold and silence > 500ms:
    consolidate()
```

**Result:** Timing inconsistent, some sequences get only 2 vectors

**Lesson:** Rhythms provide reliable, biology-inspired timing.

---

## Successful Solutions

### ✅ Solution 1: Long-Window FFT in Theta

**Problem:** Poor frequency resolution with 25ms FFT

**Solution:**
```python
def process_theta_frame(self):
    # Use FULL echoic buffer for FFT
    audio = self.echoic_memory.flatten()  # 8000 samples = 500ms
    
    window = np.hamming(len(audio))
    spectrum = np.fft.rfft(audio * window, n=8192)
    
    # Frequency resolution: 16000/8192 = 1.95 Hz!
```

**Result:** Tones clearly separated in different bands

---

### ✅ Solution 2: Rhythms in Agent Process

**Problem:** Multiprocessing breaks rhythm synchronization

**Solution:**
```python
def agent_process(audio_queue):
    # Create LOCAL rhythms IN THIS PROCESS
    RHYTHMS = {"theta": BrainRhythm(0), "delta": BrainRhythm(0)}
    
    # Start rhythm thread IN THIS PROCESS
    t = threading.Thread(target=update_rhythms, args=(RHYTHMS,))
    t.start()
    
    # Agent uses local rhythms (shares memory with thread)
    agent = BioAgent6(rhythms=RHYTHMS)
```

**Result:** Rhythms trigger reliably (33 theta, 8 delta)

---

### ✅ Solution 3: Pre-Delta Snapshot Capture

**Problem:** Snapshots empty because captured after WM cleared

**Solution:**
```python
# Detect peak BEFORE processing
if delta_peak and not processed:
    wm_snapshot = [self.working_memory[i].copy() for i in range(7)]
    
# THEN process (clears WM)
self.process_delta_frequency()

# Save snapshot if non-empty
if len(wm_snapshot) > 0:
    snapshots.append(wm_snapshot)
```

**Result:** Snapshots now have correct 3-vector sequences

---

### ✅ Solution 4: Wider Frequency Spacing

**Problem:** C (2000 Hz) and Z (2500 Hz) too close

**Solution:**
```python
# OLD:
C = 2000 Hz → band 12
Z = 2500 Hz → band 15
Distance: 3 bands

# NEW:
C = 2500 Hz → band 15
Z = 3500 Hz → band 21
Distance: 6 bands
```

**Result:** Distinctiveness improved (C-Z similarity: 0.900 → 0.184)

---

### ✅ Solution 5: Higher Training Ratio

**Problem:** Recency bias (X-B-Z dominates)

**Solution:** Increase A-B-C repetitions from 3x to 7x

**Result:**
```
3:1 ratio: Z dominates (0.806 vs 0.335)
7:1 ratio: C leads (0.580 vs 0.550)
```

Still ambiguous but trending correct direction.

---

### ✅ Solution 6: Slower Delta Rhythm

**Problem:** Multiple delta triggers during long pauses → spurious consolidations

**Solution:** Reduce delta from 1.0 Hz to 0.5 Hz (2-second period)

**Result:** Cleaner consolidations, fewer spurious snapshots

---

## Performance Data

### Timing Measurements

```
Audio Processing:
  Hop duration: 25ms (40 Hz)
  Hops per second: 40
  Processing time: <1ms per hop
  Real-time factor: 40x faster than real-time

Theta Sampling:
  Target frequency: 4.0 Hz
  Measured frequency: 3.5 Hz (33 triggers in 9.5s)
  Period: ~287ms (target 250ms)
  Accuracy: 87%

Delta Consolidation:
  Target frequency: 0.5 Hz
  Measured: 8 triggers in 9.5s = 0.84 Hz
  Period: ~1187ms (target 2000ms)
  Note: Some triggers during pauses (expected)

Total Latency:
  Audio → Echoic: <1ms
  Echoic → Working: ~250ms (theta period)
  Working → LTM: ~2000ms (delta period)
```

### Memory Usage

```
Echoic Memory:
  Size: (20, 400) floats = 32 KB
  
Working Memory:
  Size: (7, 32) floats = 896 bytes

Hebbian LTM:
  W_fast: (32, 32) floats = 4 KB
  W_slow: (32, 32) floats = 4 KB
  W: (32, 32) floats = 4 KB
  Total: 12 KB

Agent Total: ~50 KB (tiny!)
```

### Learning Statistics

```
After 4 sequences (3x A-B-C, 1x X-B-Z):
  
  LTM weight norm: 0.29-0.34
  Non-zero weights: ~800/1024 (78%)
  Max weight: 0.15-0.20
  
  Fast trace norm: 0.18-0.22
  Slow trace norm: 0.11-0.15
  
  Patterns stored: 4-8 unique
  Consolidations: 8-12 total
```

---

## Debug Traces

### Trace 1: Successful Sequence Learning

```
[GEN] A-B-C #1...
[AGENT] Hop 0: theta=-0.99, delta=0.41

Processing theta frame
[AGENT] Theta #1 @ 0.40s: stored vector, DOM=1  ← A

Processing theta frame  
[AGENT] Theta #2 @ 0.68s: stored vector, DOM=6  ← B

Processing theta frame
[AGENT] Theta #3 @ 0.98s: stored vector, DOM=15 ← C

Processing delta frequency
[AGENT] WM before delta @ 1.00s: 3 vectors
[CONSOLIDATE] Sequence length: 3

[AGENT] ✓ Snapshot saved: 3 vectors
[GEN] ✓ A-B-C #1
```

**Analysis:** Perfect timing - 3 theta samples captured 3 tones, delta consolidated at end.

---

### Trace 2: Ambiguous Composition

```
======================================================================
GENERATION TEST: X → ? → ?
======================================================================

Generated 3 vectors:
  Step 0: bands [3, 4, 5], DOM=4     ← X (seed)
  Step 1: bands [5, 6, 7], DOM=6     ← B (correct!)
  Step 2: bands [14, 15, 16], DOM=15 ← ? (C or Z?)

--- Evaluation ---
Step 1 vs B: 0.976 ✓

Step 2:
  vs C: 0.580  ← Moderate match
  vs Z: 0.550  ← Also moderate match
  Difference: 0.030  ← Too small!

? Ambiguous
```

**Analysis:** System correctly predicts B after X, but struggles to choose between C and Z due to similar activation levels.

---

### Trace 3: Spurious Delta Triggers

```
Processing delta frequency
[CONSOLIDATE] Sequence length: 3
[AGENT] ✓ Snapshot saved: 3 vectors

Processing theta frame
[AGENT] Theta #4 @ 1.25s: stored vector, DOM=15  ← Leftover C?

Processing theta frame  (no store - empty echoic)
Processing theta frame  (no store)
Processing theta frame  (no store)

Processing delta frequency
[AGENT] WM before delta @ 2.30s: 1 vectors  ← Only 1!
[CONSOLIDATE] Sequence length: 1
[AGENT] Snapshot captured: 1 vectors  ← Spurious
```

**Analysis:** Delta triggers during silence, consolidates leftover vector. Solution: Slower delta (0.5 Hz).

---

## Parameter Exploration

### Learning Rate Sweep (Pending)

```
lr_fast  W_norm  B_recog  Composition
0.05     0.18    0.89     ?
0.10     0.29    0.96     0.03 diff
0.15     ?       ?        ?
0.20     ?       ?        ?
```

### Decay Rate Sweep (Pending)

```
decay_fast  W_norm  Retention  Composition
0.90        0.31    Poor       ?
0.95        0.29    Better     0.03 diff
0.98        ?       ?          ?
0.99        ?       ?          ?
```

### Training Ratio Sweep

```
Ratio  C_sim  Z_sim  Difference  Result
3:1    0.335  0.806  -0.471      Z dominates
5:1    ?      ?      ?           ?
7:1    0.580  0.550  +0.030      Ambiguous
10:1   ?      ?      ?           ?
15:1   ?      ?      ?           ?
```

---

## Recommendations for Next Session

### High Priority Tests

1. **Direct Association Query**
   ```python
   # Test without generation context
   response_B = agent.hebbian_memory.W @ vec_B_normalized
   
   sim_to_C = cosine(response_B, vec_C)
   sim_to_Z = cosine(response_B, vec_Z)
   
   # Should C > Z by significant margin if learning worked
   ```

2. **Context Ablation**
   ```python
   # Modify generate() to ignore context
   generated = agent.hebbian_memory.generate(
       vec_X, 
       n_steps=2,
       context_vecs=None  # ← Disable context
   )
   ```

3. **10:1 Training Ratio**
   ```python
   sequences = [
       *[("A-B-C", seq([300, 1000, 2500])) for _ in range(10)],
       ("X-B-Z", seq([600, 1000, 3500]))
   ]
   ```

### Medium Priority

4. **Learning Rate = 0.15**
   ```python
   self.learn_pair(vec_a, vec_b, lr_fast=0.15)  # Was 0.10
   ```

5. **Decay = 0.98**
   ```python
   self.W_fast *= 0.98  # Was 0.95
   ```

6. **Logarithmic Frequencies**
   ```python
   # Instead of linear spacing
   freqs = [300, 600, 1200, 2400, 4800]  # Octaves
   ```

---

## Open Research Questions

1. **What is the theoretical minimum training ratio for composition?**
   - Current data suggests >7:1 needed
   - Is there a formula? f(distinctiveness, decay_rate, context_strength)

2. **Why does context sometimes hurt?**
   - Should context be weighted differently for early vs late generation steps?
   - Should context decay over time?

3. **Can we predict composition success from training metrics?**
   - E.g., if W[B→C] / W[B→Z] > threshold, composition will work
   - What threshold?

4. **How does this scale to longer sequences?**
   - A-B-C-D-E + X-B-C-D-F → X-B-C-D-E?
   - Does composition degrade with length?

---

## Conclusion

**What works:**
- Multi-process architecture ✅
- Rhythm synchronization ✅  
- Memory consolidation ✅
- B recognition ✅

**What's on the edge:**
- C vs Z discrimination (0.03 difference, need 0.15)
- Requires 7:1+ training ratio
- Sensitive to parameter tuning

**What's next:**
- Debug direct associations
- Test without context
- Systematic parameter sweep
- Consider different task (not just tones)

**Bottom line:** The system is **fundamentally working** but needs optimization to reliably exhibit composition. This is actually a good place to be - we're close!

---

**Log End Date:** January 5, 2026  
**Total Test Sessions:** 15+  
**Lines of Debug Code Written:** ~2000  
**Coffee Consumed:** Substantial ☕
