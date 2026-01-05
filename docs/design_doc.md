# Oscillatory Hierarchical Memory System
## Design Document v1.0

**Author:** Research Project  
**Date:** January 2026  
**Status:** Pre-Implementation Phase

---

## Executive Summary

This document describes a bio-inspired neural architecture that uses brain oscillations at different frequencies to create temporal hierarchy, enabling compositional learning through reusable chunks. The system is grounded in neuroscience (theta-gamma coupling, phase coding) and aims to solve the compositionality problem that current AI struggles with.

**Core Innovation:** Oscillatory rhythms don't just modulate learning - they ARE the fundamental mechanism for temporal segmentation and hierarchical memory organization.

---

## 1. Core Hypothesis

### The Central Claim
**"Brain oscillations at different frequencies create natural temporal chunking that enables compositional, reusable representations - analogous to functions, classes, and programs in code."**

### Key Principles
1. **Frequency = Level of Abstraction**
   - Fast rhythms (40 Hz): micro-events, details
   - Medium rhythms (4-8 Hz): chunks, phrases
   - Slow rhythms (1-2 Hz): sequences, episodes

2. **Reusability Through Chunking**
   - Chunk "A-B-C" learned once, reused in multiple contexts
   - Like functions in programming: define once, call many times
   - Enables compositional generation (e.g., "Bob ma kota" from "Ala ma kota" + "Bob ma psa")

3. **Biological Grounding**
   - Theta-gamma coupling (documented in hippocampus)
   - Phase coding (information in timing, not just rate)
   - Working memory capacity limits (~7 chunks)
   - Dual-trace learning (STM + LTM on same synapses)

---

## 2. Architecture Overview

### 2.1 Memory Hierarchy

```
┌─────────────────────────────────────────┐
│ AUDIO STREAM (16 kHz)                   │
│ Continuous raw audio                    │
└──────────────┬──────────────────────────┘
               │ (25ms hops)
               ↓
┌─────────────────────────────────────────┐
│ ECHOIC MEMORY (~500ms)                  │
│ - Ring buffer: 20 frames × 32 bands     │
│ - Ultra-short sensory storage           │
│ - np.roll() implementation              │
└──────────────┬──────────────────────────┘
               │ (theta-gated sampling)
               ↓
┌─────────────────────────────────────────┐
│ WORKING MEMORY (~2s, theta-driven)      │
│ - Fixed capacity: 7 chunks              │
│ - Dual-trace learning (fast + slow)    │
│ - Pause-based segmentation              │
└──────────────┬──────────────────────────┘
               │ (consolidation)
               ↓
┌─────────────────────────────────────────┐
│ LONG-TERM MEMORY (permanent)            │
│ - Vector quantization (discrete chunks) │
│ - Bidirectional associations            │
│ - Compositional retrieval               │
└─────────────────────────────────────────┘
```

### 2.2 Key Components

#### Echoic Memory
- **Purpose:** Ultra-short-term audio buffer
- **Implementation:** Fixed-size ring buffer (20, 32)
- **Update:** Every audio hop via `np.roll(shift=-1)`
- **Biology:** Auditory sensory memory (200-500ms window)
- **Role:** Provides temporal context for theta sampling

#### Working Memory (Theta Level)
- **Purpose:** Active chunk storage and manipulation
- **Capacity:** 7 chunks (Miller's Law, biological constraint)
- **Control:** Theta rhythm (4-8 Hz, adaptive via entrainment)
- **Learning:** Dual-trace (see Section 3.2)
- **Segmentation:** Pause detection triggers consolidation

#### Long-Term Memory
- **Purpose:** Permanent pattern storage
- **Representation:** Discrete chunk IDs (vector quantization)
- **Structure:** Bidirectional association matrix
- **Query modes:** Forward, backward, context, fill-in

---

## 3. Key Mechanisms

### 3.1 Entrainment

**Purpose:** Automatically detect and synchronize with signal rhythm

**Implementation:**
```python
class HybridFrequencyEntrainment:
    - Onset detection (energy-based)
    - Inter-onset interval (IOI) measurement
    - Median filtering (robust to outliers)
    - Confidence metric (CV-based)
    - Adaptive frequency (exponential smoothing)
```

**Gating Logic:**
```python
if confidence > 0.8:
    theta_rhythm.start(detected_frequency)
    # Begin sampling only when confident
else:
    # Continue entraining, don't commit to sampling
```

**Parameters:**
- Confidence threshold: 0.8
- Adaptation rate: 0.1
- Frequency range: 3-8 Hz (theta band)

### 3.2 Dual-Trace Learning

**Biological Basis:** Same synapses support both STM and LTM through different molecular mechanisms (CaMKII for STM, PKMζ for LTM)

**Implementation:**
```python
# Fast trace (Short-Term Memory)
W_fast += lr_fast * error * input
W_fast *= decay_fast

# Slow trace (Long-Term Memory)
W_slow += lr_slow * error * input  
W_slow *= decay_slow

# Effective weight
W = W_fast + W_slow
```

**Parameters:**
- lr_fast: 0.1 (high learning rate)
- decay_fast: 0.9 (rapid forgetting)
- lr_slow: 0.001 (low learning rate)
- decay_slow: 0.999 (slow forgetting)

**Effect:**
- Single exposure → only W_fast affected (temporary)
- Repeated exposure → W_slow accumulates (permanent)
- Natural consolidation without explicit threshold

### 3.3 Pause-Based Segmentation

**Purpose:** Detect natural boundaries in sequences (variable length!)

**Energy-Based Detection:**
```python
energy = np.sum(S_auditory ** 2)
if energy < threshold:
    silence_frames += 1
else:
    silence_frames = 0

is_pause = silence_frames >= min_pause_frames
```

**Parameters:**
- Silence threshold: 0.1
- Minimum pause duration: 150ms (6 frames @ 25ms)

**Boundary Actions:**
1. Consolidate current working memory
2. Process variable-length sequence
3. Clear working memory buffer
4. Reset pause counter

**Why Not Fixed Delta Rhythm?**
- Real sequences have variable length (3, 4, 5... elements)
- Pauses are natural semantic boundaries
- Delta serves as backup timeout (1 Hz), not strict clock

### 3.4 Vector Quantization

**Purpose:** Convert continuous representations to discrete, reusable chunks

**Implementation:**
```python
class VectorQuantizer:
    def __init__(self, n_codes=16, code_dim=32):
        self.codebook = np.random.randn(n_codes, code_dim)
    
    def encode(self, vector):
        # Find nearest code
        distances = np.linalg.norm(self.codebook - vector, axis=1)
        return np.argmin(distances)
    
    def update(self, vector, code_id):
        # Move code toward vector (online learning)
        self.codebook[code_id] += lr * (vector - self.codebook[code_id])
```

**Parameters:**
- Codebook size: 16 codes
- Code dimension: 32 (frequency bands)
- Update rate: 0.01

**Why Discrete?**
- Enables symbolic compositionality
- Reusable building blocks
- Interpretable representations

### 3.5 Bidirectional Associations

**Purpose:** Support forward, backward, and fill-in queries from ONE structure

**Why?** Brain doesn't have separate "forward" and "backward" networks - it has associative memory that works in all directions.

**Implementation Option A: Symmetric Matrix**
```python
class AssociativeMemory:
    def __init__(self, n_chunks=16):
        self.W = np.zeros((n_chunks, n_chunks))  # Symmetric!
    
    def learn_sequence(self, chunk_ids):
        # Learn all pairwise associations
        for i in range(len(chunk_ids)):
            for j in range(i+1, len(chunk_ids)):
                self.W[chunk_ids[i], chunk_ids[j]] += 1
                self.W[chunk_ids[j], chunk_ids[i]] += 1  # Bidirectional
    
    def query_forward(self, chunk_id):
        return np.argmax(self.W[chunk_id, :])
    
    def query_backward(self, chunk_id):
        return np.argmax(self.W[:, chunk_id])
```

**Implementation Option B: Temporal Context Model** (TBD)
- Context vector evolves over time
- Chunks associated with contexts
- Both forward and backward inference natural

---

## 4. Key Design Decisions

### ✅ Adopted

| Decision | Rationale |
|----------|-----------|
| **Fixed capacity buffers** | Biological constraint (working memory ~7 items), forces hierarchical organization |
| **Ring buffers (not lists)** | Neurons can't do infinite append, displacement is natural |
| **Pause-driven segmentation** | Variable-length sequences, natural boundaries |
| **Bidirectional associations** | One structure, multiple query types (like brain) |
| **Dual-trace learning** | STM/LTM on same weights, consolidation emerges from repetition |
| **Echoic memory naming** | Semantic clarity, biological grounding |
| **Discrete chunks (VQ)** | Enables symbolic compositionality, reusability |

### ❌ Rejected

| Decision | Reason |
|----------|--------|
| **Infinite lists** | Not biologically plausible, no capacity constraint |
| **Fixed 4-slot buffer** | Assumes fixed sequence length, doesn't generalize |
| **Slot-based learning** | Doesn't work for variable-length sequences |
| **Forward-only prediction** | Brain uses bidirectional associations |
| **Explicit gamma rhythm** | Audio hop rate IS the gamma layer (implicit) |
| **Arbitrary consolidation threshold** | Dual-trace makes it emergent |

### 🤔 Open Questions

1. **Exact quantization method:**
   - Simple k-means vs. online VQ?
   - How to handle codebook size growth?
   - Similarity metric: Euclidean vs. cosine?

2. **Association mechanism:**
   - Symmetric matrix (Hopfield-style)?
   - Temporal Context Model (Howard & Kahana)?
   - Energy-based model?
   - → Need to test which gives best compositionality

3. **Oscillatory coupling:**
   - Should theta frequency adapt echoic memory size?
   - Delta as strict clock or just timeout?

4. **Baseline comparison:**
   - What to compare against? (LSTM, Transformer, vanilla RNN?)
   - Fair comparison metrics?

---

## 5. Test Case: Compositional Generation

### 5.1 "Bob ma kota" Experiment

**Hypothesis:** System can generate novel valid combinations from learned components.

**Training Data:**
```
Sequence 1: A-B-C (e.g., "Ala ma kota")
Sequence 2: X-B-D (e.g., "Bob ma psa")

Repeated 50 times each
```

**Expected Learning:**
- Chunk vocabulary: {A, B, C, X, D}
- Associations: A↔B, B↔C, X↔B, B↔D
- Bidirectional: can query forward/backward

**Test:**
```
Query 1: [X, B, ?] → should generate C
         "Bob ma ___" → "kota"

Query 2: [?, B, C] → should generate A or X
         "___ ma kota" → "Ala" lub "Bob"

Novel combination: X-B-C never seen in training!
```

**Success Metric:**
- Compositionality score: % of generated sequences that are:
  - Novel (not in training)
  - Valid (use learned chunks correctly)
  - Semantically sensible

**Baseline:**
- Random generation: ~6% (2/32 chance)
- Flat network: 0% (never saw X-B-C)
- Expected from system: >50%

### 5.2 Additional Tests

**Transfer learning:**
- Train on rhythm 1 (4 Hz), test on rhythm 2 (6 Hz)
- Does entrainment + chunking enable tempo-invariance?

**Sample efficiency:**
- How many examples needed vs. baseline?
- Does reusability reduce data requirements?

**Hierarchical grammar:**
- Words: [A-B-C] = word1, [X-Y-Z] = word2
- Sentences: word1 [pause] word2
- Does system discover word level automatically?

---

## 6. Implementation Roadmap

### Phase 1: Core Components (Week 1-2)
- [x] Entrainment (HybridFrequencyEntrainment)
- [x] Echoic memory (ring buffer)
- [ ] Working memory (fixed capacity, dual-trace)
- [ ] Pause detection (energy-based)
- [ ] Basic integration

### Phase 2: Memory Systems (Week 3)
- [ ] Vector quantization (codebook learning)
- [ ] Bidirectional associations (choose mechanism)
- [ ] Consolidation flow (echoic → working → long-term)

### Phase 3: Testing (Week 4)
- [ ] Generate synthetic sequences (A-B-C, X-B-D)
- [ ] Train system
- [ ] Compositionality test ("Bob ma kota")
- [ ] Metrics and visualization

### Phase 4: Iteration (Week 5+)
- [ ] Analyze results
- [ ] Refine mechanisms based on data
- [ ] Scale up (more chunks, longer sequences)
- [ ] Real audio (if synthetic works)

---

## 7. Technical Specifications

### 7.1 Audio Processing
```python
FS = 16000              # Sampling rate (Hz)
HOP_MS = 25             # Hop size (ms)
hop_size = 400          # Samples per hop
n_sensors = 32          # Frequency bands
FFT_SIZE = 2048         # FFT window
```

### 7.2 Memory Sizes
```python
echoic_size = 20        # Frames (~500ms)
working_capacity = 7    # Chunks (~2s @ 4Hz)
codebook_size = 16      # Discrete chunks
code_dim = 32           # Chunk dimensionality
```

### 7.3 Learning Rates
```python
# Dual-trace
lr_fast = 0.1
decay_fast = 0.9
lr_slow = 0.001
decay_slow = 0.999

# VQ
lr_vq = 0.01

# Associations
lr_assoc = 0.05
```

### 7.4 Thresholds
```python
confidence_threshold = 0.8    # Entrainment gating
silence_threshold = 0.1       # Pause detection
min_pause_frames = 6          # 150ms
```

---

## 8. Expected Outcomes

### 8.1 Success Criteria

**Minimum Viable:**
- System entrains to rhythm (confidence >0.8)
- Segments sequences on pauses
- Learns discrete chunks (codebook converges)
- Generates SOME novel combinations

**Strong Success:**
- Compositionality score >50% on "Bob ma kota"
- Tempo-invariant (works for 3-8 Hz)
- Sample-efficient (fewer examples than baseline)
- Interpretable chunks (visualizable codebook)

**Exceptional:**
- Discovers hierarchical structure (words, sentences)
- Transfers to new domains (music, speech)
- Emergent grammar learning

### 8.2 Potential Issues

| Issue | Mitigation |
|-------|------------|
| Entrainment fails | Manual rhythm setting for debugging |
| Codebook collapse | Regularization, minimum distance constraint |
| No compositionality | Try different association mechanism |
| Too slow | Optimize NumPy operations, consider Numba |
| Overfitting | Larger codebook, regularization |

---

## 9. Biological Plausibility Checklist

- [x] Fixed capacity working memory (~7 chunks)
- [x] Ring buffers (displacement, not infinite growth)
- [x] Dual-trace learning (STM + LTM same substrate)
- [x] Echoic memory (~500ms auditory buffer)
- [x] Theta rhythm control (4-8 Hz chunking)
- [x] Phase-dependent processing
- [x] Bidirectional associations
- [x] Pause-based segmentation (natural boundaries)
- [x] Entrainment (synchronization to external rhythm)
- [ ] Backward replay (during consolidation) - future work
- [ ] Multi-scale coupling (theta-gamma) - simplified for now

---

## 10. Comparison to Related Work

| Approach | Oscillations? | Compositionality? | Biological? | Our Advantage |
|----------|---------------|-------------------|-------------|---------------|
| Transformers | ❌ | Partial (attention) | ❌ | True temporal hierarchy |
| LSTM/GRU | ❌ | ❌ | Partial | Fixed capacity, dual-trace |
| Liquid Time Constants | ✅ | ❌ | ✅ | Compositional chunks |
| Spiking Networks | ✅ | ❌ | ✅ | Symbolic compositionality |
| Hopfield Networks | ❌ | ✅ | Partial | Oscillatory control |
| VQ-VAE | ❌ | ✅ | ❌ | Temporal chunking |
| **This Work** | ✅ | ✅ | ✅ | **Unique combination** |

---

## 11. Key References

### Neuroscience
- **Theta-gamma coupling:** Lisman & Idiart (1995), "Storage of 7±2 short-term memories in oscillatory subcycles"
- **Phase coding:** O'Keefe & Recce (1993), "Phase relationship between hippocampal place units and the EEG theta rhythm"
- **Dual-trace learning:** Benna & Fusi (2016), "Computational principles of synaptic memory consolidation"
- **Working memory capacity:** Miller (1956), "The magical number seven"; Cowan (2001), "The magical number 4"
- **Echoic memory:** Neisser (1967), "Cognitive Psychology"

### Related ML Work
- **VQ-VAE:** van den Oord et al. (2017)
- **Liquid Networks:** Hasani et al. (2021)
- **Temporal Context Model:** Howard & Kahana (2002)
- **Hopfield Networks:** Hopfield (1982)

---

## 12. Next Steps

### Before Implementation
1. **Decide association mechanism** (symmetric matrix vs TCM)
2. **Sketch compositionality test** on paper
3. **Define exact metrics** (compositionality score formula)

### Week 1 Goals
1. Integrate echoic → working memory flow
2. Implement dual-trace weights
3. Add pause detection
4. Test on simple metronome

### Future Directions
- Real speech/music data
- Deeper hierarchy (3+ levels)
- Online learning (continuous stream)
- Neuromorphic hardware deployment

---

## Appendix A: Code Structure (Planned)

```python
# Core components
class EchoicMemory:
    - Ring buffer (np.roll)
    - Fixed size (20, 32)

class WorkingMemory:
    - Fixed capacity (7 chunks)
    - Dual-trace weights
    - Pause detection
    - Consolidation trigger

class LongTermMemory:
    - VectorQuantizer (codebook)
    - AssociativeMemory (bidirectional)
    - Query methods (forward/backward/fill)

class BrainRhythm:
    - Phase accumulation
    - Frequency adaptation
    - Value (sin wave)

class BioAgent:
    - Integrates all components
    - Audio processing (hear)
    - Oscillatory control (theta, delta)
    - Learning and retrieval
```

---

## Appendix B: Terminology

- **Chunk:** Discrete unit of memory (e.g., syllable, note, gesture)
- **Entrainment:** Synchronization of internal rhythm to external signal
- **Eligibility trace:** Credit assignment signal for learning
- **Compositionality:** Ability to combine known elements in novel ways
- **Theta rhythm:** 4-8 Hz oscillation, associated with memory and chunking
- **Working memory:** Limited-capacity active storage (~7 items, ~2s)
- **Echoic memory:** Ultra-short sensory buffer for auditory input (~500ms)
- **Dual-trace:** Fast (STM) and slow (LTM) learning on same connections
- **Phase coding:** Information encoded in timing relative to oscillation

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Ready for implementation phase

This design represents the culmination of deep thinking about biological mechanisms, architectural abstractions, and practical implementation. The system aims to demonstrate that oscillatory hierarchies can enable compositional learning - a fundamental capability that current AI lacks.

**Next:** Prototype implementation and empirical validation.