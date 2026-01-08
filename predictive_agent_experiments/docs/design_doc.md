# Predictive Agent Design Document

**Project:** Causal Chain Reasoning in Bio-Inspired AI Agents  
**Author:** Marcin  
**Date:** January 7, 2026  
**Status:** Initial Design / Pre-Implementation

---

## Executive Summary

This document outlines the design for a bio-inspired AI agent capable of causal chain reasoning - the ability to predict outcomes of actions by understanding and composing cause-effect relationships. Building on the foundation of the Oscillatory Memory system (theta/delta rhythms, Hebbian learning), this project extends the agent's capabilities from simple sequence learning to complex multi-step causal inference.

**Core Challenge:** Can an agent learn that `X→Y causes A` and `A→B causes C`, then correctly infer that `X→Y→B causes C`?

---

## 1. Motivation & Background

### 1.1 The Problem

Current Oscillatory Memory system successfully implements:
- Compositional learning (can combine A-B-C and X-B-Z to form X-B-C)
- Temporal sequence memory with theta/delta oscillations
- Hebbian association learning

However, it lacks **predictive reasoning** - the ability to:
- Understand cause-effect relationships
- Chain multiple causal rules together
- Simulate outcomes before taking actions ("mental simulation")

### 1.2 Biological Inspiration

In biological brains:
- **Hippocampus** performs "replay" - simulating action sequences offline during sleep/rest
- **Prefrontal cortex** implements planning and future prediction
- **Place cells** and **grid cells** encode state spaces for navigation
- **Sharp-wave ripples** during sleep consolidate causal relationships

### 1.3 Why This Matters

An agent that can predict consequences enables:
- Planning (choose action sequences to reach desired states)
- Transfer learning (apply learned rules in novel contexts)
- Counterfactual reasoning ("what if I had done X instead?")
- True compositional understanding beyond surface-level pattern matching

---

## 2. Core Concepts

### 2.1 Definitions

**Causal Rule:** A learned association that "when sequence S occurs in context C, outcome O follows"
- Example: `A→B` in any context causes `C`

**Causal Chain:** Multiple causal rules where the outcome of one becomes the input/context for another
- Example: `X→Y ⇒ A`, then `A→B ⇒ C`
- Full chain: `X→Y→B ⇒ C`

**Mental Simulation:** Agent's ability to "imagine" executing a sequence and predict its outcome without physically performing it

**State vs Stimulus:**
- **Stimulus:** External input/observation (what the agent perceives)
- **State:** Internal representation/context (what the agent "knows" about current situation)
- Critical distinction: `A` can be both an outcome (state) and a precondition (context)

### 2.2 Test Scenario (Canonical Example)

```
Training Phase:
1. Observe: A→B ⇒ C  (repeated 5x)
2. Observe: X→Y ⇒ A  (repeated 5x)

Test Phase:
Query: X→Y→B ⇒ ?
Expected: C

Reasoning Chain:
- X→Y triggers rule 2 → produces A
- A becomes internal state/context
- A→B triggers rule 1 → produces C
```

---

## 3. Architecture Design

### 3.1 High-Level Components

```
┌─────────────────────────────────────────────┐
│         Causal Chain Agent                   │
├─────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   Causal Memory (Rule Storage)      │    │
│  │   - Stores: Sequence → Outcome      │    │
│  │   - Hebbian associations            │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   Simulation Engine                 │    │
│  │   - "Imagines" action sequences     │    │
│  │   - Applies causal rules            │    │
│  │   - Chains outcomes as new inputs   │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │   State Representation              │    │
│  │   - Current context/state           │    │
│  │   - Distinguishes stimulus vs state │    │
│  └────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

### 3.2 Detailed Component Design

#### 3.2.1 Causal Memory

**Purpose:** Store learned cause-effect relationships

**Data Structure:**
```python
class CausalRule:
    sequence: List[str]      # e.g., ['A', 'B']
    outcome: str             # e.g., 'C'
    confidence: float        # Hebbian weight (strengthens with repetition)
    context: Optional[str]   # Contextual dependencies (future extension)
```

**Operations:**
- `store_rule(sequence, outcome)` - Add or strengthen rule
- `query_rule(sequence) -> outcome` - Retrieve outcome for sequence
- `get_matching_rules(partial_sequence)` - Find rules that could apply

**Implementation Notes:**
- Use dictionary/hash map for O(1) lookup: `{tuple(sequence): outcome}`
- Hebbian strengthening: increment confidence on repeated observations
- Decay mechanism (optional): weaken unused rules over time

#### 3.2.2 Simulation Engine

**Purpose:** Predict outcomes by mentally executing sequences

**Algorithm (Recursive Chain Simulation):**
```python
def predict_outcome(sequence):
    # Base case: direct rule match
    if sequence in causal_memory:
        return causal_memory[sequence]
    
    # Recursive case: try chaining
    for split_point in range(1, len(sequence)):
        prefix = sequence[:split_point]
        suffix = sequence[split_point:]
        
        # Get outcome of prefix
        intermediate = predict_outcome(prefix)
        
        if intermediate is not None:
            # Try using intermediate as context for suffix
            new_sequence = [intermediate] + suffix
            result = predict_outcome(new_sequence)
            
            if result is not None:
                return result
    
    return None  # No prediction possible
```

**Key Insight:** The outcome of one sequence becomes part of the input for the next.

#### 3.2.3 State Representation

**Challenge:** Distinguish between:
- **External stimuli** (e.g., X, Y, B) - observed from environment
- **Internal states** (e.g., A) - produced by agent's own reasoning

**Solution:**
- Tag each element as either `stimulus` or `state`
- Internal states can trigger causal rules just like external stimuli
- This enables the "A becomes context for A→B" mechanism

---

## 4. Integration with Oscillatory Memory

### 4.1 Where Does This Fit?

Current oscillatory architecture:
- **Gamma (40Hz):** Sensory processing (audio features)
- **Theta (4Hz):** Working memory / sequence encoding
- **Delta (1Hz):** Long-term consolidation (Hebbian learning)

### 4.2 Proposed Extension

**Delta oscillations** already perform Hebbian learning for sequences. Extend this to:
1. Learn not just sequences, but sequence→outcome mappings
2. During "offline" consolidation (when no input), perform causal chain simulation
3. Store successful simulations as new composite rules

**Theta oscillations** could handle:
- Maintaining current "imagined" state during simulation
- Stepping through multi-step causal chains
- Similar to hippocampal replay during theta oscillations

### 4.3 Biological Parallel

- **Sharp-wave ripples (SWRs)** in hippocampus during slow-wave sleep
- Replay sequences at 10x speed
- Consolidate causal structure, not just temporal sequences
- Our system: Delta phase could trigger simulation replays

---

## 5. Implementation Phases

### Phase 1: Minimal Viable Agent (Week 1-2)
**Goal:** Pass basic causal chain test (Test 3 from test suite)

Deliverables:
- Simple dictionary-based causal memory
- Recursive simulation engine
- Pass: `X→Y→B ⇒ C` inference

### Phase 2: Robustness & Edge Cases (Week 3-4)
**Goal:** Handle all test scenarios

Additions:
- Branching outcomes (Test 7: A→B⇒C vs A→D⇒E)
- Longer chains (Test 6: 3-level chains)
- False inference prevention (Test 5: X→B should not give C)

### Phase 3: Oscillatory Integration (Week 5-6)
**Goal:** Integrate with existing Oscillatory Memory system

Implementation:
- Add causal reasoning to delta phase
- Implement offline simulation/replay
- Test on hardware (Raspberry Pi setup)

### Phase 4: Hierarchical Abstraction (Future)
**Goal:** Learn abstract causal rules

Example:
- Learn: "Moving forward causes distance to wall decrease"
- Abstract: "Action toward object → closer to object"
- Transfer: Apply to any object, not just walls

---

## 6. Test Strategy

### 6.1 Unit Tests

See `test_causal_chain_reasoning.py` for comprehensive test suite:
- Test 1-2: Basic rule learning
- Test 3-4: **Core causal chain inference** (critical)
- Test 5: Negative cases (no false inferences)
- Test 6-8: Advanced scenarios

### 6.2 Integration Tests

With full Oscillatory Memory system:
- Audio input → causal prediction
- Real-time environment interaction (robot navigation)
- Multi-modal integration (vision + audio)

### 6.3 Success Metrics

**Quantitative:**
- 100% pass rate on unit test suite
- Inference latency < 100ms for 3-level chains
- Zero false positive predictions (Test 5)

**Qualitative:**
- Agent demonstrates "understanding" through correct novel compositions
- Emergent behaviors (agent discovers new action sequences)
- Biological plausibility (resembles hippocampal replay patterns)

---

## 7. Open Questions & Challenges

### 7.1 Computational Complexity

**Problem:** Recursive search could explode with long sequences

**Solutions to explore:**
- Depth-limited search (max 3-5 chain levels)
- Beam search (keep only top-k candidates)
- Caching intermediate results
- Pruning based on confidence scores

### 7.2 Context Sensitivity

**Problem:** Real-world rules often depend on context
- Example: "Pushing button → light on" (only if switch is on)

**Future extension:**
- Add context field to CausalRule
- Match rules based on both sequence AND current state
- Requires richer state representation

### 7.3 Temporal Abstraction

**Problem:** Rules at different timescales
- Fast: "Press key → character appears" (ms)
- Slow: "Plant seed → flower grows" (weeks)

**Potential solution:**
- Multi-timescale oscillations (already in architecture!)
- Delta for slow rules, theta for fast rules
- Hierarchical time representation

### 7.4 Learning from Failure

**Problem:** Agent needs both positive and negative examples
- Positive: "X→Y causes A"
- Negative: "X→Z does NOT cause A"

**Implementation:**
- Store null outcomes: `{('X', 'Z'): None}`
- Or explicit failure records with confidence penalties
- Prevents overgeneralization

---

## 8. Expected Outcomes

### 8.1 Successful Implementation

Agent should demonstrate:
1. **Chaining:** Correctly infer X→Y→B ⇒ C from learned rules
2. **Composition:** Combine rules in novel ways not seen during training
3. **Discrimination:** Not make false inferences (X→B ≠ C)
4. **Scalability:** Handle 3+ level causal chains
5. **Transfer:** Apply learned causal structure to new domains

### 8.2 Potential Novel Findings

Interesting emergent properties to look for:
- **Primacy effects** in causal reasoning (first-learned rules stronger)
- **Temporal asymmetry** (forward prediction ≠ backward inference)
- **Spontaneous planning** (agent discovers optimal action sequences)
- **Causal abstraction** (agent forms higher-order rules)

Similar to what was observed in Oscillatory Memory (primacy, asymmetry weren't explicitly programmed).

---

## 9. Documentation & Publication Plan

Following the research blog series approach:

### Blog Post 1: "The Question"
- Motivation: Why causal reasoning matters
- The challenge of multi-step inference
- Connection to biological systems

### Blog Post 2: "The Design"
- Architecture overview
- Key design decisions
- Why this approach (vs alternatives)

### Blog Post 3: "The Implementation"
- Code walkthrough
- Interesting technical challenges
- Test results

### Blog Post 4: "The Surprises"
- Unexpected emergent behaviors
- What worked / what didn't
- Lessons learned

---

## 10. References & Related Work

### Biological Neuroscience
- Hippocampal replay and sharp-wave ripples
- Place cells and grid cells (state space representation)
- Prefrontal cortex planning mechanisms

### AI/ML Literature
- World models (Ha & Schmidhuber, 2018)
- Model-based reinforcement learning
- Neural Turing Machines / Memory-Augmented Networks
- Compositional learning in neural networks

### Previous Work (Oscillatory Memory)
- Theta/delta rhythm architecture
- Hebbian learning for sequences
- Emergent properties (primacy, temporal asymmetry)
- Compositional sequence generation

---

## 11. Next Steps

### Immediate (This Week)
1. Implement `CausalChainAgent` base class
2. Create simple dictionary-based causal memory
3. Implement recursive simulation engine
4. Get Test 3 passing

### Short-term (Next 2 Weeks)
1. Complete all unit tests (Tests 1-8)
2. Add confidence/Hebbian strengthening
3. Document implementation decisions
4. Write Blog Post 1 & 2

### Medium-term (Month 2)
1. Integrate with Oscillatory Memory
2. Test on hardware (Raspberry Pi)
3. Measure performance metrics
4. Write Blog Post 3 & 4

---

## Appendix A: Alternative Approaches Considered

### A.1 Graph-Based Reasoning

**Approach:** Represent causal rules as directed graph, use graph search

**Pros:**
- Natural representation of causal chains
- Well-established algorithms (BFS, DFS)
- Easy visualization

**Cons:**
- Less biologically plausible
- Harder to integrate with oscillatory architecture
- Doesn't leverage existing Hebbian learning

**Decision:** Deferred to future extension

### A.2 Neural Network End-to-End

**Approach:** Train neural net to map sequences → outcomes

**Pros:**
- Leverages modern deep learning
- Could handle noisy/fuzzy patterns

**Cons:**
- Black box (hard to interpret)
- Not bio-inspired
- Requires large training data
- Doesn't explain the mechanism

**Decision:** Not aligned with project goals (biological plausibility, interpretability)

### A.3 Symbolic Logic / Planning

**Approach:** Use PDDL-style planning with symbolic rules

**Pros:**
- Provably correct
- Handles complex constraints
- Well-studied in AI

**Cons:**
- Not compatible with oscillatory architecture
- Brittle (requires perfect symbolic encoding)
- No learning mechanism

**Decision:** Too different from bio-inspired approach

---

## Appendix B: Glossary

- **Causal Rule:** Association between sequence and outcome
- **Mental Simulation:** Internal prediction without action
- **Hebbian Learning:** "Cells that fire together, wire together"
- **Oscillatory Memory:** Previous project using brain rhythms for memory
- **State Space:** Set of all possible internal representations
- **Compositional Learning:** Ability to combine learned elements in novel ways
- **Replay:** Biological mechanism of re-experiencing sequences offline
- **Sharp-Wave Ripple:** Hippocampal oscillation during memory consolidation

---

**End of Design Document**

*This is a living document and will be updated as implementation progresses and new insights emerge.*
