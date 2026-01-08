# Predictive Agent Experiments

Bio-inspired AI agent capable of causal chain reasoning.

## What is this?

This project implements an agent that can:
- Learn cause-effect relationships (e.g., "A→B causes C")
- Chain multiple rules together (e.g., "X→Y causes A" + "A→B causes C" = "X→Y→B causes C")
- Predict outcomes of action sequences through "mental simulation"

**Key innovation:** The agent discovers that `X→Y→B ⇒ C` by composing two learned rules, even though it never observed this specific sequence during training.

## Quick Start

### Run the demo:
```bash
python demo.py
```

This will show:
1. Training phase (teaching the agent two rules)
2. Basic predictions (direct rule lookup)
3. **Causal chain inference** (the key capability)
4. False inference prevention (no hallucinations)

### Run individual components:
```bash
# Test just the memory system
python -m src.memory

# Test just the simulation engine
python -m src.simulator
```

## Project Structure

```
predictive_agent_experiments/
├── src/
│   ├── memory.py      # CausalMemory: Stores rules with Hebbian learning
│   ├── simulator.py   # SimulationEngine: Recursive chain reasoning
│   └── __init__.py
├── docs/
│   └── design_doc.md  # Detailed design document
├── demo.py            # Interactive demonstration
└── test_causal_chain_reasoning.py  # Comprehensive test suite
```

## Current Status

**Phase 1: Minimal Viable Agent** ✓
- [x] CausalMemory implementation
- [x] SimulationEngine with recursive chaining
- [x] Basic demo working
- [ ] CausalChainAgent wrapper (next step)
- [ ] Full test suite passing

## Architecture

### CausalMemory
- Dictionary-based storage for O(1) lookup
- Hebbian strengthening (repeated observations increase confidence)
- Prefix indexing for fast partial matching

### SimulationEngine
- Recursive algorithm for multi-step inference
- Depth-limited search (max 5 levels)
- Cycle detection (prevents infinite loops)
- Optional reasoning trace (for debugging)

**Key insight:** The outcome of one rule becomes the input for the next rule, enabling composition.

## Example

```python
from src.memory import CausalMemory
from src.simulator import SimulationEngine

# Initialize
memory = CausalMemory()
engine = SimulationEngine(memory)

# Train
memory.store_rule(['A', 'B'], 'C')  # A→B causes C
memory.store_rule(['X', 'Y'], 'A')  # X→Y causes A

# Predict
result = engine.predict_outcome(['X', 'Y', 'B'])
print(result)  # Output: 'C'

# Show reasoning
result, trace = engine.predict_with_trace(['X', 'Y', 'B'])
for step in trace:
    print(step)
```

## Biological Inspiration

This architecture is inspired by:
- **Hippocampal replay**: Offline simulation of action sequences
- **Sharp-wave ripples**: Consolidation of causal structure during sleep
- **Hebbian learning**: "Cells that fire together, wire together"
- **Prefrontal cortex**: Planning and future prediction

## Next Steps

1. **Phase 1 completion**: CausalChainAgent wrapper class
2. **Phase 2**: Pass all 8 tests from test suite
3. **Phase 3**: Integration with Oscillatory Memory system (theta/delta rhythms)
4. **Phase 4**: Hierarchical abstraction and transfer learning

## Design Philosophy

Following the Oscillatory Memory project approach:
- Biological plausibility over pure performance
- Emergent properties over explicit programming
- Documentation of exploration process (even failures)
- Publishing findings as blog series

## Documentation

See `docs/design_doc.md` for:
- Complete architecture design
- Implementation phases
- Open questions and challenges
- References to neuroscience literature

## Related Work

This builds on the **Oscillatory Memory** project, which implements:
- Dual-trace Hebbian learning
- Theta (4Hz) and delta (1Hz) brain rhythms
- Compositional sequence learning
- Emergent properties (primacy effects, temporal asymmetry)

## License

Research project - see individual file headers for details.

## Author

Marcin - Bio-inspired AI researcher
January 2026