#!/usr/bin/env python3
"""
Quick demo of causal chain reasoning.

This demonstrates the core functionality we've built so far:
- CausalMemory: Stores cause-effect rules with Hebbian strengthening
- SimulationEngine: Predicts outcomes through recursive chain reasoning

Run this to see the agent in action!
"""

from src.memory import CausalMemory
from src.simulator import SimulationEngine


def print_section(title):
    """Pretty print section headers."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print_section("PREDICTIVE AGENT - CAUSAL CHAIN REASONING DEMO")

    # Initialize components
    memory = CausalMemory()
    engine = SimulationEngine(memory)

    # ===== TRAINING PHASE =====
    print_section("PHASE 1: Training")

    print("Teaching the agent two causal rules:\n")

    print("Rule 1: A → B causes C")
    print("  (showing this 5 times to strengthen the association)")
    for i in range(5):
        memory.store_rule(["A", "B"], "C")

    print("\nRule 2: X → Y causes A")
    print("  (showing this 5 times to strengthen the association)")
    for i in range(5):
        memory.store_rule(["X", "Y"], "A")

    print("\n" + "-" * 60)
    print("Memory state after training:")
    print("-" * 60)
    print(memory)

    # ===== TESTING PHASE =====
    print_section("PHASE 2: Testing Basic Predictions")

    test_cases = [
        (["A", "B"], "C", "Direct rule lookup"),
        (["X", "Y"], "A", "Direct rule lookup"),
    ]

    for sequence, expected, description in test_cases:
        seq_str = " → ".join(sequence)
        result = engine.predict_outcome(sequence)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status}: {seq_str} ⇒ {result} ({description})")

    # ===== THE KEY TEST =====
    print_section("PHASE 3: Causal Chain Inference (THE KEY TEST!)")

    print("Question: What happens if we do X → Y → B?")
    print("\nThe agent should reason:")
    print("  1. X → Y produces A (from rule 2)")
    print("  2. A → B produces C (from rule 1)")
    print("  3. Therefore: X → Y → B produces C")
    print("\nLet's see if the agent can figure this out...\n")

    print("-" * 60)
    print("Agent's reasoning process:")
    print("-" * 60)

    result, trace = engine.predict_with_trace(["X", "Y", "B"])
    for step in trace:
        print(step)

    print("\n" + "-" * 60)
    print("RESULT:")
    print("-" * 60)

    if result == "C":
        print("✓✓✓ SUCCESS! ✓✓✓")
        print(f"\nThe agent correctly predicted: {result}")
        print("\nThis demonstrates true causal chain reasoning!")
        print("The agent composed two separate rules to make a novel inference.")
    else:
        print(f"✗ FAILED: Got {result}, expected C")

    # ===== NEGATIVE TEST =====
    print_section("PHASE 4: False Inference Prevention")

    print("Question: What happens if we do X → B (without Y)?")
    print("\nThe agent should NOT predict C, because:")
    print("  - X alone doesn't trigger any rule")
    print("  - We need the full sequence X → Y to get A")
    print("\nLet's verify...\n")

    result = engine.predict_outcome(["X", "B"])

    if result is None:
        print("✓ PASS: Agent correctly returned None")
        print("  (no false inference - good!)")
    else:
        print(f"✗ FAIL: Agent predicted {result}")
        print("  (this would be a false inference)")

    # ===== SUMMARY =====
    print_section("SUMMARY")

    print("What we've demonstrated:")
    print("  ✓ Storing causal rules with Hebbian strengthening")
    print("  ✓ Direct rule lookup (A→B ⇒ C)")
    print("  ✓ Causal chain inference (X→Y→B ⇒ C)")
    print("  ✓ False inference prevention (X→B ⇒ None)")
    print("\nThis is a minimal but functional predictive agent!")
    print("\nNext steps:")
    print("  - Wrap in CausalChainAgent class")
    print("  - Run full test suite")
    print("  - Integrate with Oscillatory Memory system")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
