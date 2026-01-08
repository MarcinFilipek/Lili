"""
SimulationEngine - Mental simulation for causal chain reasoning.

This module implements the "mental simulation" capability - the agent's
ability to predict outcomes by chaining causal rules together recursively.

Key insight: The outcome of one rule can become the input for another rule,
enabling multi-step causal inference.
"""

from typing import List, Optional, Set, Tuple
from memory import CausalMemory


class SimulationEngine:
    """
    Engine for predicting outcomes through causal chain reasoning.

    This implements a recursive algorithm that:
    1. Tries direct rule lookup first (base case)
    2. If no direct match, tries to split the sequence
    3. Predicts outcome of prefix, uses it as context for suffix
    4. Chains predictions together to form multi-step inference

    Example:
        Given rules: X→Y ⇒ A, A→B ⇒ C
        Query: X→Y→B ⇒ ?

        Process:
        1. No direct rule for X→Y→B
        2. Split at position 2: prefix=[X,Y], suffix=[B]
        3. Predict prefix: X→Y ⇒ A
        4. Chain: A→B ⇒ C
        5. Return: C
    """

    def __init__(self, memory: CausalMemory, max_depth: int = 5):
        """
        Initialize the simulation engine.

        Args:
            memory: The causal memory to query
            max_depth: Maximum recursion depth (prevents infinite loops)
        """
        self.memory = memory
        self.max_depth = max_depth

    def predict_outcome(
        self,
        sequence: List[str],
        depth: int = 0,
        visited: Optional[Set[Tuple[str, ...]]] = None,
    ) -> Optional[str]:
        """
        Predict the outcome of a sequence through causal chain reasoning.

        This is the core algorithm. It uses recursion to:
        1. Check for direct rule match (base case)
        2. Try splitting sequence at different points
        3. Recursively predict prefix outcome
        4. Use that outcome as context for suffix
        5. Chain the predictions together

        Args:
            sequence: The action sequence to predict
            depth: Current recursion depth (internal)
            visited: Set of already-visited sequences (prevents cycles)

        Returns:
            Predicted outcome, or None if no prediction possible

        Example:
            >>> engine = SimulationEngine(memory)
            >>> memory.store_rule(['A', 'B'], 'C')
            >>> memory.store_rule(['X', 'Y'], 'A')
            >>> engine.predict_outcome(['X', 'Y', 'B'])
            'C'
        """
        # Initialize visited set on first call
        if visited is None:
            visited = set()

        # Depth limit check (prevent infinite recursion)
        if depth >= self.max_depth:
            return None

        # Convert to tuple for hashing
        seq_tuple = tuple(sequence)

        # Cycle detection (prevent infinite loops)
        if seq_tuple in visited:
            return None
        visited.add(seq_tuple)

        # Base case: Direct rule lookup
        direct_outcome = self.memory.query_rule(sequence)
        if direct_outcome is not None:
            return direct_outcome

        # Recursive case: Try chaining rules
        # We'll try splitting the sequence at different points
        # For sequence [X, Y, B], we try:
        #   - prefix=[X], suffix=[Y, B]
        #   - prefix=[X, Y], suffix=[B]

        for split_point in range(1, len(sequence)):
            prefix = sequence[:split_point]
            suffix = sequence[split_point:]

            # Predict outcome of prefix
            intermediate_state = self.predict_outcome(prefix, depth + 1, visited.copy())

            if intermediate_state is not None:
                # Success! Now try using this intermediate state with suffix
                # The key insight: intermediate becomes part of new sequence
                new_sequence = [intermediate_state] + suffix

                # Recursively predict with new sequence
                final_outcome = self.predict_outcome(
                    new_sequence, depth + 1, visited.copy()
                )

                if final_outcome is not None:
                    return final_outcome

        # No prediction possible
        return None

    def predict_with_trace(
        self, sequence: List[str]
    ) -> Tuple[Optional[str], List[str]]:
        """
        Predict outcome AND return the reasoning trace.

        This is useful for debugging and understanding how the agent
        arrived at its prediction.

        Args:
            sequence: The sequence to predict

        Returns:
            Tuple of (outcome, trace) where trace is list of reasoning steps

        Example:
            >>> outcome, trace = engine.predict_with_trace(['X', 'Y', 'B'])
            >>> print(outcome)
            'C'
            >>> for step in trace:
            ...     print(step)
            'Trying: X→Y→B'
            'No direct rule for X→Y→B'
            'Split at 2: prefix=X→Y, suffix=B'
            'X→Y ⇒ A (direct rule)'
            'Trying: A→B'
            'A→B ⇒ C (direct rule)'
            'Final prediction: C'
        """
        trace = []
        outcome = self._predict_with_trace_internal(sequence, trace, 0, set())
        return outcome, trace

    def _predict_with_trace_internal(
        self,
        sequence: List[str],
        trace: List[str],
        depth: int,
        visited: Set[Tuple[str, ...]],
    ) -> Optional[str]:
        """Internal helper for predict_with_trace."""
        seq_str = "→".join(sequence)
        trace.append(f"{'  ' * depth}Trying: {seq_str}")

        # Depth and cycle checks
        if depth >= self.max_depth:
            trace.append(f"{'  ' * depth}Max depth reached")
            return None

        seq_tuple = tuple(sequence)
        if seq_tuple in visited:
            trace.append(f"{'  ' * depth}Already visited (cycle)")
            return None
        visited.add(seq_tuple)

        # Base case: Direct rule
        direct_outcome = self.memory.query_rule(sequence)
        if direct_outcome is not None:
            trace.append(f"{'  ' * depth}{seq_str} ⇒ {direct_outcome} (direct rule)")
            return direct_outcome

        trace.append(f"{'  ' * depth}No direct rule for {seq_str}")

        # Recursive case: Chaining
        for split_point in range(1, len(sequence)):
            prefix = sequence[:split_point]
            suffix = sequence[split_point:]

            prefix_str = "→".join(prefix)
            suffix_str = "→".join(suffix)
            trace.append(
                f"{'  ' * depth}Split at {split_point}: prefix={prefix_str}, suffix={suffix_str}"
            )

            intermediate = self._predict_with_trace_internal(
                prefix, trace, depth + 1, visited.copy()
            )

            if intermediate is not None:
                new_sequence = [intermediate] + suffix
                final = self._predict_with_trace_internal(
                    new_sequence, trace, depth + 1, visited.copy()
                )

                if final is not None:
                    trace.append(
                        f"{'  ' * depth}✓ Chain successful: {seq_str} ⇒ {final}"
                    )
                    return final

        trace.append(f"{'  ' * depth}✗ No prediction possible")
        return None

    def can_predict(self, sequence: List[str]) -> bool:
        """
        Check if a prediction is possible (without computing it).

        Args:
            sequence: The sequence to check

        Returns:
            True if prediction is possible, False otherwise
        """
        return self.predict_outcome(sequence) is not None


if __name__ == "__main__":
    # Demo / sanity check
    print("=== SimulationEngine Demo ===\n")

    from memory import CausalMemory

    memory = CausalMemory()
    engine = SimulationEngine(memory)

    # Train the canonical example from design doc
    print("Training rules:")
    print("  A→B ⇒ C (observed 5x)")
    for _ in range(5):
        memory.store_rule(["A", "B"], "C")

    print("  X→Y ⇒ A (observed 5x)")
    for _ in range(5):
        memory.store_rule(["X", "Y"], "A")

    print("\nMemory state:")
    print(memory)

    # Test basic predictions
    print("\n=== Basic Predictions ===")
    test_cases = [
        (["A", "B"], "C"),
        (["X", "Y"], "A"),
    ]

    for sequence, expected in test_cases:
        result = engine.predict_outcome(sequence)
        status = "✓" if result == expected else "✗"
        print(f"{status} {' →'.join(sequence)} ⇒ {result} (expected: {expected})")

    # Test causal chain inference (THE KEY TEST!)
    print("\n=== Causal Chain Inference ===")
    print("Query: X→Y→B ⇒ ?")
    print("Expected reasoning: X→Y⇒A, then A→B⇒C, so result is C\n")

    result, trace = engine.predict_with_trace(["X", "Y", "B"])

    print("Reasoning trace:")
    for step in trace:
        print(step)

    print(f"\n{'='*50}")
    if result == "C":
        print("✓✓✓ SUCCESS! Causal chain reasoning works! ✓✓✓")
        print(f"Predicted: {result} (correct)")
    else:
        print(f"✗ FAILED: Predicted {result}, expected C")
    print(f"{'='*50}")

    # Test false inference prevention
    print("\n=== False Inference Prevention ===")
    print("Query: X→B ⇒ ? (missing Y, should NOT predict C)")
    result = engine.predict_outcome(["X", "B"])
    if result is None:
        print("✓ Correctly returned None (no false inference)")
    else:
        print(f"✗ Incorrectly predicted: {result}")
