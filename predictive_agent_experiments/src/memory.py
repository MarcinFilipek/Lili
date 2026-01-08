"""
CausalMemory - Storage for cause-effect relationships.

This module implements a simple but powerful memory system for storing
and retrieving causal rules. It uses Hebbian-style learning where
repeated observations strengthen associations.
"""

from typing import List, Optional, Dict, Tuple
from collections import defaultdict


class CausalRule:
    """
    Represents a single causal rule: sequence → outcome.

    Attributes:
        sequence: The input sequence (e.g., ['A', 'B'])
        outcome: The resulting outcome (e.g., 'C')
        confidence: Strength of association (Hebbian weight)
        observations: Number of times this rule was observed
    """

    def __init__(self, sequence: List[str], outcome: str):
        self.sequence = tuple(sequence)  # Immutable for hashing
        self.outcome = outcome
        self.confidence = 1.0
        self.observations = 1

    def strengthen(self):
        """Hebbian learning: strengthen rule with repeated observation."""
        self.observations += 1
        # Simple linear strengthening (could be more sophisticated)
        self.confidence = min(10.0, self.confidence + 0.5)

    def __repr__(self):
        seq_str = "→".join(self.sequence)
        return f"Rule({seq_str} ⇒ {self.outcome}, conf={self.confidence:.1f}, n={self.observations})"


class CausalMemory:
    """
    Memory system for storing and retrieving causal rules.

    This implements a bio-inspired associative memory where:
    - Rules are strengthened through repetition (Hebbian learning)
    - Lookup is fast (O(1) dictionary-based)
    - Can handle multiple rules with same sequence (branching outcomes)

    Example:
        >>> memory = CausalMemory()
        >>> memory.store_rule(['A', 'B'], 'C')
        >>> memory.query_rule(['A', 'B'])
        'C'
    """

    def __init__(self):
        # Main storage: sequence tuple → CausalRule
        self._rules: Dict[Tuple[str, ...], CausalRule] = {}

        # Index for partial matching (future optimization)
        # Maps first element → list of rules starting with it
        self._prefix_index: Dict[str, List[CausalRule]] = defaultdict(list)

    def store_rule(self, sequence: List[str], outcome: str) -> CausalRule:
        """
        Store or strengthen a causal rule.

        If the rule already exists, it's strengthened (Hebbian learning).
        Otherwise, a new rule is created.

        Args:
            sequence: The action/stimulus sequence (e.g., ['A', 'B'])
            outcome: The resulting state/effect (e.g., 'C')

        Returns:
            The stored/strengthened CausalRule

        Example:
            >>> memory = CausalMemory()
            >>> rule = memory.store_rule(['X', 'Y'], 'A')
            >>> rule.confidence
            1.0
            >>> memory.store_rule(['X', 'Y'], 'A')  # Strengthen
            >>> rule.confidence
            1.5
        """
        key = tuple(sequence)

        if key in self._rules:
            # Rule exists - strengthen it (Hebbian learning)
            rule = self._rules[key]
            rule.strengthen()
        else:
            # New rule - create and index it
            rule = CausalRule(sequence, outcome)
            self._rules[key] = rule

            # Add to prefix index for fast partial matching
            if sequence:
                self._prefix_index[sequence[0]].append(rule)

        return rule

    def query_rule(self, sequence: List[str]) -> Optional[str]:
        """
        Query the outcome for a given sequence.

        Args:
            sequence: The action/stimulus sequence to query

        Returns:
            The predicted outcome, or None if no rule matches

        Example:
            >>> memory = CausalMemory()
            >>> memory.store_rule(['A', 'B'], 'C')
            >>> memory.query_rule(['A', 'B'])
            'C'
            >>> memory.query_rule(['X', 'Y'])
            None
        """
        key = tuple(sequence)
        rule = self._rules.get(key)
        return rule.outcome if rule else None

    def get_rule(self, sequence: List[str]) -> Optional[CausalRule]:
        """
        Get the full CausalRule object (not just outcome).

        Useful for inspecting confidence, observation count, etc.

        Args:
            sequence: The sequence to look up

        Returns:
            The CausalRule object, or None if not found
        """
        key = tuple(sequence)
        return self._rules.get(key)

    def get_rules_starting_with(self, prefix: str) -> List[CausalRule]:
        """
        Get all rules that start with a given element.

        This is useful for the simulation engine to find applicable rules.

        Args:
            prefix: The first element to match

        Returns:
            List of rules starting with prefix

        Example:
            >>> memory = CausalMemory()
            >>> memory.store_rule(['A', 'B'], 'C')
            >>> memory.store_rule(['A', 'D'], 'E')
            >>> rules = memory.get_rules_starting_with('A')
            >>> len(rules)
            2
        """
        return self._prefix_index.get(prefix, [])

    def clear(self):
        """Reset the memory (forget all rules)."""
        self._rules.clear()
        self._prefix_index.clear()

    def __len__(self) -> int:
        """Return the number of stored rules."""
        return len(self._rules)

    def __repr__(self):
        return f"CausalMemory({len(self)} rules)"

    def __str__(self):
        """Pretty-print all rules for debugging."""
        if not self._rules:
            return "CausalMemory(empty)"

        lines = [f"CausalMemory with {len(self)} rules:"]
        for rule in sorted(
            self._rules.values(), key=lambda r: (-r.confidence, r.sequence)
        ):
            lines.append(f"  {rule}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo / sanity check
    print("=== CausalMemory Demo ===\n")

    memory = CausalMemory()

    # Learn some rules
    print("Training: A→B ⇒ C")
    for i in range(3):
        memory.store_rule(["A", "B"], "C")

    print("Training: X→Y ⇒ A")
    for i in range(2):
        memory.store_rule(["X", "Y"], "A")

    print("\nMemory state:")
    print(memory)

    # Query rules
    print("\n=== Queries ===")
    print(f"A→B predicts: {memory.query_rule(['A', 'B'])}")
    print(f"X→Y predicts: {memory.query_rule(['X', 'Y'])}")
    print(f"Z→W predicts: {memory.query_rule(['Z', 'W'])}")

    # Inspect confidence
    print("\n=== Rule Details ===")
    rule_ab = memory.get_rule(["A", "B"])
    print(f"A→B rule: {rule_ab}")

    # Prefix search
    print("\n=== Prefix Search ===")
    rules_starting_a = memory.get_rules_starting_with("A")
    print(f"Rules starting with 'A': {rules_starting_a}")
