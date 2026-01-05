"""
Hebbian Memory with Winner-Take-All retrieval

Instead of returning blended response, returns the most similar
stored pattern (like episodic memory retrieval)
"""

import numpy as np


class HebbianMemoryWTA:
    """
    Hebbian Memory with Winner-Take-All (WTA) retrieval

    Stores sequence patterns and retrieves by nearest-neighbor matching
    More biologically plausible: recall specific episodes, not blends
    """

    def __init__(self, dim=32):
        self.dim = dim

        # Dual-trace weights (same as before)
        self.W_fast = np.zeros((dim, dim))
        self.W_slow = np.zeros((dim, dim))
        self.W = np.zeros((dim, dim))

        # NEW: Store actual patterns for WTA retrieval
        self.stored_patterns = []  # List of vectors
        self.pattern_counts = []  # How many times each was seen

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            return vec / norm
        return vec

    def learn_pair(self, vec_a, vec_b, lr_fast=0.1, lr_slow=0.001):
        """Hebbian learning (same as before)"""
        vec_a = self._normalize(vec_a)
        vec_b = self._normalize(vec_b)

        delta = np.outer(vec_a, vec_b)

        self.W_fast += lr_fast * delta
        self.W_slow += lr_slow * delta

        self.W_fast *= 0.9
        self.W_slow *= 0.999

        self.W = self.W_fast + self.W_slow

        # NEW: Store vec_b as pattern
        self._store_pattern(vec_b)

    def _store_pattern(self, vec):
        """Store pattern for WTA retrieval"""
        vec = self._normalize(vec)

        # Check if similar pattern already stored
        for i, stored in enumerate(self.stored_patterns):
            sim = np.dot(vec, stored)
            if sim > 0.9:  # Very similar - just increment count
                self.pattern_counts[i] += 1
                return

        # New pattern - add it
        self.stored_patterns.append(vec.copy())
        self.pattern_counts.append(1)

    def learn_sequence(self, vectors):
        """Learn sequence (same as before)"""
        if len(vectors) < 2:
            return

        for i in range(len(vectors) - 1):
            vec_a = vectors[i]
            vec_b = vectors[i + 1]

            if not np.any(vec_a) or not np.any(vec_b):
                continue

            self.learn_pair(vec_a, vec_b)
            self.learn_pair(vec_b, vec_a)  # bidirectional

    def query_wta(self, vec_current, context_vecs=None, alpha=0.3):
        """
        Winner-Take-All query: return most similar stored pattern

        This is more like episodic memory - recall specific patterns,
        not blended combinations
        """
        vec_current = self._normalize(vec_current)

        # Compute activation for each stored pattern
        activations = []

        for i, pattern in enumerate(self.stored_patterns):
            # Base activation: how associated is pattern with current?
            base_act = np.dot(self.W @ vec_current, pattern)

            # Context boost
            context_act = 0.0
            if context_vecs is not None and len(context_vecs) > 0:
                for ctx_vec in context_vecs:
                    if np.any(ctx_vec):
                        ctx_vec = self._normalize(ctx_vec)
                        context_act += np.dot(self.W @ ctx_vec, pattern)
                context_act /= len(context_vecs)

            # Frequency boost (patterns seen more often = more likely)
            freq_boost = np.log1p(self.pattern_counts[i]) / 10.0

            # Total activation
            total = base_act + alpha * context_act + freq_boost
            activations.append(total)

        # Winner-take-all: return most activated pattern
        if len(activations) > 0:
            winner_idx = np.argmax(activations)
            return self.stored_patterns[winner_idx].copy()

        # Fallback: return normalized response
        return self._normalize(self.W @ vec_current)

    def generate(self, seed_vector, n_steps=5, exploration=0.1, context_window=3):
        """
        Generate sequence using WTA retrieval
        """
        sequence = [seed_vector.copy()]
        current = seed_vector.copy()

        for step in range(n_steps):
            # Context
            context = sequence[-context_window:] if len(sequence) > 1 else None

            # WTA query
            next_vec = self.query_wta(current, context_vecs=context, alpha=0.3)

            # Exploration (add noise)
            if exploration > 0:
                noise = np.random.randn(self.dim) * exploration
                next_vec = next_vec + noise
                next_vec = self._normalize(next_vec)

            sequence.append(next_vec)
            current = next_vec

        return sequence

    def get_stats(self):
        """Stats for debugging"""
        return {
            "W_norm": np.linalg.norm(self.W),
            "W_fast_norm": np.linalg.norm(self.W_fast),
            "W_slow_norm": np.linalg.norm(self.W_slow),
            "n_stored_patterns": len(self.stored_patterns),
            "total_pattern_observations": sum(self.pattern_counts),
        }
