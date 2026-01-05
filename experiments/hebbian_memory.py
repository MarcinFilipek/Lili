"""
Hebbian Memory - biologicznie inspirowana pamięć asocjacyjna
Używa ciągłych wektorów (bez dyskretyzacji) i Hebbian learning

OPTIMIZED VERSION (Jan 5, 2026)
================================
Key parameters found through systematic testing:
- lr_slow = 0.01 (10x increase from 0.001) - CRITICAL for slow trace accumulation
- decay_fast = 0.98 (reduced from 0.95) - Less aggressive decay
- decay_slow = 0.99 (increased from 0.9998) - Allows forgetting old patterns

These parameters enable compositional learning while exhibiting biological
primacy/recency effects.
"""

import numpy as np


class HebbianMemory:
    """
    Biologiczna pamięć asocjacyjna oparta na regule Hebbiana:
    "Neurons that fire together, wire together"

    Dual-trace architecture:
    - Fast trace: Rapid learning, moderate decay (captures novelty)
    - Slow trace: Gradual learning, slow decay (builds expertise through repetition)

    Przechowuje skojarzenia między wektorami aktywacji (32D)
    bez dyskretyzacji na chunk IDs.
    """

    def __init__(self, dim=32):
        """
        Args:
            dim: wymiarowość wektorów (liczba frequency bands)
        """
        self.dim = dim

        # Dual-trace learning (STM + LTM na tych samych synapsach)
        self.W_fast = np.zeros(
            (dim, dim)
        )  # Short-term (szybkie uczenie, średnie zapominanie)
        self.W_slow = np.zeros(
            (dim, dim)
        )  # Long-term (średnie uczenie, wolne zapominanie)

        # Effective weight matrix (suma obu śladów)
        self.W = np.zeros((dim, dim))

    def learn_pair(self, vec_a, vec_b, lr_fast=0.1, lr_slow=0.01):
        """
        Ucz skojarzenie między dwoma wektorami (Hebbian rule)

        Args:
            vec_a, vec_b: wektory do skojarzenia
            lr_fast: learning rate dla fast trace (default: 0.1)
            lr_slow: learning rate dla slow trace (default: 0.01)
                     OPTIMIZED: 10x increase from original 0.001
                     This allows slow trace to accumulate effectively
                     over multiple repetitions (e.g., 7x → 0.07 total strength)
        """
        # Normalizacja (neurony mają bounded firing rates)
        vec_a = self._normalize(vec_a)
        vec_b = self._normalize(vec_b)

        # Outer product: każdy neuron z A łączy się z każdym z B
        delta = np.outer(vec_a, vec_b)

        # Dual-trace update
        self.W_fast += lr_fast * delta
        self.W_slow += lr_slow * delta

        # Decay (biologiczne zapominanie)
        # OPTIMIZED PARAMETERS:
        self.W_fast *= 0.98  # 2% decay per step (was 0.95 = 5% decay)
        # Less aggressive to preserve learning
        self.W_slow *= 0.99  # 1% decay per step (was 0.9998 = 0.02% decay)
        # Faster decay allows overwriting old primacy patterns
        # Balance: retain repeated patterns, forget one-shot old ones

        # Effective weight (suma śladów)
        self.W = self.W_fast + self.W_slow

    def learn_sequence(self, vectors):
        """
        Ucz się sekwencji wektorów (A→B→C→...)

        Tworzy skojarzenia między kolejnymi wektorami (pary sąsiadujące)
        oraz bidirectional (A↔B, B↔C)

        Args:
            vectors: lista wektorów (np. z working memory)
        """
        if len(vectors) < 2:
            return

        # Ucz się par sąsiadujących
        for i in range(len(vectors) - 1):
            vec_a = vectors[i]
            vec_b = vectors[i + 1]

            if not np.any(vec_a) or not np.any(vec_b):
                continue

            # Forward: A → B
            self.learn_pair(vec_a, vec_b)

            # Backward: B → A (bidirectional, biologiczne!)
            self.learn_pair(vec_b, vec_a)

    def query(self, vec_current, context_vecs=None, alpha=0.3):
        """
        Aktywuj sieć skojarzeń - "co jest skojarzone z vec_current?"

        Args:
            vec_current: aktualny wektor zapytania
            context_vecs: lista wektorów kontekstu (np. z working memory)
            alpha: siła wpływu kontekstu (0-1)

        Returns:
            Wektor odpowiedzi (następny przewidywany wektor)
        """
        vec_current = self._normalize(vec_current)

        # Bazowa odpowiedź przez macierz wag
        response = self.W @ vec_current

        # Wzmocnienie przez kontekst (jeśli podany)
        if context_vecs is not None and len(context_vecs) > 0:
            context_boost = np.zeros(self.dim)

            for ctx_vec in context_vecs:
                if np.any(ctx_vec):
                    ctx_vec = self._normalize(ctx_vec)
                    context_boost += self.W @ ctx_vec

            # Kombinacja: bazowa odpowiedź + kontekstowe wzmocnienie
            response = response + alpha * context_boost

        return self._normalize(response)

    def generate(self, seed_vector, n_steps=5, exploration=0.1, context_window=3):
        """
        Generuj sekwencję wektorów startując od seed_vector

        To jest kluczowa metoda do testowania kompozycyjności!

        Args:
            seed_vector: wektor początkowy (np. vec_X)
            n_steps: ile kroków wygenerować
            exploration: siła szumu eksploracyjnego (0-1)
            context_window: ile ostatnich wektorów używać jako kontekst

        Returns:
            Lista wektorów: [seed, step1, step2, ...]
        """
        sequence = [seed_vector.copy()]
        current = seed_vector.copy()

        for step in range(n_steps):
            # Kontekst = ostatnie N wektorów z sekwencji
            if len(sequence) > 1:
                context = sequence[-context_window:]
            else:
                context = None

            # Query LTM z kontekstem
            next_vec = self.query(current, context_vecs=context, alpha=0.3)

            # Eksploracja (szum dla discovering nowych kombinacji)
            if exploration > 0:
                noise = np.random.randn(self.dim) * exploration
                next_vec = next_vec + noise
                next_vec = self._normalize(next_vec)

            sequence.append(next_vec)
            current = next_vec

        return sequence

    def _normalize(self, vec):
        """
        Normalizacja wektora (L2 norm)

        Biologiczne: neurony mają ograniczoną firing rate (0-1 lub -1 do 1)
        """
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            return vec / norm
        return vec

    def get_stats(self):
        """Zwraca statystyki pamięci (do debugowania)"""
        return {
            "W_norm": np.linalg.norm(self.W),
            "W_fast_norm": np.linalg.norm(self.W_fast),
            "W_slow_norm": np.linalg.norm(self.W_slow),
            "max_weight": np.max(self.W),
            "non_zero_count": np.count_nonzero(self.W > 0.001),
            "total_weights": self.W.size,
        }
