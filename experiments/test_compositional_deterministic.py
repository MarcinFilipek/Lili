"""
DETERMINISTIC COMPOSITIONAL TEST
=================================

Fully synchronous, single-process test for 100% reproducibility.
No multiprocessing, no timing issues, no race conditions.

All audio generated upfront, processed hop-by-hop deterministically.
"""

import numpy as np
import sys
import time

# Add current directory to path for imports
sys.path.insert(0, "/Users/marcinfilipek/Documents/python_workspace/Lili")

from brain_rythm import BrainRhythm
from hebbian_memory_optimized import HebbianMemory


# ============================================================================
# AUDIO GENERATION (Deterministic)
# ============================================================================


def generate_tone(freq, duration_ms=200, sample_rate=16000):
    """Generate pure tone with envelope"""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)

    # Envelope (fade in/out)
    envelope = np.ones(n_samples)
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    if n_samples > 2 * fade_samples:
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    signal = 0.3 * np.sin(2 * np.pi * freq * t) * envelope
    return signal.astype(np.float32)


def generate_sequence(freqs, sample_rate=16000):
    """Generate sequence of tones with pauses"""
    parts = []
    for freq in freqs:
        # Tone (200ms)
        parts.append(generate_tone(freq, 200, sample_rate))
        # Pause between tones (200ms)
        parts.append(np.zeros(int(sample_rate * 0.2), dtype=np.float32))

    # Long pause after sequence (2.0s)
    parts.append(np.zeros(int(sample_rate * 2.0), dtype=np.float32))

    return np.concatenate(parts)


def generate_all_audio():
    """Generate all training sequences deterministically"""
    sequences = {
        "A-B-C #1": generate_sequence([300, 1000, 2500]),
        "A-B-C #2": generate_sequence([300, 1000, 2500]),
        "A-B-C #3": generate_sequence([300, 1000, 2500]),
        "A-B-C #4": generate_sequence([300, 1000, 2500]),
        "A-B-C #5": generate_sequence([300, 1000, 2500]),
        "A-B-C #6": generate_sequence([300, 1000, 2500]),
        "A-B-C #7": generate_sequence([300, 1000, 2500]),
        "A-B-C #8": generate_sequence([300, 1000, 2500]),
        "A-B-C #9": generate_sequence([300, 1000, 2500]),
        "A-B-C #10": generate_sequence([300, 1000, 2500]),
        "X-B-Z": generate_sequence([600, 1000, 3500]),
        # "A-B-C #11": generate_sequence([600, 1000, 2500]),
        # "A-B-C #12": generate_sequence([600, 1000, 2500]),
        # "A-B-C #13": generate_sequence([600, 1000, 2500]),
        # "A-B-C #14": generate_sequence([600, 1000, 2500]),
        # "A-B-C #15": generate_sequence([600, 1000, 2500]),
        # "A-B-C #16": generate_sequence([600, 1000, 2500]),
        # "A-B-C #17": generate_sequence([600, 1000, 2500]),
        # "A-B-C #18": generate_sequence([600, 1000, 2500]),
        # "A-B-C #19": generate_sequence([600, 1000, 2500]),
        # "A-B-C #20": generate_sequence([600, 1000, 2500]),
        # "A-B-C #21": generate_sequence([600, 1000, 2500]),
        # "A-B-C #22": generate_sequence([600, 1000, 2500]),
        # "A-B-C #23": generate_sequence([600, 1000, 2500]),
        # "A-B-C #24": generate_sequence([600, 1000, 2500]),
        # "A-B-C #25": generate_sequence([600, 1000, 2500]),
        # "A-B-C #26": generate_sequence([600, 1000, 2500]),
        # "A-B-C #27": generate_sequence([600, 1000, 2500]),
        # "A-B-C #28": generate_sequence([600, 1000, 2500]),
        # "A-B-C #29": generate_sequence([600, 1000, 2500]),
        # "A-B-C #30": generate_sequence([600, 1000, 2500]),
        # "A-B-C #31": generate_sequence([600, 1000, 2500]),
        # "A-B-C #32": generate_sequence([600, 1000, 2500]),
        # "A-B-C #33": generate_sequence([600, 1000, 2500]),
        # "A-B-C #34": generate_sequence([600, 1000, 2500]),
        # "A-B-C #35": generate_sequence([600, 1000, 2500]),
        # "A-B-C #36": generate_sequence([600, 1000, 2500]),
        # "A-B-C #37": generate_sequence([600, 1000, 2500]),
        # "A-B-C #38": generate_sequence([600, 1000, 2500]),
        # "A-B-C #39": generate_sequence([600, 1000, 2500]),
        # "A-B-C #40": generate_sequence([600, 1000, 2500]),
        # "A-B-C #41": generate_sequence([600, 1000, 2500]),
        # "A-B-C #42": generate_sequence([600, 1000, 2500]),
        # "A-B-C #43": generate_sequence([600, 1000, 2500]),
        # "A-B-C #44": generate_sequence([600, 1000, 2500]),
        # "A-B-C #45": generate_sequence([600, 1000, 2500]),
        # "A-B-C #46": generate_sequence([600, 1000, 2500]),
        # "A-B-C #47": generate_sequence([600, 1000, 2500]),
        # "A-B-C #48": generate_sequence([600, 1000, 2500]),
        # "A-B-C #49": generate_sequence([600, 1000, 2500]),
        # "A-B-C #50": generate_sequence([600, 1000, 2500]),
        # "A-B-C #51": generate_sequence([600, 1000, 2500]),
        # "A-B-C #52": generate_sequence([600, 1000, 2500]),
        # "A-B-C #53": generate_sequence([600, 1000, 2500]),
        # "A-B-C #54": generate_sequence([600, 1000, 2500]),
        # "A-B-C #55": generate_sequence([600, 1000, 2500]),
        # "A-B-C #56": generate_sequence([600, 1000, 2500]),
        # "A-B-C #57": generate_sequence([600, 1000, 2500]),
        # "A-B-C #58": generate_sequence([600, 1000, 2500]),
        # "A-B-C #59": generate_sequence([600, 1000, 2500]),
        # "A-B-C #60": generate_sequence([600, 1000, 2500]),
    }

    # Concatenate all
    full_audio = np.concatenate(
        [
            sequences[name]
            for name in [
                "A-B-C #1",
                "A-B-C #2",
                "A-B-C #3",
                "A-B-C #4",
                "A-B-C #5",
                "A-B-C #6",
                "A-B-C #7",
                "A-B-C #8",
                "A-B-C #9",
                "A-B-C #10",
                "X-B-Z",
                # "A-B-C #11",
                # "A-B-C #12",
                # "A-B-C #13",
                # "A-B-C #14",
                # "A-B-C #15",
                # "A-B-C #16",
                # "A-B-C #17",
                # "A-B-C #18",
                # "A-B-C #19",
                # "A-B-C #20",
                # "A-B-C #21",
                # "A-B-C #22",
                # "A-B-C #23",
                # "A-B-C #24",
                # "A-B-C #25",
                # "A-B-C #26",
                # "A-B-C #27",
                # "A-B-C #28",
                # "A-B-C #29",
                # "A-B-C #30",
                # "A-B-C #31",
                # "A-B-C #32",
                # "A-B-C #33",
                # "A-B-C #34",
                # "A-B-C #35",
                # "A-B-C #36",
                # "A-B-C #37",
                # "A-B-C #38",
                # "A-B-C #39",
                # "A-B-C #40",
                # "A-B-C #41",
                # "A-B-C #42",
                # "A-B-C #43",
                # "A-B-C #44",
                # "A-B-C #45",
                # "A-B-C #46",
                # "A-B-C #47",
                # "A-B-C #48",
                # "A-B-C #49",
                # "A-B-C #50",
                # "A-B-C #51",
                # "A-B-C #52",
                # "A-B-C #53",
                # "A-B-C #54",
                # "A-B-C #55",
                # "A-B-C #56",
                # "A-B-C #57",
                # "A-B-C #58",
                # "A-B-C #59",
                # "A-B-C #60",
            ]
        ]
    )

    return full_audio, sequences


# ============================================================================
# SIMPLIFIED AGENT (No multiprocessing)
# ============================================================================


class DeterministicAgent:
    """Simplified agent for deterministic testing"""

    def __init__(self, hop_size=400):
        self.n_sensors = 32
        self.hop_size = hop_size
        self.sample_rate = 16000

        # Adaptive threshold
        self.theta = np.zeros(self.n_sensors)

        # Echoic memory (gamma buffer)
        self.echoic_memory_size = 20
        self.echoic_memory = np.zeros((self.echoic_memory_size, self.hop_size))

        # Working memory (theta buffer)
        self.working_memory_size = 7
        self.working_memory = np.zeros((self.working_memory_size, self.n_sensors))

        # Hebbian LTM
        self.hebbian_memory = HebbianMemory(self.n_sensors)

        # Rhythms (local)
        self.theta_rhythm = BrainRhythm(0)
        self.delta_rhythm = BrainRhythm(0)

        # State tracking
        self._theta_processed = True
        self._delta_processed = True

        # Statistics
        self.theta_count = 0
        self.delta_count = 0
        self.snapshots = []

    def start_rhythms(self):
        """Initialize rhythms"""
        # Use empirically determined frequencies from multiprocess tests
        # Multiprocess achieves ~3.5 Hz theta, ~0.8 Hz delta in practice
        self.theta_rhythm.start(3.5)  # was 4.0
        self.delta_rhythm.start(0.8)  # was 0.5

    def update_rhythms(self):
        """Update rhythm phases (called every hop)"""
        # Update multiple times per hop for smoother phase progression
        # hop_duration = 400/16000 = 25ms = 25 updates at 1ms each
        for _ in range(25):
            self.theta_rhythm.update()
            self.delta_rhythm.update()

    def hear(self, frame):
        """Process audio frame through sensory layer"""
        # Ensure frame length
        if len(frame) < 2048:
            frame = np.pad(frame, (0, 2048 - len(frame)))

        # FFT processing
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=2048)
        power = np.abs(spectrum) ** 2

        # Bin into frequency bands
        bins_per_band = len(power) // self.n_sensors
        n_bins = self.n_sensors * bins_per_band
        bands = power[:n_bins].reshape(self.n_sensors, bins_per_band).sum(axis=1)

        # Log compression
        signal = np.log1p(bands)

        # Adaptive thresholding
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)

        # Normalize
        if np.max(activations) > 0:
            activations /= np.max(activations) + 1e-9

        return activations

    def process_gamma(self, frame):
        """Store in echoic memory"""
        self.echoic_memory = np.roll(self.echoic_memory, shift=-1, axis=0)
        self.echoic_memory[-1] = frame

    def process_theta(self):
        """Theta-gated sampling to working memory"""
        val = self.theta_rhythm.value

        if val > 0.95 and not self._theta_processed:
            self._theta_processed = True
            self.theta_count += 1

            # FFT on full echoic buffer (500ms)
            recent_audio = self.echoic_memory.flatten()
            window = np.hamming(len(recent_audio))
            windowed = recent_audio * window
            spectrum = np.fft.rfft(windowed, n=2048)
            power = np.abs(spectrum) ** 2

            # Extract features
            bins_per_band = len(power) // self.n_sensors
            n_bins = self.n_sensors * bins_per_band
            bands = power[:n_bins].reshape(self.n_sensors, bins_per_band).sum(axis=1)
            features = np.log1p(bands)

            # Normalize
            if np.max(features) > 0:
                features /= np.max(features) + 1e-9

            # Store in working memory
            self.working_memory = np.roll(self.working_memory, shift=-1, axis=0)
            self.working_memory[-1] = features

        elif val < 0:
            self._theta_processed = False

    def process_delta(self):
        """Delta-gated consolidation to LTM"""
        val = self.delta_rhythm.value

        # CAPTURE SNAPSHOT BEFORE PROCESSING
        if val > 0.95 and not self._delta_processed:
            # Snapshot WM before clearing
            wm_snapshot = []
            for i in range(self.working_memory_size):
                vec = self.working_memory[i]
                if np.any(vec):
                    wm_snapshot.append(vec.copy())

            if len(wm_snapshot) > 0:
                self.snapshots.append(wm_snapshot)

            # Now process delta
            self._delta_processed = True
            self.delta_count += 1

            # Extract sequence from WM
            sequence = []
            for i in range(self.working_memory_size):
                vec = self.working_memory[i]
                if np.any(vec):
                    sequence.append(vec.copy())

            # Learn sequence
            if len(sequence) > 1:
                self.hebbian_memory.learn_sequence(sequence)

            # Clear WM
            self.working_memory.fill(0)

        elif val < 0:
            self._delta_processed = False

    def process_hop(self, frame):
        """Process single audio hop"""
        # Update rhythms
        self.update_rhythms()

        # Sensory processing
        activations = self.hear(frame)
        self.process_gamma(frame)

        # Memory operations
        self.process_theta()
        self.process_delta()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


def vector_signature(vec, label=""):
    """Print vector signature for debugging"""
    active = list(np.where(vec > 0.1)[0])
    if active:
        dom = max(active, key=lambda i: vec[i])
        bands_str = str(active[:8])
        if len(active) > 8:
            bands_str += "..."
        print(f"{label}: bands {bands_str}, DOM={dom}")
    else:
        print(f"{label}: EMPTY")


# ============================================================================
# MAIN TEST
# ============================================================================


def main():
    print("=" * 70)
    print("DETERMINISTIC COMPOSITIONAL TEST")
    print("=" * 70)
    print("\nNo multiprocessing - fully synchronous and reproducible")
    print("Testing: 7x A-B-C + 1x X-B-Z → X-B-C?\n")

    # Set random seed for numpy (though we don't use random in critical paths)
    np.random.seed(42)

    # Generate all audio upfront
    print("[1/4] Generating audio...")
    full_audio, sequences = generate_all_audio()
    print(f"      Total duration: {len(full_audio)/16000:.1f}s")

    # Create agent
    print("[2/4] Initializing agent...")
    agent = DeterministicAgent(hop_size=400)
    agent.start_rhythms()

    # Warm up rhythms (advance phase so they start in proper position)
    print("      Warming up rhythms...")
    for _ in range(500):  # 500ms warm-up
        agent.theta_rhythm.update()
        agent.delta_rhythm.update()

    # Process audio hop-by-hop
    print("[3/4] Processing audio...")
    hop_size = 400
    hop_count = 0

    start_time = time.time()
    for i in range(0, len(full_audio) - hop_size, hop_size):
        hop = full_audio[i : i + hop_size]
        agent.process_hop(hop)
        hop_count += 1

    elapsed = time.time() - start_time

    print(f"      Processed {hop_count} hops in {elapsed:.2f}s")
    print(f"      Theta triggers: {agent.theta_count}")
    print(f"      Delta triggers: {agent.delta_count}")
    print(f"      Snapshots captured: {len(agent.snapshots)}")

    # Analysis
    print("\n[4/4] Analysis...")
    print("=" * 70)
    print("DIRECT ASSOCIATION ANALYSIS")
    print("=" * 70)

    if len(agent.snapshots) < 2:
        print("\n✗ Insufficient snapshots for analysis")
        print(f"  Got {len(agent.snapshots)}, need at least 2")
        return

    # Find snapshots with 3+ vectors
    good_snapshots = [snap for snap in agent.snapshots if len(snap) >= 3]

    if len(good_snapshots) < 2:
        print(f"\n✗ Only {len(good_snapshots)} snapshots have ≥3 vectors")
        return

    # Use last 2 good snapshots
    abc_snap = good_snapshots[-2]
    xbz_snap = good_snapshots[-1]

    print(f"\nUsing snapshots:")
    print(f"  A-B-C: {len(abc_snap)} vectors")
    print(f"  X-B-Z: {len(xbz_snap)} vectors")

    # Extract vectors
    vec_A = abc_snap[0]
    vec_B1 = abc_snap[1]
    vec_C = abc_snap[2]

    vec_X = xbz_snap[0]
    vec_B2 = xbz_snap[1]
    vec_Z = xbz_snap[2]

    # Vector signatures
    print("\n--- Vector Signatures ---")
    vector_signature(vec_A, "  A (300Hz)")
    vector_signature(vec_B1, "  B (1000Hz) from ABC")
    vector_signature(vec_C, "  C (2500Hz)")
    vector_signature(vec_X, "  X (600Hz)")
    vector_signature(vec_B2, "  B (1000Hz) from XBZ")
    vector_signature(vec_Z, "  Z (3500Hz)")

    # B recognition
    sim_B = cosine_similarity(vec_B1, vec_B2)
    print(f"\n--- B Recognition ---")
    print(f"B similarity (ABC vs XBZ): {sim_B:.3f}")

    if sim_B > 0.7:
        print("✓ B recognized as same element")
    else:
        print(f"⚠ B similarity low ({sim_B:.2f})")

    # Direct associations
    W = agent.hebbian_memory.W

    print("\n--- Test 1: B → ? (Direct Association) ---")
    response_B = W @ vec_B1
    sim_B_to_C = cosine_similarity(response_B, vec_C)
    sim_B_to_Z = cosine_similarity(response_B, vec_Z)
    diff_B = abs(sim_B_to_C - sim_B_to_Z)

    print(f"B → C similarity: {sim_B_to_C:.3f}")
    print(f"B → Z similarity: {sim_B_to_Z:.3f}")
    print(f"Difference: {diff_B:.3f} (threshold: 0.15)")

    if sim_B_to_C > sim_B_to_Z + 0.15:
        print("✓ Hebbian weights favor C (correct!)")
        learning_success = True
    elif sim_B_to_Z > sim_B_to_C + 0.15:
        print("✗ Hebbian weights favor Z (recency bias!)")
        learning_success = False
    else:
        print("? Ambiguous - weights don't distinguish C from Z")
        learning_success = False

    # Weight statistics
    stats = agent.hebbian_memory.get_stats()
    print("\n--- Weight Statistics ---")
    print(f"Total weight norm: {stats['W_norm']:.3f}")
    print(f"Fast trace norm: {stats['W_fast_norm']:.3f}")
    print(f"Slow trace norm: {stats['W_slow_norm']:.3f}")
    print(f"Slow/Fast ratio: {stats['W_slow_norm']/stats['W_fast_norm']*100:.1f}%")

    # Generation test
    print("\n" + "=" * 70)
    print("GENERATION TEST: X → ? → ?")
    print("=" * 70)

    generated = agent.hebbian_memory.generate(
        vec_X, n_steps=2, exploration=0.0, context_window=3
    )

    print(f"\nGenerated {len(generated)} vectors:")
    for i, vec in enumerate(generated):
        vector_signature(vec, f"  Step {i}")

    # Evaluate
    if len(generated) >= 3:
        gen_1 = generated[1]
        gen_2 = generated[2]

        sim_1_B = cosine_similarity(gen_1, vec_B1)
        sim_2_C = cosine_similarity(gen_2, vec_C)
        sim_2_Z = cosine_similarity(gen_2, vec_Z)

        print(f"\n--- Evaluation ---")
        print(f"Step 1 vs B: {sim_1_B:.3f}", "✓" if sim_1_B > 0.5 else "✗")

        print(f"\nStep 2:")
        print(f"  vs C: {sim_2_C:.3f}")
        print(f"  vs Z: {sim_2_Z:.3f}")
        print(f"  Difference: {abs(sim_2_C - sim_2_Z):.3f}")

        # Final verdict
        print("\n" + "=" * 70)
        if sim_2_C > sim_2_Z + 0.15:
            print("🎉" * 35)
            print("✓✓✓ COMPOSITIONAL LEARNING!")
            print("    Generated X-B-C (novel combination)")
            print("🎉" * 35)
            return True
        elif sim_2_Z > sim_2_C + 0.15:
            print("✗ Repetition: X-B-Z")
            print("    System repeated training instead of composing")
            return False
        else:
            print("? Ambiguous")
            print(f"    Difference {abs(sim_2_C - sim_2_Z):.3f} < 0.15 threshold")
            return False

    print("=" * 70)
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
