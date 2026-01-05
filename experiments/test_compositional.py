"""
FULL COMPOSITIONAL TEST - WORKING VERSION

Based on test_multiprocess_original.py (which works!)
Added full compositional analysis
"""

import numpy as np
import multiprocessing as mp
import time
import threading


# ============================================================================
# RHYTHM CLASSES (in agent process)
# ============================================================================


class BrainRhythm:
    """Rhythm oscillator"""

    def __init__(self, freq=0):
        self.target_freq = freq
        self.current_freq = freq
        self.phase = 0
        self.value = 0
        self._started = False

    def start(self, freq):
        self.target_freq = freq
        self.current_freq = freq
        self._started = True

    def is_started(self):
        return self._started

    def update(self):
        if self._started:
            dt = 0.001
            self.phase += 2 * np.pi * self.current_freq * dt
            self.phase = self.phase % (2 * np.pi)
            self.value = np.sin(self.phase)


def update_brain_state_loop(rhythms):
    """Background thread for rhythms"""
    while True:
        for name, rhythm in rhythms.items():
            rhythm.update()
        time.sleep(0.001)


def start_global_rhythms(rhythms):
    """Start rhythm thread"""
    print("[RHYTHMS] Starting rhythm thread...")
    t = threading.Thread(target=update_brain_state_loop, args=(rhythms,), daemon=True)
    t.start()
    time.sleep(0.1)


# ============================================================================
# AUDIO GENERATOR
# ============================================================================


def audio_generator_process(audio_queue, sample_rate=16000, hop_size=400):
    """Generate test sequences"""
    print("[GEN] Starting...")

    def tone(freq, dur_ms=200):
        n = int(sample_rate * dur_ms / 1000)
        t = np.linspace(0, dur_ms / 1000, n)
        env = np.ones(n)
        fade = int(0.01 * sample_rate)
        if n > 2 * fade:
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
        return (0.3 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

    def seq(freqs):
        parts = []
        for f in freqs:
            parts.append(tone(f, 200))
            parts.append(np.zeros(int(sample_rate * 0.2)))  # 200ms pause (was 100ms)
        parts.append(np.zeros(int(sample_rate * 2.0)))  # 2.0s pause (was 1.5s)
        return np.concatenate(parts)

    sequences = [
        ("A-B-C #1", seq([300, 1000, 2500])),
        ("A-B-C #2", seq([300, 1000, 2500])),
        ("A-B-C #3", seq([300, 1000, 2500])),
        ("A-B-C #4", seq([300, 1000, 2500])),  # EXTRA
        ("A-B-C #5", seq([300, 1000, 2500])),  # EXTRA
        ("A-B-C #6", seq([300, 1000, 2500])),  # EXTRA
        ("A-B-C #7", seq([300, 1000, 2500])),  # EXTRA
        ("X-B-Z", seq([600, 1000, 3500])),
    ]

    hop_duration = hop_size / sample_rate

    for name, audio in sequences:
        print(f"[GEN] {name}...")
        for i in range(0, len(audio) - hop_size, hop_size):
            hop = audio[i : i + hop_size]
            audio_queue.put(hop)
            time.sleep(hop_duration)
        print(f"[GEN] ✓ {name}")

    audio_queue.put(None)
    print("[GEN] Done")


# ============================================================================
# AGENT WITH FULL COMPOSITIONAL ANALYSIS
# ============================================================================


def agent_process(audio_queue):
    """Agent process with full analysis"""
    print("[AGENT] Starting...")

    import sys

    sys.path.insert(0, "/home/claude")

    # === SETUP RHYTHMS IN THIS PROCESS ===
    import types

    # Create local RHYTHMS
    RHYTHMS = {
        "theta": BrainRhythm(0),
        "delta": BrainRhythm(0),
    }

    # Create brain_rythm module
    brain_rythm_module = types.ModuleType("brain_rythm")
    brain_rythm_module.RHYTHMS = RHYTHMS
    brain_rythm_module.BrainRhythm = BrainRhythm
    sys.modules["brain_rythm"] = brain_rythm_module

    # Start rhythms IN THIS PROCESS
    start_global_rhythms(RHYTHMS)

    RHYTHMS["theta"].start(4.0)
    RHYTHMS["delta"].start(0.5)  # 0.5 Hz = 2 second period (was 1.0 Hz)
    print("[AGENT] Rhythms started: theta=4Hz, delta=0.5Hz")

    time.sleep(0.5)  # Stabilize

    # Import agent
    from bio_agent_6_rythms import BioAgent6

    agent = BioAgent6(
        echoic_memory_size=20, working_memory_size=7, hop_size=400, rhythms=RHYTHMS
    )

    print("[AGENT] Ready\n")

    # === TRACKING ===
    hop_count = 0
    theta_count = 0
    delta_count = 0
    snapshots = []  # Store WM snapshots after each delta

    # === MAIN LOOP ===
    while True:
        hop = audio_queue.get()
        if hop is None:
            break

        # Debug rhythms
        if hop_count % 100 == 0:
            print(
                f"[AGENT] Hop {hop_count}: theta={RHYTHMS['theta'].value:.2f}, delta={RHYTHMS['delta'].value:.2f}"
            )

        # Process
        S_current = agent.hear(hop)
        agent.process_gamma_frequency(hop)

        # Theta
        prev_theta = agent._theta_processed
        agent.process_theta_frame()
        if not prev_theta and agent._theta_processed:
            theta_count += 1

            # Check what was stored
            last_vec = agent.working_memory[-1]
            if np.any(last_vec):
                active = list(np.where(last_vec > 0.1)[0])
                if active:
                    dom = max(active, key=lambda i: last_vec[i])
                    print(
                        f"[AGENT] Theta #{theta_count} @ {hop_count * 0.025:.2f}s: stored vector, DOM={dom}"
                    )

        # Delta - CAPTURE BEFORE PROCESSING
        prev_delta = agent._delta_processed

        # Check if delta will trigger
        if RHYTHMS["delta"].value > 0.95 and not prev_delta:
            # Delta is about to trigger - CAPTURE NOW before WM is cleared!
            wm_snap_before = []
            for i in range(agent.working_memory_size):
                vec = agent.working_memory[i]
                if np.any(vec):
                    wm_snap_before.append(vec.copy())

            if len(wm_snap_before) > 0:
                print(
                    f"[AGENT] WM before delta @ {hop_count * 0.025:.2f}s: {len(wm_snap_before)} vectors"
                )
        else:
            wm_snap_before = None

        # Now process delta (this will clear WM)
        agent.process_delta_frequency()

        if not prev_delta and agent._delta_processed:
            delta_count += 1

            # Use snapshot we captured BEFORE
            if wm_snap_before is not None and len(wm_snap_before) > 0:
                snapshots.append((hop_count * 0.025, wm_snap_before))
                print(f"[AGENT] ✓ Snapshot saved: {len(wm_snap_before)} vectors")
            else:
                print(f"[AGENT] ✗ No snapshot (WM was empty)")

        hop_count += 1

    # ========================================================================
    # COMPOSITIONAL ANALYSIS
    # ========================================================================

    print("\n" + "=" * 70)
    print("COMPOSITIONAL ANALYSIS")
    print("=" * 70)

    print(f"\nProcessed {hop_count} hops ({hop_count * 0.025:.2f}s)")
    print(f"Theta triggers: {theta_count}")
    print(f"Delta triggers: {delta_count}")

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\nLTM weight norm: {W_norm:.3f}")

    if W_norm < 0.2:
        print("✗ LTM too weak - system didn't learn enough")
        print("=" * 70)
        return

    print("✓ LTM learned!")

    # === ANALYZE SNAPSHOTS ===
    print(f"\n--- Captured {len(snapshots)} snapshots ---")

    if len(snapshots) < 2:
        print("✗ Not enough snapshots to analyze")
        print("  Need at least 2 (for A-B-C and X-B-Z)")
        print("=" * 70)
        return

    # Show all snapshots
    for i, (t, snap) in enumerate(snapshots):
        print(f"  Snapshot {i+1} @ {t:.2f}s: {len(snap)} vectors")

    # Find snapshots with 3 vectors
    good_snapshots = [(t, snap) for t, snap in snapshots if len(snap) >= 3]

    if len(good_snapshots) < 2:
        print(f"\n✗ Only {len(good_snapshots)} snapshots have ≥3 vectors")
        print("  Need at least 2 for A-B-C and X-B-Z comparison")
        print("=" * 70)
        return

    # Use last 2 good snapshots (likely A-B-C #3 and X-B-Z)
    abc_time, abc_snap = good_snapshots[-2]
    xbz_time, xbz_snap = good_snapshots[-1]

    print(f"\nSelected for analysis:")
    print(f"  A-B-C snapshot @ {abc_time:.2f}s: {len(abc_snap)} vectors")
    print(f"  X-B-Z snapshot @ {xbz_time:.2f}s: {len(xbz_snap)} vectors")

    # Extract vectors
    vec_A = abc_snap[0]
    vec_B1 = abc_snap[1]
    vec_C = abc_snap[2]

    vec_X = xbz_snap[0]
    vec_B2 = xbz_snap[1]
    vec_Z = xbz_snap[2]

    # === HELPER FUNCTIONS ===
    def compare(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return np.dot(v1, v2) / (n1 * n2)

    def signature(vec, label=""):
        active = list(np.where(vec > 0.1)[0])
        if active:
            dom = max(active, key=lambda i: vec[i])
            print(
                f"{label}: bands {active[:8]}{'...' if len(active)>8 else ''}, DOM={dom}"
            )
        else:
            print(f"{label}: EMPTY")

    # === VECTOR SIGNATURES ===
    print("\n--- Vector Signatures ---")
    signature(vec_A, "  A (300Hz)")
    signature(vec_B1, "  B (1000Hz) from A-B-C")
    signature(vec_C, "  C (2500Hz)")
    signature(vec_X, "  X (600Hz)")
    signature(vec_B2, "  B (1000Hz) from X-B-Z")
    signature(vec_Z, "  Z (3500Hz)")

    # === B RECOGNITION ===
    sim_B = compare(vec_B1, vec_B2)
    print(f"\n--- B Recognition ---")
    print(f"B similarity (A-B-C vs X-B-Z): {sim_B:.3f}")

    if sim_B > 0.7:
        print("✓ B recognized as same element!")
    else:
        print(f"⚠ B similarity low ({sim_B:.2f}) - may be distinct representations")

    # === DISTINCTIVENESS CHECK ===
    print(f"\n--- Distinctiveness Check ---")
    cross_sims = {
        "A-C": compare(vec_A, vec_C),
        "A-X": compare(vec_A, vec_X),
        "C-Z": compare(vec_C, vec_Z),
        "X-Z": compare(vec_X, vec_Z),
    }

    for pair, sim in cross_sims.items():
        status = "✓ distinct" if sim < 0.5 else "✗ similar"
        print(f"  {pair}: {sim:.3f} {status}")

    avg_cross = np.mean(list(cross_sims.values()))
    print(f"\n  Average cross-similarity: {avg_cross:.3f}")

    if avg_cross < 0.5:
        print("  ✓ Vectors are distinct")
    else:
        print("  ✗ Vectors too similar (poor frequency resolution)")

    # ========================================================================
    # GENERATION TEST
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERATION TEST: X → ? → ?")
    print("=" * 70)

    if not hasattr(agent.hebbian_memory, "generate"):
        print("\n⚠ HebbianMemory doesn't have generate() method")
        print("   Add generate() to test compositional generation")
        print("=" * 70)
        return

    # Generate from X
    try:
        generated = agent.hebbian_memory.generate(
            vec_X, n_steps=2, exploration=0.0  # Deterministic
        )

        print(f"\nGenerated {len(generated)} vectors:")
        for i, vec in enumerate(generated):
            signature(vec, f"  Step {i}")

        # === EVALUATION ===
        if len(generated) >= 3:
            gen_1 = generated[1]
            gen_2 = generated[2]

            sim_1_B = compare(gen_1, vec_B1)
            sim_2_C = compare(gen_2, vec_C)
            sim_2_Z = compare(gen_2, vec_Z)

            print(f"\n--- Evaluation ---")
            print(f"Step 1 vs B: {sim_1_B:.3f}", "✓" if sim_1_B > 0.5 else "✗")

            print(f"\nStep 2:")
            print(f"  vs C: {sim_2_C:.3f}")
            print(f"  vs Z: {sim_2_Z:.3f}")
            print(f"  Difference: {abs(sim_2_C - sim_2_Z):.3f}")

            # === FINAL VERDICT ===
            if sim_2_C > sim_2_Z + 0.15:
                print("\n" + "🎉" * 35)
                print("✓✓✓ COMPOSITIONAL LEARNING!")
                print("    Generated X-B-C (novel combination)")
                print("🎉" * 35)
            elif sim_2_Z > sim_2_C + 0.15:
                print("\n✗ Repetition: X-B-Z")
                print("    System repeated training instead of composing")
            else:
                print("\n? Ambiguous")
                print(f"    Difference {abs(sim_2_C - sim_2_Z):.3f} < 0.15 threshold")
                print("    System cannot reliably distinguish C from Z")
                print("\n    Possible causes:")
                print("    - Insufficient frequency resolution")
                print("    - Need more training (5x A-B-C vs 1x X-B-Z)")
                print("    - Hebbian weights too weak")

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")

    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("FULL COMPOSITIONAL TEST")
    print("=" * 70)
    print("\nTesting: A-B-C (7x) + X-B-Z (1x) → X-B-C?")
    print("Strong training ratio (7:1) to overcome recency bias")
    print()

    audio_queue = mp.Queue(maxsize=10)

    generator = mp.Process(target=audio_generator_process, args=(audio_queue,))

    agent = mp.Process(target=agent_process, args=(audio_queue,))

    print("[MAIN] Starting processes...\n")

    generator.start()
    agent.start()

    generator.join()
    agent.join()

    print("\n[MAIN] Test complete.")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
