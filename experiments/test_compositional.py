"""
FINAL FIX:
- Delta phase starts at -π/2 (first peak after full cycle)
- Capture WM snapshot BEFORE last delta consolidation
"""

import numpy as np
from bio_agent_5 import BioAgent5, RHYTHMS, BRAIN_STATE

FS = 16000
HOP_MS = 25
HOP_DURATION_SEC = HOP_MS / 1000.0


def generate_tone(frequency, duration_ms=200, amplitude=0.3):
    n_samples = int(FS * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)

    envelope = np.ones(n_samples)
    fade_len = int(0.01 * FS)
    if n_samples > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    tone = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return tone.astype(np.float32)


def manual_update_rhythms(dt):
    """Manualnie update rhythms"""
    for rhythm in RHYTHMS.values():
        if rhythm.target_freq > 0:
            rhythm.phase += 2 * np.pi * rhythm.target_freq * dt
            rhythm.phase = rhythm.phase % (2 * np.pi)
            rhythm.value = np.sin(rhythm.phase)

    BRAIN_STATE["theta"] = RHYTHMS["theta"].value
    BRAIN_STATE["delta"] = RHYTHMS["delta"].value


def process_signal_smart(agent, signal, hop_size, label=""):
    """
    Process signal and capture WM at the RIGHT moment
    (before last delta consolidation clears it)
    """
    n_hops = len(signal) // hop_size

    print(f"\nProcessing {label}...")
    print(f"  Duration: {len(signal)/FS:.2f}s")

    delta_count = 0
    last_wm_snapshot = None

    for hop_idx in range(n_hops):
        samples = signal[hop_idx * hop_size : (hop_idx + 1) * hop_size]

        prev_delta = BRAIN_STATE.get("delta", -1)

        manual_update_rhythms(HOP_DURATION_SEC)

        S_current = agent.hear(samples)
        agent.process_gamma_frequency(S_current)
        agent.process_theta_frame()

        # BEFORE delta processing - capture WM if it has content
        curr_delta = BRAIN_STATE["delta"]
        if prev_delta < 0.95 and curr_delta >= 0.95:
            # Delta peak incoming! Save WM NOW
            wm_vecs = []
            for i in range(agent.working_memory_size):
                vec = agent.working_memory[i]
                if np.any(vec):
                    wm_vecs.append(vec.copy())

            if len(wm_vecs) > 0:
                last_wm_snapshot = wm_vecs
                delta_count += 1
                print(
                    f"    [{hop_idx * HOP_DURATION_SEC:.2f}s] Delta peak #{delta_count} - captured {len(wm_vecs)} vectors"
                )

        # NOW process delta (which may clear WM)
        agent.process_delta_frequency()

    return last_wm_snapshot if last_wm_snapshot else []


def compare_vectors(vec1, vec2):
    """Compare two vectors - return cosine similarity"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def main():
    print("=" * 70)
    print("COMPOSITIONAL TEST (FINAL): A-B-C + X-B-Z → X-B-C?")
    print("=" * 70)

    hop_size = int(FS * HOP_MS / 1000)

    # Setup rhythms
    print("\n[1/6] Setting up rhythms...")
    RHYTHMS["theta"].target_freq = 4.0
    RHYTHMS["theta"].current_freq = 4.0
    RHYTHMS["theta"].phase = 0.0

    # Delta with phase offset to avoid immediate peak
    RHYTHMS["delta"].target_freq = 0.5
    RHYTHMS["delta"].current_freq = 0.5
    RHYTHMS["delta"].phase = -np.pi / 2  # ← START HERE! Peak after 2s

    print(f"  Theta: {RHYTHMS['theta'].target_freq} Hz")
    print(f"  Delta: {RHYTHMS['delta'].target_freq} Hz")
    print(f"  Delta first peak at: ~{2.0}s")

    # Agent
    print("[2/6] Creating agent...")
    agent = BioAgent5(echoic_memory_size=20, working_memory_size=7)

    # === SEQUENCES ===
    print("\n[3/6] Generating sequences...")

    tone_duration = 200
    pause_short = 100
    pause_long = 1000  # 1s pause (delta will fire during this)

    # Sequence 1: A-B-C
    seq1 = np.concatenate(
        [
            generate_tone(400, tone_duration),
            np.zeros(int(FS * pause_short / 1000)),
            generate_tone(600, tone_duration),
            np.zeros(int(FS * pause_short / 1000)),
            generate_tone(800, tone_duration),
            np.zeros(int(FS * pause_long / 1000)),
        ]
    )

    # Sequence 2: X-B-Z
    seq2 = np.concatenate(
        [
            generate_tone(500, tone_duration),
            np.zeros(int(FS * pause_short / 1000)),
            generate_tone(600, tone_duration),
            np.zeros(int(FS * pause_short / 1000)),
            generate_tone(900, tone_duration),
            np.zeros(int(FS * pause_long / 1000)),
        ]
    )

    print(f"  A(400) - B(600) - C(800)")
    print(f"  X(500) - B(600) - Z(900)")
    print(f"  Each sequence: {len(seq1)/FS:.2f}s")

    # === TRAINING ===
    print("\n[4/6] Training...")

    snapshot_seq1 = process_signal_smart(agent, seq1, hop_size, "Seq 1 (A-B-C)")
    print(f"  → Captured: {len(snapshot_seq1)} vectors")

    snapshot_seq2 = process_signal_smart(agent, seq2, hop_size, "Seq 2 (X-B-Z)")
    print(f"  → Captured: {len(snapshot_seq2)} vectors")

    # === ANALYSIS ===
    print("\n[5/6] Analysis...")

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\nLTM: W norm = {W_norm:.4f}")

    if W_norm > 0.1:
        print("  ✓ LTM is learning")

    if len(snapshot_seq1) >= 3 and len(snapshot_seq2) >= 3:
        print(f"\n✓ Both sequences captured!")
        print(f"  Seq1: {len(snapshot_seq1)} vectors")
        print(f"  Seq2: {len(snapshot_seq2)} vectors")

        # Extract vectors (assume order: A,B,C and X,B,Z)
        vec_A = snapshot_seq1[0]
        vec_B1 = snapshot_seq1[1]
        vec_C = snapshot_seq1[2]

        vec_X = snapshot_seq2[0]
        vec_B2 = snapshot_seq2[1]
        vec_Z = snapshot_seq2[2]

        # Check B similarity
        sim_B = compare_vectors(vec_B1, vec_B2)
        print(f"\nB similarity (seq1 vs seq2): {sim_B:.4f}")
        if sim_B > 0.6:
            print("  ✓ B recognized as same!")
        else:
            print(f"  ⚠ B differs (sim={sim_B:.2f})")

        # Dominant bands
        print(f"\nDominant bands:")
        print(
            f"  A: {np.argmax(vec_A):2d}  B: {np.argmax(vec_B1):2d}  C: {np.argmax(vec_C):2d}"
        )
        print(
            f"  X: {np.argmax(vec_X):2d}  B: {np.argmax(vec_B2):2d}  Z: {np.argmax(vec_Z):2d}"
        )

        # === GENERATION ===
        print("\n[6/6] Generation test: X → ? → ?")

        generated = agent.hebbian_memory.generate(
            vec_X, n_steps=2, exploration=0.2, context_window=3
        )

        print(f"\nGenerated {len(generated)} vectors:")
        for i, vec in enumerate(generated):
            print(f"  Step {i}: band {np.argmax(vec):2d}")

        # === EVALUATION ===
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        if len(generated) >= 3:
            gen_1 = generated[1]
            gen_2 = generated[2]

            sim_1_B = compare_vectors(gen_1, vec_B1)
            sim_2_C = compare_vectors(gen_2, vec_C)
            sim_2_Z = compare_vectors(gen_2, vec_Z)

            print(f"\nStep 1 vs B: {sim_1_B:.3f}", "✓" if sim_1_B > 0.5 else "✗")
            print(f"\nStep 2:")
            print(f"  vs C: {sim_2_C:.3f}")
            print(f"  vs Z: {sim_2_Z:.3f}")

            if sim_2_C > sim_2_Z and sim_2_C > 0.35:
                print("\n✓✓ COMPOSITIONAL! Generated X-B-C")
            elif sim_2_Z > sim_2_C:
                print("\n○ Repetition: X-B-Z")
            else:
                print("\n? Novel/ambiguous")

        print("=" * 70)

    else:
        print(f"\n✗ Insufficient vectors")
        print(f"  Seq1: {len(snapshot_seq1)}, Seq2: {len(snapshot_seq2)}")
        print("\nDEBUG: Try manually inspecting agent.working_memory during processing")


if __name__ == "__main__":
    main()
