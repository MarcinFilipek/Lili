"""
DETERMINISTIC TEST - Sparse Coding + No Exploration

Goal: See if system REALLY learns compositional structure
or just gets lucky with noise.
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


def hear_sparse(agent, frame, k_sparse=6):
    """
    Modified hear() with sparse coding
    Only top-K bands are active
    """
    if len(frame) < 2048:
        frame = np.pad(frame, (0, 2048 - len(frame)))

    windowed = frame * np.hamming(len(frame))
    spectrum = np.fft.rfft(windowed, n=2048)
    power = np.abs(spectrum) ** 2
    bins_per_band = len(power) // agent.n_sensors

    n_bins = agent.n_sensors * bins_per_band
    bands = power[:n_bins].reshape(agent.n_sensors, bins_per_band).sum(axis=1)
    signal = np.log1p(bands)

    activations = np.maximum(0, signal - agent.theta)
    agent.theta += 0.01 * (signal - agent.theta)

    # === SPARSE CODING ===
    if np.max(activations) > 0:
        # Keep only top-K
        top_k = np.argpartition(activations, -k_sparse)[-k_sparse:]
        sparse = np.zeros_like(activations)
        sparse[top_k] = activations[top_k]

        # Normalize
        if np.max(sparse) > 0:
            sparse /= np.max(sparse) + 1e-9

        return sparse

    return activations


def manual_update_rhythms(dt):
    for rhythm in RHYTHMS.values():
        if rhythm.target_freq > 0:
            rhythm.phase += 2 * np.pi * rhythm.target_freq * dt
            rhythm.phase = rhythm.phase % (2 * np.pi)
            rhythm.value = np.sin(rhythm.phase)

    BRAIN_STATE["theta"] = RHYTHMS["theta"].value
    BRAIN_STATE["delta"] = RHYTHMS["delta"].value


def process_signal_smart(agent, signal, hop_size, label=""):
    n_hops = len(signal) // hop_size
    print(f"  Processing {label}...")

    last_wm_snapshot = None

    for hop_idx in range(n_hops):
        samples = signal[hop_idx * hop_size : (hop_idx + 1) * hop_size]

        prev_delta = BRAIN_STATE.get("delta", -1)
        manual_update_rhythms(HOP_DURATION_SEC)

        # Use sparse hear()
        S_current = hear_sparse(agent, samples, k_sparse=6)
        agent.process_gamma_frequency(S_current)
        agent.process_theta_frame()

        curr_delta = BRAIN_STATE["delta"]
        if prev_delta < 0.95 and curr_delta >= 0.95:
            wm_vecs = []
            for i in range(agent.working_memory_size):
                vec = agent.working_memory[i]
                if np.any(vec):
                    wm_vecs.append(vec.copy())

            if len(wm_vecs) > 0:
                last_wm_snapshot = wm_vecs

        agent.process_delta_frequency()

    return last_wm_snapshot if last_wm_snapshot else []


def compare_vectors(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def show_vector_signature(vec, label=""):
    """Show which bands are active"""
    active = np.where(vec > 0.01)[0]
    if len(active) > 0:
        print(f"{label}: bands {list(active)} (n={len(active)})")
    else:
        print(f"{label}: NO ACTIVE BANDS")


def main():
    print("=" * 70)
    print("DETERMINISTIC TEST: Sparse Coding + No Exploration")
    print("=" * 70)

    hop_size = int(FS * HOP_MS / 1000)

    # Setup
    print("\n[1] Setup...")
    RHYTHMS["theta"].target_freq = 4.0
    RHYTHMS["theta"].current_freq = 4.0
    RHYTHMS["theta"].phase = 0.0

    RHYTHMS["delta"].target_freq = 0.5
    RHYTHMS["delta"].current_freq = 0.5
    RHYTHMS["delta"].phase = -np.pi / 2

    agent = BioAgent5(echoic_memory_size=20, working_memory_size=7)
    print(f"  n_sensors: {agent.n_sensors}")
    print(f"  Sparse coding: top-6 bands active")

    # Sequences
    print("\n[2] Generating sequences...")

    seq1 = np.concatenate(
        [
            generate_tone(400, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(600, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(800, 200),
            np.zeros(int(FS * 1.0)),
        ]
    )

    seq2 = np.concatenate(
        [
            generate_tone(500, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(600, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(900, 200),
            np.zeros(int(FS * 1.0)),
        ]
    )

    print("  A(400) - B(600) - C(800)")
    print("  X(500) - B(600) - Z(900)")

    # Training with MULTIPLE repetitions
    print("\n[3] Training (multiple repetitions)...")
    print("  Training A-B-C (3x) to strengthen...")

    snap1 = None
    for i in range(3):
        snap = process_signal_smart(agent, seq1, hop_size, f"A-B-C #{i+1}")
        if snap:
            snap1 = snap

    print(f"  → Captured: {len(snap1) if snap1 else 0} vectors")

    print("  Training X-B-Z (1x)...")
    snap2 = process_signal_smart(agent, seq2, hop_size, "X-B-Z")
    print(f"  → Captured: {len(snap2) if snap2 else 0} vectors")

    # Analysis
    print("\n[4] Analysis...")

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\nLTM: W norm = {W_norm:.4f}")

    if W_norm < 0.1:
        print("  ✗ LTM not learning properly!")
        return

    if len(snap1) >= 3 and len(snap2) >= 3:
        print(f"\n✓ Sequences captured ({len(snap1)}, {len(snap2)})")

        vec_A = snap1[0]
        vec_B1 = snap1[1]
        vec_C = snap1[2]

        vec_X = snap2[0]
        vec_B2 = snap2[1]
        vec_Z = snap2[2]

        # Show sparse signatures
        print("\n--- Vector Signatures (active bands) ---")
        show_vector_signature(vec_A, "  A")
        show_vector_signature(vec_B1, "  B (from A-B-C)")
        show_vector_signature(vec_C, "  C")
        show_vector_signature(vec_X, "  X")
        show_vector_signature(vec_B2, "  B (from X-B-Z)")
        show_vector_signature(vec_Z, "  Z")

        # B similarity
        sim_B = compare_vectors(vec_B1, vec_B2)
        print(f"\nB similarity: {sim_B:.4f}")
        if sim_B > 0.7:
            print("  ✓ B recognized as same")
        else:
            print(f"  ⚠ B similarity low ({sim_B:.2f})")

        # Pairwise similarities
        print("\n--- All pairwise similarities ---")
        vecs = {"A": vec_A, "B": vec_B1, "C": vec_C, "X": vec_X, "Z": vec_Z}
        for name1, v1 in vecs.items():
            for name2, v2 in vecs.items():
                if name1 < name2:  # avoid duplicates
                    sim = compare_vectors(v1, v2)
                    print(f"  {name1}-{name2}: {sim:.3f}")

        # Generation WITHOUT exploration
        print("\n[5] Generation (NO EXPLORATION)...")
        print("  exploration=0.0 (deterministic)")

        generated = agent.hebbian_memory.generate(
            vec_X, n_steps=2, exploration=0.0, context_window=3  # ← NO NOISE!
        )

        print(f"\nGenerated {len(generated)} vectors:")
        for i, vec in enumerate(generated):
            show_vector_signature(vec, f"  Step {i}")

        # Evaluation
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
            print(f"  Difference: {abs(sim_2_C - sim_2_Z):.3f}")

            # STRICT criteria for compositional learning
            if sim_2_C > sim_2_Z + 0.1:  # C must WIN by at least 0.1
                print("\n✓✓ COMPOSITIONAL! (deterministic)")
            elif sim_2_Z > sim_2_C + 0.1:
                print("\n✗ Repetition (X-B-Z)")
            else:
                print("\n? Ambiguous (difference < 0.1)")
                print("  → System cannot reliably distinguish C from Z")
                print("  → Need better frequency resolution")

        # Summary
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)

        # Check if vectors are distinct
        distinct = True
        if sim_B < 0.7:
            print("✗ B not recognized consistently")
            distinct = False

        sims = [
            compare_vectors(v1, v2)
            for v1 in [vec_A, vec_C, vec_X, vec_Z]
            for v2 in [vec_A, vec_C, vec_X, vec_Z]
            if v1 is not v2
        ]
        avg_cross_sim = np.mean(sims)

        if avg_cross_sim > 0.5:
            print(f"✗ Vectors too similar (avg cross-sim: {avg_cross_sim:.3f})")
            print("  → Frequency bands not distinctive enough")
            distinct = False

        if distinct:
            print("✓ Vectors are distinct")
            if abs(sim_2_C - sim_2_Z) > 0.1:
                print("✓ System learned compositional structure!")
            else:
                print("? Learning unclear - may need more training")
        else:
            print("\nRECOMMENDATION:")
            print("  - Increase sparsity (k_sparse=4 instead of 6)")
            print("  - Or use wider frequency range (200Hz, 1000Hz, 2000Hz)")
            print("  - Or try random projection to 32D")

    else:
        print(
            f"✗ Insufficient vectors: {len(snap1) if snap1 else 0}, {len(snap2) if snap2 else 0}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
