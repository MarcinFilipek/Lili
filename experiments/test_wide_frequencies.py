"""
QUICK FIX: Use widely separated frequencies
This should give DISTINCT band signatures
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

    if np.max(activations) > 0:
        top_k = np.argpartition(activations, -k_sparse)[-k_sparse:]
        sparse = np.zeros_like(activations)
        sparse[top_k] = activations[top_k]
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


def process_signal_smart(agent, signal, hop_size):
    n_hops = len(signal) // hop_size
    last_wm_snapshot = None

    for hop_idx in range(n_hops):
        samples = signal[hop_idx * hop_size : (hop_idx + 1) * hop_size]
        prev_delta = BRAIN_STATE.get("delta", -1)
        manual_update_rhythms(HOP_DURATION_SEC)

        S_current = hear_sparse(agent, samples, k_sparse=6)
        agent.process_gamma_frequency(S_current)
        agent.process_theta_frame()

        curr_delta = BRAIN_STATE["delta"]
        if prev_delta < 0.95 and curr_delta >= 0.95:
            wm_vecs = [
                agent.working_memory[i].copy()
                for i in range(agent.working_memory_size)
                if np.any(agent.working_memory[i])
            ]
            if wm_vecs:
                last_wm_snapshot = wm_vecs

        agent.process_delta_frequency()

    return last_wm_snapshot or []


def compare_vectors(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def show_signature(vec, label=""):
    active = list(np.where(vec > 0.01)[0])
    print(f"{label}: bands {active}")


def main():
    print("=" * 70)
    print("WIDELY SEPARATED FREQUENCIES TEST")
    print("=" * 70)

    hop_size = int(FS * HOP_MS / 1000)

    # Setup
    RHYTHMS["theta"].target_freq = 4.0
    RHYTHMS["theta"].current_freq = 4.0
    RHYTHMS["theta"].phase = 0.0

    RHYTHMS["delta"].target_freq = 0.5
    RHYTHMS["delta"].current_freq = 0.5
    RHYTHMS["delta"].phase = -np.pi / 2

    agent = BioAgent5(echoic_memory_size=20, working_memory_size=7)

    # WIDELY SEPARATED FREQUENCIES
    print("\nFrequencies (widely separated):")
    print("  A:  300 Hz  (low)")
    print("  B: 1000 Hz  (mid)")
    print("  C: 2000 Hz  (high)")
    print("  X:  600 Hz  (low-mid)")
    print("  Z: 2500 Hz  (very high)")
    print("\n  Expected: DISTINCT band patterns")

    # Sequences
    seq1 = np.concatenate(
        [
            generate_tone(300, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(1000, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(2000, 200),
            np.zeros(int(FS * 1.0)),
        ]
    )

    seq2 = np.concatenate(
        [
            generate_tone(600, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(1000, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(2500, 200),
            np.zeros(int(FS * 1.0)),
        ]
    )

    # Training (3x A-B-C, 1x X-B-Z)
    print("\nTraining: 3x A-B-C, 1x X-B-Z...")
    snap1 = None
    for _ in range(3):
        snap = process_signal_smart(agent, seq1, hop_size)
        if snap:
            snap1 = snap

    snap2 = process_signal_smart(agent, seq2, hop_size)

    print(f"Captured: {len(snap1) if snap1 else 0}, {len(snap2) if snap2 else 0}")

    # Analysis
    if len(snap1) >= 3 and len(snap2) >= 3:
        vec_A, vec_B1, vec_C = snap1[0], snap1[1], snap1[2]
        vec_X, vec_B2, vec_Z = snap2[0], snap2[1], snap2[2]

        print("\n--- Vector Signatures ---")
        show_signature(vec_A, "  A (300Hz)")
        show_signature(vec_B1, "  B (1000Hz)")
        show_signature(vec_C, "  C (2000Hz)")
        show_signature(vec_X, "  X (600Hz)")
        show_signature(vec_B2, "  B (1000Hz)")
        show_signature(vec_Z, "  Z (2500Hz)")

        # Check distinctiveness
        print("\n--- Distinctiveness Check ---")
        sims = {
            "A-B": compare_vectors(vec_A, vec_B1),
            "A-C": compare_vectors(vec_A, vec_C),
            "B-C": compare_vectors(vec_B1, vec_C),
            "X-Z": compare_vectors(vec_X, vec_Z),
            "C-Z": compare_vectors(vec_C, vec_Z),
        }

        for pair, sim in sims.items():
            status = "✓ distinct" if sim < 0.5 else "✗ similar"
            print(f"  {pair}: {sim:.3f} {status}")

        avg_sim = np.mean(list(sims.values()))
        print(f"\n  Average cross-similarity: {avg_sim:.3f}")

        if avg_sim < 0.5:
            print("  ✓ Vectors are DISTINCT!")
        else:
            print("  ✗ Still too similar")

        # B similarity
        sim_B = compare_vectors(vec_B1, vec_B2)
        print(f"\n  B (seq1 vs seq2): {sim_B:.3f}")
        if sim_B > 0.7:
            print("  ✓ B recognized as same")

        # Generation (NO exploration)
        print("\n--- Generation (exploration=0.0) ---")
        generated = agent.hebbian_memory.generate(vec_X, n_steps=2, exploration=0.0)

        print("\nGenerated:")
        for i, vec in enumerate(generated):
            show_signature(vec, f"  Step {i}")

        # Evaluation
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)

        gen_1, gen_2 = generated[1], generated[2]

        sim_1_B = compare_vectors(gen_1, vec_B1)
        sim_2_C = compare_vectors(gen_2, vec_C)
        sim_2_Z = compare_vectors(gen_2, vec_Z)

        print(f"\nStep 1 vs B: {sim_1_B:.3f}", "✓" if sim_1_B > 0.5 else "✗")
        print(f"\nStep 2:")
        print(f"  vs C: {sim_2_C:.3f}")
        print(f"  vs Z: {sim_2_Z:.3f}")
        print(f"  Difference: {abs(sim_2_C - sim_2_Z):.3f}")

        if sim_2_C > sim_2_Z + 0.15:
            print("\n✓✓✓ SUCCESS! COMPOSITIONAL LEARNING!")
            print("    Generated X-B-C (novel combination)")
            print("    WITHOUT exploration (deterministic)")
        elif sim_2_Z > sim_2_C + 0.15:
            print("\n✗ Repetition (X-B-Z)")
        else:
            print("\n? Ambiguous (need difference > 0.15)")

        print("\n" + "=" * 70)

    else:
        print("✗ Insufficient vectors captured")


if __name__ == "__main__":
    main()
