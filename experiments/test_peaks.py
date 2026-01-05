"""
SIMPLIFIED: Use PAUSE detection for consolidation
(not delta rhythm, which has timing issues)
"""

import numpy as np
from bio_agent_5 import BioAgent5, RHYTHMS, BRAIN_STATE
from hebbian_memory_wta import HebbianMemoryWTA

FS = 16000
HOP_MS = 25
HOP_DURATION_SEC = HOP_MS / 1000.0


def generate_tone(frequency, duration_ms=150, amplitude=0.3):
    n_samples = int(FS * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    envelope = np.ones(n_samples)
    fade_len = int(0.01 * FS)
    if n_samples > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    tone = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return tone.astype(np.float32)


def hear_peaks(agent, frame, min_prominence=0.3, max_peaks=6):
    """Peak detection sparse coding"""
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

    if np.max(activations) == 0:
        return activations

    # Peak detection
    peaks = []
    for i in range(1, len(activations) - 1):
        if activations[i] > activations[i - 1] and activations[i] > activations[i + 1]:
            prominence = activations[i] - max(activations[i - 1], activations[i + 1])
            if prominence > min_prominence * np.max(activations):
                peaks.append((i, activations[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    peaks = peaks[:max_peaks]

    sparse = np.zeros_like(activations)
    for peak_idx, peak_val in peaks:
        sparse[peak_idx] = peak_val
        if peak_idx > 0:
            sparse[peak_idx - 1] = peak_val * 0.3
        if peak_idx < len(sparse) - 1:
            sparse[peak_idx + 1] = peak_val * 0.3

    if np.max(sparse) > 0:
        sparse /= np.max(sparse) + 1e-9

    return sparse


def manual_update_rhythms(dt):
    for rhythm in RHYTHMS.values():
        if rhythm.target_freq > 0:
            rhythm.phase += 2 * np.pi * rhythm.target_freq * dt
            rhythm.phase = rhythm.phase % (2 * np.pi)
            rhythm.value = np.sin(rhythm.phase)
    BRAIN_STATE["theta"] = RHYTHMS["theta"].value


def process_signal_with_pauses(agent, signal, hop_size):
    """
    Process signal with PAUSE-based consolidation
    (not delta rhythm)
    """
    n_hops = len(signal) // hop_size

    silence_threshold = 0.01  # Energy threshold for pause
    min_pause_frames = 20  # 500ms of silence
    silence_frames = 0

    wm_snapshots = []

    for hop_idx in range(n_hops):
        samples = signal[hop_idx * hop_size : (hop_idx + 1) * hop_size]

        manual_update_rhythms(HOP_DURATION_SEC)

        S_current = hear_peaks(agent, samples, min_prominence=0.3, max_peaks=6)
        agent.process_gamma_frequency(S_current)
        agent.process_theta_frame()

        # Pause detection (energy-based)
        energy = np.sum(S_current**2)

        if energy < silence_threshold:
            silence_frames += 1
        else:
            silence_frames = 0

        # If pause detected, consolidate
        if silence_frames >= min_pause_frames:
            # Capture WM before consolidation
            wm_vecs = [
                agent.working_memory[i].copy()
                for i in range(agent.working_memory_size)
                if np.any(agent.working_memory[i])
            ]

            if len(wm_vecs) > 1:
                print(
                    f"  [{hop_idx * HOP_DURATION_SEC:.2f}s] PAUSE → consolidate({len(wm_vecs)} vecs)"
                )

                # Consolidate
                agent.hebbian_memory.learn_sequence(wm_vecs)

                # Save snapshot
                wm_snapshots.append(wm_vecs)

                # Clear WM
                agent.working_memory.fill(0)

            # Reset silence counter
            silence_frames = 0

    return wm_snapshots


def compare_vectors(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def show_signature(vec, label=""):
    active = list(np.where(vec > 0.1)[0])
    if active:
        dominant = max(active, key=lambda i: vec[i])
        print(f"{label}: bands {active}, DOM={dominant}")
    else:
        print(f"{label}: EMPTY")


def main():
    print("=" * 70)
    print("PAUSE-BASED CONSOLIDATION TEST")
    print("=" * 70)

    hop_size = int(FS * HOP_MS / 1000)

    # Setup (only theta, no delta!)
    RHYTHMS["theta"].target_freq = 4.0
    RHYTHMS["theta"].current_freq = 4.0
    RHYTHMS["theta"].phase = 0.0

    agent = BioAgent5(echoic_memory_size=20, working_memory_size=7)
    agent.hebbian_memory = HebbianMemoryWTA(agent.n_sensors)

    print("\nFrequencies (widely separated):")
    print("  A:  300 Hz")
    print("  B: 1000 Hz")
    print("  C: 2000 Hz")
    print("  X:  600 Hz")
    print("  Z: 2500 Hz")

    # Sequences (LONGER tones so theta catches all)
    # Theta @ 4Hz = sample every 250ms
    # Need tone + pause > 250ms to guarantee sampling
    seq1 = np.concatenate(
        [
            generate_tone(300, 200),  # 200ms tone
            np.zeros(int(FS * 0.1)),  # 100ms pause = 300ms total
            generate_tone(1000, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(2000, 200),
            np.zeros(int(FS * 0.8)),  # LONG PAUSE (800ms)
        ]
    )

    seq2 = np.concatenate(
        [
            generate_tone(600, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(1000, 200),
            np.zeros(int(FS * 0.1)),
            generate_tone(2500, 200),
            np.zeros(int(FS * 0.8)),
        ]
    )

    # Training
    print("\nTraining: 3x A-B-C, 1x X-B-Z...")

    all_snaps = []
    for i in range(3):
        snaps = process_signal_with_pauses(agent, seq1, hop_size)
        all_snaps.extend(snaps)
        print(f"  Seq1 #{i+1}: captured {len(snaps)} consolidations")

    snaps2 = process_signal_with_pauses(agent, seq2, hop_size)
    all_snaps.extend(snaps2)
    print(f"  Seq2: captured {len(snaps2)} consolidations")

    print(f"\nTotal consolidations: {len(all_snaps)}")

    # Get last two snapshots (from last A-B-C and X-B-Z)
    if len(all_snaps) >= 2:
        snap_abc = all_snaps[-2]  # Last A-B-C
        snap_xbz = all_snaps[-1]  # X-B-Z

        print(f"\nLast A-B-C: {len(snap_abc)} vectors")
        print(f"Last X-B-Z: {len(snap_xbz)} vectors")

        if len(snap_abc) >= 3 and len(snap_xbz) >= 3:
            vec_A, vec_B1, vec_C = snap_abc[0], snap_abc[1], snap_abc[2]
            vec_X, vec_B2, vec_Z = snap_xbz[0], snap_xbz[1], snap_xbz[2]

            print("\n--- Signatures ---")
            show_signature(vec_A, "A")
            show_signature(vec_B1, "B")
            show_signature(vec_C, "C")
            show_signature(vec_X, "X")
            show_signature(vec_Z, "Z")

            sim_B = compare_vectors(vec_B1, vec_B2)
            print(f"\nB similarity: {sim_B:.3f}")

            # Generation
            print("\n--- Generation ---")
            generated = agent.hebbian_memory.generate(vec_X, n_steps=2, exploration=0.0)

            for i, vec in enumerate(generated):
                show_signature(vec, f"Step {i}")

            # Eval
            print("\n" + "=" * 70)
            gen_1, gen_2 = generated[1], generated[2]

            sim_1_B = compare_vectors(gen_1, vec_B1)
            sim_2_C = compare_vectors(gen_2, vec_C)
            sim_2_Z = compare_vectors(gen_2, vec_Z)

            print(f"Step 1 vs B: {sim_1_B:.3f}")
            print(f"Step 2 vs C: {sim_2_C:.3f}")
            print(f"Step 2 vs Z: {sim_2_Z:.3f}")
            print(f"Difference:  {abs(sim_2_C - sim_2_Z):.3f}")

            if sim_2_C > sim_2_Z + 0.15:
                print("\n✓✓✓ COMPOSITIONAL!")
            elif sim_2_Z > sim_2_C + 0.15:
                print("\n✗ Repetition")
            else:
                print("\n? Ambiguous")

            print("=" * 70)
        else:
            print("✗ Insufficient vectors in snapshots")
    else:
        print("✗ Not enough consolidations captured")


if __name__ == "__main__":
    main()
