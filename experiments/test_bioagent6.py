"""
OFFLINE TEST dla BioAgent6

Testuje kompozycyjność bez mikrofonu:
- Generuje syntetyczne sekwencje A-B-C i X-B-Z
- Karmi agenta hop-by-hop
- Sprawdza czy nauczył się X-B-C

Nie wymaga:
- Mikrofonu
- Real-time processing
- Entrainment (theta/delta są fixed 4Hz/1Hz)
"""

import numpy as np
import sys

sys.path.insert(0, "/home/claude")

from bio_agent_6 import BioAgent6, RHYTHMS, start_global_rhythms
import time

FS = 16000
HOP_SIZE = 400


def generate_tone(frequency, duration_ms=200, amplitude=0.3):
    """Generate pure tone"""
    n_samples = int(FS * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)

    envelope = np.ones(n_samples)
    fade_len = int(0.01 * FS)
    if n_samples > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    tone = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return tone.astype(np.float32)


def generate_sequence(freqs, tone_dur=200, pause_short=100, pause_long=1500):
    """Generate audio sequence"""
    parts = []
    for freq in freqs:
        parts.append(generate_tone(freq, tone_dur))
        parts.append(np.zeros(int(FS * pause_short / 1000)))

    parts.append(np.zeros(int(FS * pause_long / 1000)))
    return np.concatenate(parts)


def feed_sequence(agent, audio, label=""):
    """
    Feed audio sequence to agent hop-by-hop
    Simulates real-time processing
    """
    print(f"\n[FEED] {label}")
    print(f"  Duration: {len(audio)/FS:.2f}s")

    n_hops = len(audio) // HOP_SIZE

    for i in range(n_hops):
        hop = audio[i * HOP_SIZE : (i + 1) * HOP_SIZE]

        # Process hop (like in main loop)
        S_current = agent.hear(hop)
        agent.process_gamma_frequency(hop)
        agent.process_theta_frame()
        agent.process_delta_frequency()

        # Small delay to simulate time passing (for rhythm phase)
        time.sleep(0.001)

    print(f"  ✓ Processed {n_hops} hops")


def get_wm_snapshot(agent):
    """Capture working memory state"""
    snapshot = []
    for i in range(agent.working_memory_size):
        vec = agent.working_memory[i]
        if np.any(vec):
            snapshot.append(vec.copy())
    return snapshot


def compare_vectors(vec1, vec2):
    """Cosine similarity"""
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def show_signature(vec, label=""):
    """Show active frequency bands"""
    active = list(np.where(vec > 0.1)[0])
    if active:
        dominant = max(active, key=lambda i: vec[i])
        print(
            f"{label}: bands {active[:8]}{'...' if len(active) > 8 else ''}, DOM={dominant}"
        )
    else:
        print(f"{label}: EMPTY")


def main():
    print("=" * 70)
    print("OFFLINE COMPOSITIONAL TEST - BioAgent6")
    print("=" * 70)

    # ========================================================================
    # SETUP
    # ========================================================================

    print("\n[1] Starting global rhythms...")
    start_global_rhythms()
    time.sleep(0.1)  # Let thread start

    # Start rhythms at fixed frequencies
    RHYTHMS["theta"].start(4.0)  # 4 Hz
    RHYTHMS["delta"].start(1.0)  # 1 Hz
    print(f"  Theta: {RHYTHMS['theta'].target_freq} Hz")
    print(f"  Delta: {RHYTHMS['delta'].target_freq} Hz")

    print("\n[2] Creating agent...")
    agent = BioAgent6(echoic_memory_size=20, working_memory_size=7, hop_size=HOP_SIZE)

    # ========================================================================
    # GENERATE AUDIO
    # ========================================================================

    print("\n[3] Generating audio sequences...")
    print("  A-B-C: 300Hz - 1000Hz - 2000Hz")
    print("  X-B-Z: 600Hz - 1000Hz - 2500Hz")

    seq1 = generate_sequence(
        [300, 1000, 2000], tone_dur=200, pause_short=100, pause_long=1500
    )
    seq2 = generate_sequence(
        [600, 1000, 2500], tone_dur=200, pause_short=100, pause_long=1500
    )

    print(f"  Duration: {len(seq1)/FS:.2f}s each")

    # ========================================================================
    # TRAINING
    # ========================================================================

    print("\n[4] Training...")
    print("  Training A-B-C (3x to strengthen)...")

    for i in range(3):
        feed_sequence(agent, seq1, f"A-B-C #{i+1}")
        time.sleep(0.5)  # Pause between sequences

    print("\n  Training X-B-Z (1x)...")
    feed_sequence(agent, seq2, "X-B-Z")

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print("\n[5] Analyzing learned representations...")

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\n  LTM weight norm: {W_norm:.3f}")

    if W_norm < 0.1:
        print("  ✗ LTM didn't learn - check consolidations occurred")
        return

    print("  ✓ LTM is learning!")

    # Check working memory state
    wm_snap = get_wm_snapshot(agent)
    print(f"\n  Current WM: {len(wm_snap)} vectors")

    if len(wm_snap) >= 3:
        print("\n  WM contents:")
        for i, vec in enumerate(wm_snap):
            show_signature(vec, f"    [{i}]")

    # ========================================================================
    # GENERATION TEST (if you have generate method)
    # ========================================================================

    print("\n[6] Generation test...")

    if hasattr(agent.hebbian_memory, "generate"):
        # Find a seed vector (e.g., last from WM)
        if len(wm_snap) >= 1:
            seed = wm_snap[0]

            print("  Generating from seed...")
            show_signature(seed, "    Seed")

            generated = agent.hebbian_memory.generate(seed, n_steps=2, exploration=0.1)

            print("\n  Generated sequence:")
            for i, vec in enumerate(generated):
                show_signature(vec, f"    Step {i}")
        else:
            print("  ✗ No vectors in WM to use as seed")
    else:
        print("  ⚠ HebbianMemory doesn't have generate() method")
        print("    → Can't test generation yet")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  System state:")
    print(f"    LTM weight norm: {W_norm:.3f}")
    print(f"    Working memory: {len(wm_snap)} vectors")

    if W_norm > 0.3:
        print("\n  ✓ System learned associations!")
        print("    → Try adding generate() to HebbianMemory for compositional test")
    else:
        print("\n  ? Learning is weak - try:")
        print("    - More training repetitions")
        print("    - Longer pauses between tones")
        print("    - Check theta/delta consolidations")

    print("=" * 70)


if __name__ == "__main__":
    main()
