"""
MULTI-PROCESS TEST dla TWOJEGO BioAgent6

Używa oryginalnego BioAgent6 z external rhythms
(nie wymaga funkcji process())
"""

import numpy as np
import multiprocessing as mp
import time
import threading


# ============================================================================
# EXTERNAL RHYTHMS (dla twojego agenta)
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


RHYTHMS = {
    "theta": BrainRhythm(0),
    "delta": BrainRhythm(0),
}

BRAIN_STATE = {"theta": 0.0, "delta": 0.0}


def update_brain_state_loop():
    """Background thread for rhythms"""
    while True:
        for name, rhythm in RHYTHMS.items():
            rhythm.update()
            BRAIN_STATE[name] = rhythm.value
        time.sleep(0.001)


def start_global_rhythms():
    """Start rhythm thread"""
    print("[RHYTHMS] Starting rhythm thread...")
    t = threading.Thread(target=update_brain_state_loop, daemon=True)
    t.start()
    time.sleep(0.1)


# ============================================================================
# AUDIO GENERATOR PROCESS
# ============================================================================


def audio_generator_process(audio_queue, sample_rate=16000, hop_size=400):
    """Generate and send audio"""
    print("[GENERATOR] Starting...")

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
            parts.append(np.zeros(int(sample_rate * 0.1)))
        parts.append(np.zeros(int(sample_rate * 1.5)))
        return np.concatenate(parts)

    sequences = [
        ("A-B-C #1", seq([300, 1000, 2000])),
        ("A-B-C #2", seq([300, 1000, 2000])),
        ("A-B-C #3", seq([300, 1000, 2000])),
        ("X-B-Z", seq([600, 1000, 2500])),
    ]

    hop_duration = hop_size / sample_rate

    for name, audio in sequences:
        print(f"[GENERATOR] Sending {name}...")

        for i in range(0, len(audio) - hop_size, hop_size):
            hop = audio[i : i + hop_size]
            audio_queue.put(hop)
            time.sleep(hop_duration)  # Real-time

        print(f"[GENERATOR] ✓ {name}")

    audio_queue.put(None)
    print("[GENERATOR] Done.")


# ============================================================================
# AGENT PROCESS (twój BioAgent6)
# ============================================================================


def agent_process(audio_queue):
    """BioAgent process - używa TWOJEGO agenta"""
    print("[AGENT] Starting...")

    # Import twojego agenta
    import sys

    sys.path.insert(0, "/home/claude")

    # === FIX: Start rhythms WEWNĄTRZ procesu agenta ===
    # Rhythm thread musi być w tym samym process space co agent!

    # Stwórz moduł brain_rythm dla importu
    import types

    brain_rythm_module = types.ModuleType("brain_rythm")
    brain_rythm_module.RHYTHMS = RHYTHMS
    brain_rythm_module.BRAIN_STATE = BRAIN_STATE
    brain_rythm_module.BrainRhythm = BrainRhythm
    sys.modules["brain_rythm"] = brain_rythm_module

    # Start rhythms IN THIS PROCESS
    start_global_rhythms()

    # Start theta & delta
    RHYTHMS["theta"].start(4.0)
    RHYTHMS["delta"].start(1.0)
    print("[AGENT] Rhythms started: theta=4Hz, delta=1Hz")

    # Wait a bit for rhythms to stabilize
    time.sleep(0.5)

    # Import agenta (musi być po stworzeniu brain_rythm)
    from bio_agent_6_rythms import BioAgent6

    agent = BioAgent6(
        echoic_memory_size=20, working_memory_size=7, hop_size=400, rhythms=RHYTHMS
    )

    print("[AGENT] Ready. Waiting for audio...\n")

    hop_count = 0
    theta_count = 0
    delta_count = 0

    while True:
        hop = audio_queue.get()

        if hop is None:
            break

        # Debug: Check rhythm values periodically
        if hop_count % 100 == 0:
            print(
                f"[AGENT] Hop {hop_count}: theta={RHYTHMS['theta'].value:.2f}, delta={RHYTHMS['delta'].value:.2f}"
            )

        # === TWOJA OBECNA LOGIKA ===
        S_current = agent.hear(hop)
        agent.process_gamma_frequency(hop)

        # Check if theta triggered
        prev_theta = agent._theta_processed
        agent.process_theta_frame()
        if not prev_theta and agent._theta_processed:
            theta_count += 1

        # Check if delta triggered
        prev_delta = agent._delta_processed
        agent.process_delta_frequency()
        if not prev_delta and agent._delta_processed:
            delta_count += 1

        hop_count += 1

    print(f"\n[AGENT] Processed {hop_count} hops")
    print(f"[AGENT] Theta triggered: {theta_count} times")
    print(f"[AGENT] Delta triggered: {delta_count} times")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\nLTM weight norm: {W_norm:.3f}")

    if W_norm > 0.3:
        print("✓ System learned!")
    else:
        print("? Learning unclear")

    # Working memory
    wm_snap = []
    for i in range(agent.working_memory_size):
        vec = agent.working_memory[i]
        if np.any(vec):
            wm_snap.append(vec.copy())

    print(f"Working memory: {len(wm_snap)} vectors")

    if len(wm_snap) > 0:
        print("\nVector signatures:")
        for i, vec in enumerate(wm_snap):
            active = list(np.where(vec > 0.1)[0])
            if active:
                dom = max(active, key=lambda j: vec[j])
                print(f"  [{i}] bands {active[:5]}..., DOM={dom}")

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("MULTI-PROCESS TEST - Original BioAgent6")
    print("=" * 70)
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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
