"""
ADVANCED MULTI-PROCESS SYSTEM

Flexible audio sources:
- Synthetic tones (for testing)
- Audio file (for pre-recorded data)
- Microphone (for real-time)

All use the same agent process!
"""

import numpy as np
import multiprocessing as mp
import time
from enum import Enum


class AudioSource(Enum):
    """Audio source types"""

    SYNTHETIC = "synthetic"
    FILE = "file"
    MICROPHONE = "microphone"


# ============================================================================
# AUDIO SOURCES
# ============================================================================


def synthetic_audio_source(
    audio_queue, sample_rate=16000, hop_size=400, real_time=True
):
    """
    Generate synthetic test sequences

    Args:
        audio_queue: Queue to send audio
        sample_rate: Sample rate (Hz)
        hop_size: Hop size (samples)
        real_time: If True, simulate real-time by sleeping
    """
    print("[SOURCE:SYNTH] Starting synthetic audio generator...")

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

    # Test sequences
    sequences = [
        ("A-B-C #1", seq([300, 1000, 2000])),
        ("A-B-C #2", seq([300, 1000, 2000])),
        ("A-B-C #3", seq([300, 1000, 2000])),
        ("X-B-Z", seq([600, 1000, 2500])),
    ]

    hop_duration = hop_size / sample_rate

    for name, audio in sequences:
        print(f"[SOURCE:SYNTH] Sending {name}...")

        for i in range(0, len(audio) - hop_size, hop_size):
            hop = audio[i : i + hop_size]
            audio_queue.put(hop)

            if real_time:
                time.sleep(hop_duration)

        print(f"[SOURCE:SYNTH] ✓ {name}")

    audio_queue.put(None)  # End signal
    print("[SOURCE:SYNTH] Done.")


def file_audio_source(audio_queue, filepath, hop_size=400, real_time=False):
    """
    Stream audio from file

    Args:
        audio_queue: Queue to send audio
        filepath: Path to audio file
        hop_size: Hop size (samples)
        real_time: If True, simulate real-time playback
    """
    print(f"[SOURCE:FILE] Loading {filepath}...")

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(filepath)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Mono

        print(f"[SOURCE:FILE] Duration: {len(audio)/sample_rate:.2f}s")

        hop_duration = hop_size / sample_rate

        for i in range(0, len(audio) - hop_size, hop_size):
            hop = audio[i : i + hop_size].astype(np.float32)
            audio_queue.put(hop)

            if real_time:
                time.sleep(hop_duration)

        audio_queue.put(None)
        print("[SOURCE:FILE] Done.")

    except Exception as e:
        print(f"[SOURCE:FILE] Error: {e}")
        audio_queue.put(None)


def microphone_audio_source(
    audio_queue, hop_size=400, sample_rate=16000, duration_sec=30
):
    """
    Stream audio from microphone

    Args:
        audio_queue: Queue to send audio
        hop_size: Hop size (samples)
        sample_rate: Sample rate (Hz)
        duration_sec: How long to record
    """
    print(f"[SOURCE:MIC] Starting microphone (recording {duration_sec}s)...")

    try:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[SOURCE:MIC] Status: {status}")
            audio_queue.put(indata.flatten().copy())

        with sd.InputStream(
            samplerate=sample_rate, channels=1, callback=callback, blocksize=hop_size
        ):
            time.sleep(duration_sec)

        audio_queue.put(None)
        print("[SOURCE:MIC] Done.")

    except Exception as e:
        print(f"[SOURCE:MIC] Error: {e}")
        audio_queue.put(None)


# ============================================================================
# AGENT PROCESS (unified - works with any source)
# ============================================================================


def agent_process(audio_queue, verbose=True):
    """
    BioAgent process - receives and processes audio

    Works with ANY audio source!

    Args:
        audio_queue: Queue to receive audio
        verbose: Print detailed logs
    """
    print("[AGENT] Starting...")

    import sys

    sys.path.insert(0, "/home/claude")
    from bio_agent_6 import BioAgent6

    agent = BioAgent6()

    hop_count = 0
    start_time = time.time()

    print("[AGENT] Ready. Waiting for audio...\n")

    while True:
        hop = audio_queue.get()

        if hop is None:
            break

        status = agent.process(hop)
        hop_count += 1

        if verbose:
            if status.get("theta_triggered"):
                print(f"[AGENT] Theta @ {status['time']:.2f}s")
            if status.get("delta_triggered"):
                print(f"[AGENT] Delta @ {status['time']:.2f}s")

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nProcessed {hop_count} hops in {elapsed:.2f}s")
    print(f"Average: {hop_count/elapsed:.1f} hops/sec")

    W_norm = np.linalg.norm(agent.hebbian_memory.W)
    print(f"\nLTM weight norm: {W_norm:.3f}")

    wm = agent.get_working_memory_snapshot()
    print(f"Working memory: {len(wm)} vectors")

    if W_norm > 0.3:
        print("\n✓ System learned successfully!")
    else:
        print("\n? Learning unclear (weak W)")

    print("=" * 70)


# ============================================================================
# LAUNCHER
# ============================================================================


def launch_test(source_type=AudioSource.SYNTHETIC, source_config=None, verbose=True):
    """
    Launch multi-process test

    Args:
        source_type: AudioSource enum value
        source_config: Dict with source-specific config
        verbose: Verbose logging
    """
    print("=" * 70)
    print("MULTI-PROCESS AUDIO PROCESSING")
    print("=" * 70)
    print(f"\nAudio source: {source_type.value}")
    print()

    audio_queue = mp.Queue(maxsize=10)

    # Select source process
    if source_type == AudioSource.SYNTHETIC:
        config = source_config or {}
        source_proc = mp.Process(
            target=synthetic_audio_source, args=(audio_queue,), kwargs=config
        )

    elif source_type == AudioSource.FILE:
        if not source_config or "filepath" not in source_config:
            print("Error: FILE source requires 'filepath' in config")
            return
        source_proc = mp.Process(
            target=file_audio_source,
            args=(audio_queue, source_config["filepath"]),
            kwargs={k: v for k, v in source_config.items() if k != "filepath"},
        )

    elif source_type == AudioSource.MICROPHONE:
        config = source_config or {}
        source_proc = mp.Process(
            target=microphone_audio_source, args=(audio_queue,), kwargs=config
        )

    # Agent process
    agent_proc = mp.Process(target=agent_process, args=(audio_queue, verbose))

    # Start
    print("[MAIN] Starting processes...\n")
    source_proc.start()
    agent_proc.start()

    # Wait
    source_proc.join()
    agent_proc.join()

    print("\n[MAIN] Test complete.")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Example 1: Synthetic audio (fast, no real-time)
    print("\n### TEST 1: Synthetic audio (fast) ###\n")
    launch_test(
        source_type=AudioSource.SYNTHETIC,
        source_config={"real_time": False},  # Fast mode
    )

    # Example 2: Synthetic audio (real-time simulation)
    # launch_test(
    #     source_type=AudioSource.SYNTHETIC,
    #     source_config={'real_time': True}
    # )

    # Example 3: Audio file
    # launch_test(
    #     source_type=AudioSource.FILE,
    #     source_config={'filepath': 'test.wav', 'real_time': False}
    # )

    # Example 4: Microphone (30 seconds)
    # launch_test(
    #     source_type=AudioSource.MICROPHONE,
    #     source_config={'duration_sec': 30}
    # )
