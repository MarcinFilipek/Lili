import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import sys
import time
from collections import deque

FS = 16000
HOP_MS = 25


def generate_tone(duration_ms=200, frequency=440, amplitude=0.5):
    """Generuje pojedynczy ton"""
    n_samples = int(FS * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    envelope = np.ones(n_samples)
    # Fade in/out to avoid clicks
    fade_len = int(0.01 * FS)  # 10ms fade
    if n_samples > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    tone = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return tone.astype(np.float32)


def create_sequence(pattern=["A", "B", "A", "C"], bpm=60, duration_sec=30):
    """
    Tworzy sekwencję tonów wg wzorca

    Notes:
    A = 440 Hz
    B = 880 Hz
    C = 1320 Hz
    D = 1760 Hz
    X = 2200 Hz
    Z = 2640 Hz
    _ = 0 Hz (Pauza)
    """
    freqs = {"A": 440, "B": 880, "C": 1320, "D": 1760, "X": 2200, "Z": 2640, "_": 0}

    total_samples = int(FS * duration_sec)
    sequence = np.zeros(total_samples, dtype=np.float32)

    # Czas jednego uderzenia w próbkach
    beat_samples = int(FS * 60 / bpm)

    # Generuj tony
    tones = {}
    for note, freq in freqs.items():
        if note == "_":
            tones[note] = np.zeros(beat_samples, dtype=np.float32)
        else:
            # Ton trwa 80% czasu uderzenia, 20% przerwy
            tone_len = int(beat_samples * 0.8)
            tone = generate_tone(duration_ms=tone_len * 1000 / FS, frequency=freq)
            full_beat = np.zeros(beat_samples, dtype=np.float32)
            full_beat[: len(tone)] = tone
            tones[note] = full_beat

    # Składaj sekwencję
    position = 0
    idx = 0
    while position < total_samples:
        note = pattern[idx % len(pattern)]
        # Ignoruj nieznane znaki
        if note not in tones:
            print(f"Ostrzeżenie: Nieznany znak w sekwencji: '{note}', pomijam.")
            idx += 1
            continue

        beat = tones[note]

        end_pos = min(position + len(beat), total_samples)
        sequence[position:end_pos] = beat[: end_pos - position]

        position += beat_samples
        idx += 1

    return sequence


def run_realtime(sequence, hop_size):
    print("Uruchamianie eksperymentu realtime...")

    stream = sd.OutputStream(
        samplerate=FS, channels=1, blocksize=hop_size, dtype="float32"
    )
    stream.start()

    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle("Podgląd Sygnału", fontsize=16)

    (line_audio,) = ax1.plot(np.zeros(hop_size), color="cyan", label="Audio")
    ax1.set_ylim(-1, 1)
    ax1.set_title("Sygnał Audio")
    ax1.grid(True, alpha=0.3)

    def data_gen():
        for i in range(0, len(sequence), hop_size):
            frame = sequence[i : i + hop_size]
            if len(frame) < hop_size:
                frame = np.pad(frame, (0, hop_size - len(frame)))
            yield frame

    def update(frame):
        stream.write(frame)
        line_audio.set_ydata(frame)
        return (line_audio,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=data_gen,
        interval=1,
        blit=True,
        cache_frame_data=False,
        repeat=False,
    )
    plt.show()

    stream.stop()
    stream.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generator Sygnału - Sekwencje (Bez Agenta)"
    )
    parser.add_argument("--bpm", type=int, default=120, help="Tempo")
    parser.add_argument("--duration", type=int, default=30, help="Czas trwania")
    parser.add_argument(
        "--pattern",
        type=str,
        default="A,B,A,C,X,Z,_",
        help="Wzorzec nut oddzielony przecinkami (Dostępne: A, B, C, D, X, Z, _)",
    )
    args = parser.parse_args()
    pattern = args.pattern.split(",")

    print(f"=== Audio Signal Generator ===")
    print(f"BPM: {args.bpm}")
    print(f"Wzorzec: {pattern}")
    print(f"Generowanie sygnału...")

    sequence = create_sequence(
        pattern=pattern, bpm=args.bpm, duration_sec=args.duration
    )

    hop_size = int(FS * HOP_MS / 1000)

    print(f"Generacja zakończona. Długość sekwencji: {len(sequence)} próbek")

    run_realtime(sequence, hop_size)


if __name__ == "__main__":
    main()
