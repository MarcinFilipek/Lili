import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import csv
import argparse
import sys
import os
import time

# Dodaj ścieżkę do BioAgent4
sys.path.insert(0, os.path.dirname(__file__))
from BioAgent4 import BioAgent4

FS = 16000
HOP_MS = 80


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
    """
    freqs = {"A": 440, "B": 880, "C": 1320, "D": 1760, "_": 0}  # Pauza

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
        beat = tones[note]

        end_pos = min(position + len(beat), total_samples)
        sequence[position:end_pos] = beat[: end_pos - position]

        position += beat_samples
        idx += 1

    return sequence


def save_results_to_csv(filename, history):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Time_Sec", "Prediction_Error"])
        for i, pe in enumerate(history):
            time_sec = i * HOP_MS / 1000
            writer.writerow([i, f"{time_sec:.3f}", f"{pe:.6f}"])
    print(f"Wyniki zapisano do: {filename}")


def run_offline(agent, sequence, hop_size, save_csv=False):
    print("Uruchamianie symulacji offline...")
    pe_history = []

    start_time = time.time()

    # Przetwarzaj ramka po ramce
    for i in range(0, len(sequence), hop_size):
        frame = sequence[i : i + hop_size]
        if len(frame) < hop_size:
            frame = np.pad(frame, (0, hop_size - len(frame)))

        _, pe = agent.process(frame)
        pe_history.append(pe)

        # Pasek postępu
        if i % (hop_size * 10) == 0:
            progress = i / len(sequence) * 100
            sys.stdout.write(f"\\rPostęp: {progress:.1f}%")
            sys.stdout.flush()

    sys.stdout.write("\\rPostęp: 100.0%\\n")
    duration = time.time() - start_time
    print(
        f"Przetworzono w {duration:.2f}s (x{len(sequence)/FS/duration:.1f} szybciej niż realtime)"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(pe_history)
    plt.title("Błąd Predykcji (Tryb Offline)")
    plt.xlabel("Ramki")
    plt.ylabel("Prediction Error")
    plt.grid(True, alpha=0.3)

    if save_csv:
        save_results_to_csv("results_sequence_offline.csv", pe_history)
        plt.savefig("results_sequence_offline.png")
        print("Wykres zapisano do: results_sequence_offline.png")

    plt.show()


def run_realtime(agent, sequence, hop_size, save_csv=False):
    print("Uruchamianie eksperymentu realtime...")

    stream = sd.OutputStream(
        samplerate=FS, channels=1, blocksize=hop_size, dtype="float32"
    )
    stream.start()

    pe_history = []
    max_display = 200
    display_data = deque(maxlen=max_display)

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Eksperyment Sekwencje (Realtime)", fontsize=16)

    (line_audio,) = ax1.plot(np.zeros(hop_size), color="cyan", label="Audio")
    ax1.set_ylim(-1, 1)
    ax1.set_title("Sygnał Audio")
    ax1.grid(True, alpha=0.3)

    (line_pe,) = ax2.plot(np.zeros(max_display), color="orange", linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max_display)
    ax2.set_title("Błąd Predykcji (Prediction Error)")
    ax2.set_ylabel("Error")
    ax2.grid(True, alpha=0.3)

    def frame_generator():
        for i in range(0, len(sequence), hop_size):
            frame = sequence[i : i + hop_size]
            if len(frame) < hop_size:
                frame = np.pad(frame, (0, hop_size - len(frame)))
            yield frame

    frame_gen = frame_generator()

    def update(frame_num):
        try:
            frame = next(frame_gen)
            stream.write(frame)

            _, pe = agent.process(frame)

            pe_history.append(pe)
            display_data.append(pe)

            line_audio.set_ydata(frame)
            if len(display_data) > 0:
                line_pe.set_data(range(len(display_data)), list(display_data))
                if max(display_data) > 0.1:
                    ax2.set_ylim(0, max(display_data) * 1.2)

            return line_audio, line_pe
        except StopIteration:
            if save_csv:
                save_results_to_csv("results_sequence_realtime.csv", pe_history)
            plt.close(fig)
            return line_audio, line_pe

    ani = animation.FuncAnimation(
        fig, update, interval=1, blit=True, cache_frame_data=False
    )
    plt.show()

    stream.stop()
    stream.close()


def main():
    parser = argparse.ArgumentParser(description="Eksperyment Sekwencje - BioAgent4")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo")
    parser.add_argument("--duration", type=int, default=30, help="Czas trwania")
    parser.add_argument(
        "--pattern",
        type=str,
        default="A,B,A,C",
        help="Wzorzec nut oddzielony przecinkami",
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "realtime"],
        default="realtime",
        help="Tryb eksperymentu",
    )
    parser.add_argument("--save-csv", action="store_true", help="Zapisz wyniki")

    args = parser.parse_args()
    pattern = args.pattern.split(",")

    print("Inicjalizacja BioAgent4...")
    agent = BioAgent4()
    hop_size = int(FS * HOP_MS / 1000)

    print(f"=== Eksperyment Sekwencje ===")
    print(f"BPM: {args.bpm}")
    print(f"Wzorzec: {pattern}")
    print(f"Generowanie sygnału...")

    sequence = create_sequence(
        pattern=pattern, bpm=args.bpm, duration_sec=args.duration
    )

    if args.mode == "offline":
        run_offline(agent, sequence, hop_size, args.save_csv)
    else:
        run_realtime(agent, sequence, hop_size, args.save_csv)


if __name__ == "__main__":
    main()
