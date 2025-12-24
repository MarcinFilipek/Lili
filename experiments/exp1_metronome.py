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


def generate_click(duration_ms=50, frequency=1000, amplitude=0.5):
    """Generuje pojedyncze kliknięcie metronomu"""
    n_samples = int(FS * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples)
    envelope = np.hanning(n_samples)
    click = amplitude * np.sin(2 * np.pi * frequency * t) * envelope
    return click.astype(np.float32)


def create_metronome_sequence(bpm=60, duration_sec=60, click_duration_ms=50):
    """Tworzy sekwencję metronomu"""
    total_samples = int(FS * duration_sec)
    sequence = np.zeros(total_samples, dtype=np.float32)

    interval_samples = int(FS * 60 / bpm)
    click = generate_click(duration_ms=click_duration_ms)

    position = 0
    while position < total_samples:
        end_pos = min(position + len(click), total_samples)
        sequence[position:end_pos] = click[: end_pos - position]
        position += interval_samples

    return sequence


def save_results_to_csv(filename, history):
    """Zapisuje wyniki do pliku CSV"""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Time_Sec", "Prediction_Error"])
        for i, pe in enumerate(history):
            time_sec = i * HOP_MS / 1000
            writer.writerow([i, f"{time_sec:.3f}", f"{pe:.6f}"])
    print(f"Wyniki zapisano do: {filename}")


def run_offline(agent, sequence, hop_size, save_csv=False):
    """Tryb offline - szybka symulacja"""
    print("Uruchamianie symulacji offline...")
    pe_history = []

    start_time = time.time()

    # Przetwarzaj ramka po ramce
    for i in range(0, len(sequence), hop_size):
        frame = sequence[i : i + hop_size]
        # Dopełnij zerami jeśli ostatnia ramka jest za krótka
        if len(frame) < hop_size:
            frame = np.pad(frame, (0, hop_size - len(frame)))

        _, pe = agent.process(frame)
        pe_history.append(pe)

        # Pasek postępu
        if i % (hop_size * 10) == 0:
            progress = i / len(sequence) * 100
            sys.stdout.write(f"\rPostęp: {progress:.1f}%")
            sys.stdout.flush()

    sys.stdout.write("\rPostęp: 100.0%\n")
    duration = time.time() - start_time
    print(
        f"Przetworzono w {duration:.2f}s (x{len(sequence)/FS/duration:.1f} szybciej niż realtime)"
    )

    # Wyświetl wykres końcowy
    plt.figure(figsize=(10, 6))
    plt.plot(pe_history)
    plt.title("Błąd Predykcji (Tryb Offline)")
    plt.xlabel("Ramki")
    plt.ylabel("Prediction Error")
    plt.grid(True, alpha=0.3)

    if save_csv:
        save_results_to_csv("results_metronome_offline.csv", pe_history)
        plt.savefig("results_metronome_offline.png")
        print("Wykres zapisano do: results_metronome_offline.png")

    plt.show()


def run_realtime(agent, sequence, hop_size, save_csv=False):
    """Tryb realtime - odtwarzanie i wizualizacja na żywo"""
    print("Uruchamianie symulacji realtime...")

    # Przygotuj strumień audio
    stream = sd.OutputStream(
        samplerate=FS, channels=1, blocksize=hop_size, dtype="float32"
    )
    stream.start()

    # Przygotuj dane
    pe_history = []
    max_display = 100
    display_data = deque(maxlen=max_display)

    # Konfiguracja wykresu
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Eksperyment Metronomu (Realtime)", fontsize=16)

    # Wykres 1: Sygnał Audio i Predykcja
    (line_audio,) = ax1.plot(np.zeros(hop_size), color="cyan", label="Audio")
    ax1.set_ylim(-1, 1)
    ax1.set_title("Sygnał Audio")
    ax1.grid(True, alpha=0.3)

    # Wykres 2: Błąd Predykcji
    (line_pe,) = ax2.plot(np.zeros(max_display), color="orange", linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, max_display)
    ax2.set_title("Błąd Predykcji (Prediction Error)")
    ax2.set_ylabel("Error")
    ax2.grid(True, alpha=0.3)

    # Generator ramek
    def frame_generator():
        for i in range(0, len(sequence), hop_size):
            frame = sequence[i : i + hop_size]
            if len(frame) < hop_size:
                frame = np.pad(frame, (0, hop_size - len(frame)))
            yield frame

    frame_gen = frame_generator()

    def update(frame_num):
        try:
            # Pobierz następną ramkę
            frame = next(frame_gen)

            # Odtwórz dźwięk (blokująco dla synchronizacji)
            stream.write(frame)

            # Przetwórz przez agenta
            _, pe = agent.process(frame)

            # Aktualizuj dane
            pe_history.append(pe)
            display_data.append(pe)

            # Aktualizuj wykresy
            line_audio.set_ydata(frame)

            if len(display_data) > 0:
                line_pe.set_data(range(len(display_data)), list(display_data))
                if max(display_data) > 0.1:
                    ax2.set_ylim(0, max(display_data) * 1.2)

            return line_audio, line_pe

        except StopIteration:
            print("\nKoniec sekwencji.")
            if save_csv:
                save_results_to_csv("results_metronome_realtime.csv", pe_history)
            plt.close(fig)
            return line_audio, line_pe

    ani = animation.FuncAnimation(
        fig, update, interval=1, blit=True, cache_frame_data=False
    )
    plt.show()

    stream.stop()
    stream.close()


def main():
    parser = argparse.ArgumentParser(description="Eksperyment Metronomu - BioAgent4")
    parser.add_argument(
        "--bpm", type=int, default=60, help="Tempo (uderzenia na minutę)"
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Czas trwania w sekundach"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "realtime"],
        default="offline",
        help="Tryb eksperymentu",
    )
    parser.add_argument(
        "--save-csv", action="store_true", help="Zapisz wyniki do pliku CSV"
    )

    args = parser.parse_args()

    # Inicjalizacja Agenta
    print("Inicjalizacja BioAgent4...")
    agent = BioAgent4()
    hop_size = int(FS * HOP_MS / 1000)

    print(f"=== Eksperyment Metronomu ===")
    print(f"BPM: {args.bpm}")
    print(f"Czas: {args.duration}s")
    print(f"Tryb: {args.mode}")
    print(f"Generowanie sygnału...")

    # Generuj sekwencję
    metronome = create_metronome_sequence(bpm=args.bpm, duration_sec=args.duration)
    print(f"Wygenerowano {len(metronome)} próbek ({len(metronome)/FS:.1f}s)")

    if args.mode == "offline":
        run_offline(agent, metronome, hop_size, args.save_csv)
    else:
        run_realtime(agent, metronome, hop_size, args.save_csv)


if __name__ == "__main__":
    main()
