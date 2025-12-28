import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
import time
from matplotlib.animation import FuncAnimation
import queue
from BioAgent4 import BioAgent4
from hybrid_frequency_entrainment import HybridFrequencyEntrainment


# Globalne zmienne do komunikacji
audio_queue = queue.Queue()
FS = 16000
HOP_MS = 25  # Match default in class
HOP_SIZE = int(FS * HOP_MS / 1000)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def main():
    # Inicjalizacja entrainera
    entrainer = HybridFrequencyEntrainment(hop_duration_ms=HOP_MS)

    agent = BioAgent4()

    # Setup wykresu
    fig, (ax_wave, ax_info) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Wykres fali
    x_len = 500
    xs = np.linspace(0, 5, x_len)
    ys = np.zeros(x_len)
    (line,) = ax_wave.plot(xs, ys, lw=2)
    ax_wave.set_ylim(-1.2, 1.2)
    ax_wave.set_title("Theta Frequency Sine Wave")
    ax_wave.grid(True)

    # Info text
    info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center", fontsize=16)
    ax_info.axis("off")

    # Stan wizualizacji
    viz_state = {"phase": 0.0, "last_time": time.time(), "freq": 5.0, "conf": 0.0}

    def update(frame):
        current_time_wall = time.time()

        # 1. Przetwórz nowe audio (jeśli jest)
        while not audio_queue.empty():
            audio_data = audio_queue.get().flatten()
            S_current = agent._hear(audio_data)

            # Przekazujemy time.time() jako current_time dla uproszczenia
            # W idealnym świecie bralibysmy czas z audio callbacku
            freq, conf = entrainer.get_theta_frequency(S_current, current_time_wall)

            viz_state["freq"] = freq
            viz_state["conf"] = conf

        # 2. Aktualizuj fazę i wygeneruj falę do wizualizacji
        dt = current_time_wall - viz_state["last_time"]
        viz_state["last_time"] = current_time_wall

        # Update phase
        # phase += 2 * pi * f * dt
        delta_phase = 2 * np.pi * viz_state["freq"] * dt
        viz_state["phase"] += delta_phase

        # Generuj sinus falę "w przyszłość" lub "przesuwającą się"
        # Żeby to wyglądało jak płynący czas:
        # y(t) = sin(phase + 2*pi*f*t_offset)
        # Gdzie t_offset to oś X wykresu

        # Opcja A: Przesuwająca się fala (traveling wave)
        # xs to czas od 0 do 5s
        # phase to faza w t=0 (lewa krawędź)

        wave_ys = np.sin(viz_state["phase"] + 2 * np.pi * viz_state["freq"] * xs)

        # Aktualizacja wykresu
        line.set_ydata(wave_ys)

        color = "green" if viz_state["conf"] > 0.5 else "red"
        line.set_color(color)

        info_text.set_text(
            f"Frequency: {viz_state['freq']:.2f} Hz\nConfidence: {viz_state['conf']:.2f}"
        )

        return line, info_text

    # Start audio stream
    stream = sd.InputStream(
        samplerate=FS, blocksize=HOP_SIZE, channels=1, callback=audio_callback
    )

    with stream:
        print("Listening... Close plot window to stop.")
        ani = FuncAnimation(
            fig, update, interval=30, blit=False, cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    main()
