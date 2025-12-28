import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

FS = 16000
HOP_MS = 80


class BioAgent4:
    def __init__(self):
        self.n_sensors = 32

        # wektor "sladów". Informacja o tym co działo się ułamek sękundy wcześniej
        self.T = np.zeros(self.n_sensors)
        self.theta = np.zeros(self.n_sensors)

        # macierz wag krótkoterminowych
        self.W_short = np.random.normal(0, 0.2, (self.n_sensors, self.n_sensors))
        self.lr_short = 0.01

        # macierz wag długoterminowych
        self.W_long = np.zeros((self.n_sensors, self.n_sensors))
        self.lr_long = 0.0001

        # macierz predykcyjna
        self.W_pred = np.random.normal(0, 0.1, (self.n_sensors, self.n_sensors))
        self.lr_pred = 0.001

        # bierząca aktywność sensorów audio
        self.S = np.zeros(self.n_sensors)
        # poprzednia aktywność sensorów
        self.S_prev = np.zeros(self.n_sensors)

        # próg konsolidacji
        self.consolidation_threshold = 0.5

        # Homeostaza wag
        self.weight_decay = 0.0001
        self.max_weight = 5.0

    def _hear(self, frame):
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=2048)
        power = np.abs(spectrum) ** 2
        bins_per_band = len(power) // self.n_sensors
        bands = (
            power[: self.n_sensors * bins_per_band]
            .reshape(self.n_sensors, bins_per_band)
            .sum(axis=1)
        )
        signal = np.log1p(bands)
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)  # Adaptacja
        if np.max(activations) > 0:
            activations /= np.max(activations) + 1e-9

        return activations

    def predict_next(self, S_current):
        # przewidywanie aktywności sensorów
        S_pred = np.tanh(self.W_pred @ S_current)
        return S_pred

    def compute_prediction_error(self, S_current, S_pred):
        # obliczanie błędu predykcji
        prediction_error = np.mean((S_current - S_pred) ** 2)
        return prediction_error

    def normalize_weights(self):
        """Homeostaza wag - synaptic scaling i weight decay"""
        # Weight decay - słabe wagi zanikają
        self.W_short *= 1 - self.weight_decay
        self.W_long *= 1 - self.weight_decay
        self.W_pred *= 1 - self.weight_decay

        # Clipping - ogranicz maksymalne wartości
        self.W_short = np.clip(self.W_short, -self.max_weight, self.max_weight)
        self.W_long = np.clip(self.W_long, -self.max_weight, self.max_weight)
        self.W_pred = np.clip(self.W_pred, -self.max_weight, self.max_weight)

    def process(self, frame):
        # przetwarzaj dzwięk
        S_current = self._hear(frame)

        # przewiduj następny stan na podstawie poprzedniego
        S_pred = self.predict_next(self.S_prev)

        # oblicz błąd predykcji
        prediction_error = self.compute_prediction_error(S_current, S_pred)

        # aktualizuj ślady (najpierw zanik, potem maximum)
        self.T *= 0.99  # Zanik śladów
        self.T = np.maximum(self.T, S_current)  # Potem maximum z obecną aktywacją

        # ucz pamięć krótkoterminową
        self.W_short += self.lr_short * np.outer(self.T, S_current)

        # ucz pamięć długoterminową (tylko przy dużych błędach predykcji)
        if prediction_error > self.consolidation_threshold:
            self.W_long += self.lr_long * np.outer(self.T, S_current)

        # ucz model predykcyjny (zmniejszaj błąd)
        prediction_error_vector = S_current - S_pred
        self.W_pred += self.lr_pred * np.outer(prediction_error_vector, self.S_prev)

        # zapisz obecny stan jako poprzedni
        self.S_prev = S_current.copy()

        # Homeostaza wag
        self.normalize_weights()

        return S_current, prediction_error


def main():
    hop_size = int(FS * HOP_MS / 1000)
    audio_q = queue.Queue()
    agent = BioAgent4()

    # Historia danych do wizualizacji
    max_history = 100
    prediction_error_history = deque(maxlen=max_history)
    current_frame = np.zeros(hop_size)
    current_activations = np.zeros(agent.n_sensors)

    def audio_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    # Konfiguracja wykresu
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("BioAgent4 - Monitoring Mikrofonu", fontsize=16, fontweight="bold")

    # Wykres 1: Sygnał audio (waveform)
    (line1,) = ax1.plot(np.zeros(hop_size), color="cyan", linewidth=1)
    ax1.set_ylim(-1, 1)
    ax1.set_xlim(0, hop_size)
    ax1.set_ylabel("Amplituda")
    ax1.set_title("Sygnał Audio (Waveform)")
    ax1.grid(True, alpha=0.3)

    # Wykres 2: Aktywacje sensorów (frequency bands)
    bars = ax2.bar(
        range(agent.n_sensors), np.zeros(agent.n_sensors), color="lime", alpha=0.7
    )
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-0.5, agent.n_sensors - 0.5)
    ax2.set_ylabel("Aktywacja")
    ax2.set_xlabel("Sensor (pasmo częstotliwości)")
    ax2.set_title("Aktywacje Sensorów")
    ax2.grid(True, alpha=0.3, axis="y")

    # Wykres 3: Historia błędu predykcji
    (line3,) = ax3.plot([], [], color="orange", linewidth=2)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(0, max_history)
    ax3.set_ylabel("Błąd")
    ax3.set_xlabel("Czas (ramki)")
    ax3.set_title("Błąd Predykcji")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    def update_plot(frame_num):
        nonlocal current_frame, current_activations

        # Przetwarzaj dane audio jeśli dostępne
        if not audio_q.empty():
            frame = audio_q.get().flatten()
            current_frame = frame
            S_current, prediction_error = agent.process(frame)
            current_activations = S_current
            prediction_error_history.append(prediction_error)

            # Aktualizuj wykres waveform
            line1.set_ydata(current_frame)

            # Aktualizuj wykres aktywacji sensorów
            for bar, height in zip(bars, current_activations):
                bar.set_height(height)

            # Aktualizuj wykres błędu predykcji
            if len(prediction_error_history) > 0:
                line3.set_data(
                    range(len(prediction_error_history)), list(prediction_error_history)
                )
                # Dynamiczne skalowanie osi Y
                if max(prediction_error_history) > 0:
                    ax3.set_ylim(0, max(prediction_error_history) * 1.1)

        return line1, *bars, line3

    print("Start systemu... (Zamknij okno wykresu aby przerwac)")

    try:
        with sd.InputStream(
            samplerate=FS,
            channels=1,
            callback=audio_callback,
            blocksize=hop_size,
            dtype="float32",
        ):
            ani = animation.FuncAnimation(
                fig, update_plot, interval=50, blit=True, cache_frame_data=False
            )
            plt.show()
    except KeyboardInterrupt:
        print("\nAgent stopped")
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
