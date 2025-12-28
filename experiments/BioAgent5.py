import queue
import time
import threading
import numpy as np
import sounddevice as sd
from hybrid_frequency_entrainment import HybridFrequencyEntrainment

"""
Poziom gamma (szybki, ~50ms):
  - Pojedyncze nuty: C, D, E, F, G

Poziom theta (średni, ~200ms):  
  - Motywy: C-E-G (akord), D-F-A (akord)
  - Krótkie frazy: C-D-E-D-C

Poziom delta (wolny, ~1-2s):
  - Sekwencje akordów: I-IV-V-I (progresja)
  - Melodie: konkretna sekwen
"""
# Global Brain State
BRAIN_STATE = {"gamma": 0.0, "theta": 0.0, "delta": 0.0}


def update_brain_state_loop():
    """
    Jeden wspólny wątek aktualizujący wszystkie rytmy.
    Zapewnia synchronizację i większą płynność (częstsze odświeżanie).
    """
    params = [("gamma", 20.0), ("theta", 5.0), ("delta", 2.0)]

    while True:
        t = time.time()
        for name, freq in params:
            # Obliczamy wartość sinusa
            BRAIN_STATE[name] = np.sin(2 * np.pi * freq * t)

        # Krótszy sleep dla płynności (np. 1ms = 1000Hz)
        time.sleep(0.001)


def start_global_rhythms():
    print("Startuje globalne rytmy mózgowe (zintegrowany wątek)...")
    t = threading.Thread(target=update_brain_state_loop, daemon=True)
    t.start()


class BioAgent5:
    def __init__(self):
        self.n_sensors = 32
        self.theta = np.zeros(self.n_sensors)

    def hear(self, frame):
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=512)
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


def main():
    # Audio params
    FS = 16000
    HOP_MS = 50

    hop_size = int(FS * HOP_MS / 1000)

    audio_q = queue.Queue()

    # Uruchomienie globalne
    start_global_rhythms()

    agent = BioAgent5()
    entrainer = HybridFrequencyEntrainment(hop_duration_ms=HOP_MS)

    def audio_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    print("Startuje nasluch... (Ctrl + C aby przerwac)")
    try:
        with sd.InputStream(
            samplerate=FS, channels=1, callback=audio_callback, blocksize=hop_size
        ):
            while True:
                # Print statusu (opcjonalnie, żeby widzieć że działa)
                # print(
                #     f"G: {BRAIN_STATE['gamma']:.2f} | T: {BRAIN_STATE['theta']:.2f} | D: {BRAIN_STATE['delta']:.2f}"
                # )
                if not audio_q.empty():
                    new_samples = audio_q.get().flatten()
                    freq, conf = entrainer.get_theta_frequency(new_samples, time.time())

    except KeyboardInterrupt:
        print("\nZatrzymano.")
    except Exception as e:
        print("\nError: {e}")


if __name__ == "__main__":
    main()
