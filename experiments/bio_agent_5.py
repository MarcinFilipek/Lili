import queue
import time
import threading
import numpy as np
import sounddevice as sd
from hybrid_frequency_entrainment import HybridFrequencyEntrainment
from brain_rythm import BrainRhythm
from hebbian_memory import HebbianMemory

# Global Brain Rhythms
RHYTHMS = {
    # "gamma": BrainRhythm(0), 40Hz from audio
    "theta": BrainRhythm(0),
    "delta": BrainRhythm(0),
}

# Values for easy access
BRAIN_STATE = {"gamma": 0.0, "theta": 0.0, "delta": 0.0}


def update_brain_state_loop():
    """
    Wątek aktualizujący rytmy z akumulacją fazy.
    """
    while True:
        for name, rhythm in RHYTHMS.items():
            rhythm.update()
            BRAIN_STATE[name] = rhythm.value

        # Wysoka częstotliwość odświeżania dla płynności
        time.sleep(0.001)


def start_global_rhythms():
    print("Startuje globalne rytmy mózgowe (Phase Accumulator)...")
    t = threading.Thread(target=update_brain_state_loop, daemon=True)
    t.start()


class BioAgent5:
    def __init__(self, echoic_memory_size=20, working_memory_size=7):
        self.n_sensors = 32
        self.theta = np.zeros(self.n_sensors)

        # gamma buffer
        self.echoic_memory_size = echoic_memory_size
        self.echoic_memory = np.zeros((self.echoic_memory_size, self.n_sensors))

        # theta buffer
        self.working_memory_size = working_memory_size
        self.working_memory = np.zeros((self.working_memory_size, self.n_sensors))

        # hebbian memory
        self.hebbian_memory = HebbianMemory(self.n_sensors)

        # State flags for processing
        self._theta_processed = True
        self._delta_processed = True

    def hear(self, frame):
        # Sprawdź długość okna
        if len(frame) < 2048:
            # Padding zerami jeśli ramka jest za krótka dla FFT n=2048
            frame = np.pad(frame, (0, 2048 - len(frame)))

        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=2048)
        power = np.abs(spectrum) ** 2
        bins_per_band = len(power) // self.n_sensors

        # Bezpieczne reshape (ucinamy nadmiarowe biny)
        n_bins = self.n_sensors * bins_per_band
        bands = power[:n_bins].reshape(self.n_sensors, bins_per_band).sum(axis=1)
        signal = np.log1p(bands)
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)  # Adaptacja
        if np.max(activations) > 0:
            activations /= np.max(activations) + 1e-9

        return activations

    def process_theta_frame(self):
        val = RHYTHMS["theta"].value
        if val > 0.95 and not self._theta_processed:
            self._theta_processed = True
            print("Processing theta frame")
            echoic_memory_mean = self.echoic_memory.mean(axis=0)

            self.working_memory = np.roll(self.working_memory, shift=-1, axis=0)
            self.working_memory[-1] = echoic_memory_mean

        elif val < 0:
            self._theta_processed = False

    def process_gamma_frequency(self, frame):
        self.echoic_memory = np.roll(self.echoic_memory, shift=-1, axis=0)
        self.echoic_memory[-1] = frame

    def process_delta_frequency(self):
        val = RHYTHMS["delta"].value
        if val > 0.95 and not self._delta_processed:
            self._delta_processed = True
            print("Processing delta frequency")

            sequence = []
            for i in range(self.working_memory_size):
                vec = self.working_memory[i]
                if np.any(vec):  # jeśli nie zerowy
                    sequence.append(vec.copy())

            if len(sequence) > 1:
                print(f"[CONSOLIDATE] Sequence length: {len(sequence)}\n")
                self.hebbian_memory.learn_sequence(sequence)

                self.working_memory.fill(0)

        elif val < 0:
            self._delta_processed = False


def main():
    # Audio params
    FS = 16000
    HOP_MS = 25  # Zmniejszone dla lepszej rozdzielczości czasowej

    hop_size = int(FS * HOP_MS / 1000)

    audio_q = queue.Queue()

    # Uruchomienie globalne
    start_global_rhythms()

    agent = BioAgent5(echoic_memory_size=20, working_memory_size=7)
    entrainer = HybridFrequencyEntrainment(hop_duration_ms=HOP_MS)

    def audio_callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    print("Startuje nasluch... (Ctrl + C aby przerwac)")
    try:
        with sd.InputStream(
            samplerate=FS, channels=1, callback=audio_callback, blocksize=hop_size
        ):
            S_current = None
            while True:
                # Odczytanie nowych danych audio na podstawie częstotliwości
                if not audio_q.empty():
                    samples = audio_q.get().flatten()

                    if audio_q.qsize() > 1:
                        print("Audio queue is full!")

                    # 1. Słuchanie (Sensory)
                    S_current = agent.hear(samples)
                    agent.process_gamma_frequency(S_current)

                    freq, conf = entrainer.get_theta_frequency(S_current, time.time())

                    # 2. Aktualizacja celu rytmu Theta (jeśli pewność > 0.9 i częstotliwość się zmieniła)
                    # if conf > 0.9 and freq != RHYTHMS["theta"].target_freq:
                    #     if RHYTHMS["theta"].current_freq == 0:
                    #         RHYTHMS["theta"].start(freq)
                    if not RHYTHMS["theta"].is_started():
                        RHYTHMS["theta"].start(4)
                        RHYTHMS["delta"].start(1)

                # print(BRAIN_STATE)
                agent.process_theta_frame()
                agent.process_delta_frequency()

    except KeyboardInterrupt:
        print("\nZatrzymano.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
