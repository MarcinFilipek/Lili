import sounddevice as sd
import numpy as np
import queue

FS = 16000
HOP_MS = 80

class BioAgent3():
    def __init__(self):
        self.n_sensors = 32
        self.decay_rate = 0.2
        self.lr = 0.01

        # bierząca aktywność sensorów audio
        self.S = np.zeros(self.n_sensors)
        # wektor "sladów". Informacja o tym co działo się ułamek sękundy wcześniej
        self.T = np.zeros(self.n_sensors)
        self.T *= self.decay_rate
        # macierz wag
        self.W = np.random.normal(0, 0.2, (self.n_sensors, self.n_sensors))
        self.theta = np.zeros(self.n_sensors)

    def _hear(self, frame):
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=512)
        power = np.abs(spectrum)**2
        bins_per_band = len(power) // self.n_sensors
        bands = power[:self.n_sensors * bins_per_band].reshape(self.n_sensors, bins_per_band).sum(axis=1)
        signal = np.log1p(bands)
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)  # Adaptacja
        if np.max(activations) > 0:
            activations /= (np.max(activations) + 1e-9)

        return activations

    def process(self, frame):
        sensors_activations = self._hear(frame)
        self.T = np.maximum(self.T, sensors_activations)
        self.W += self.lr * np.outer(self.T, sensors_activations)
        
        return self._hear(frame)
        

def main():
    hop_size = int(FS * HOP_MS / 1000)
    audio_q = queue.Queue()
    agent = BioAgent3()

    def audio_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    print("Start systemu... (Ctrl + C aby przerwac)")
    try:
        with sd.InputStream(samplerate=FS, channels=1, callback=audio_callback, blocksize=hop_size, dtype='float32'):
            while True:
                if not audio_q.empty():
                    frame = audio_q.get().flatten()
                    agent.S = agent.process(frame)
                    print(agent.S)

    except KeyboardInterrupt:
        print("\nAgent stopped")

if __name__ == "__main__":
    main()