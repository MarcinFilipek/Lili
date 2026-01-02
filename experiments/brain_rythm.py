import numpy as np
import time


class BrainRhythm:
    def __init__(self, start_freq):
        self.current_freq = start_freq
        self.target_freq = start_freq
        self.phase = 0.0
        self.value = 0.0
        self.last_time = time.perf_counter()

    def update(self):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now

        # Akumulacja Fazy (rozwiązuje problem skoków przy zmianie freq)
        self.phase += 2 * np.pi * self.current_freq * dt
        self.phase %= 2 * np.pi

        self.value = np.sin(self.phase)

        # Logika aktualizacji częstotliwości tylko w szczytach (-1 lub 1)
        # Cosinus (pochodna) jest tam bliski zeru, więc zmiana jest gładka.
        # Sprawdzamy czy sin(phase) jest bliski 1 lub -1 (np. > 0.95)
        if abs(self.value) > 0.95:
            if self.current_freq != self.target_freq:
                # Płynne przejście (interpolacja lub skok w bezpiecznym punkcie)
                # Tutaj robimy skok, bo w szczycie jest to bezpieczne
                self.current_freq = self.target_freq
                self.current_freq = self.target_freq
                # print(f"Freq updated to {self.current_freq} Hz")

    def start(self, freq):
        """Uruchamia rytm natychmiastowo z nową częstotliwością."""
        self.current_freq = freq
        self.target_freq = freq
        self.phase = np.pi / 2  # Start od szczytu
        self.value = 1.0

    def is_started(self):
        return self.target_freq > 0
