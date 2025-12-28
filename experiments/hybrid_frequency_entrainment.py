import numpy as np
from collections import deque


class HybridFrequencyEntrainment:
    def __init__(self, hop_duration_ms=80):
        # Historia
        self.onset_times = deque(maxlen=8)
        self.energy_history = deque(maxlen=30)

        # Stan
        self.theta_freq = 5.0
        self.confidence = 0.0  # jak pewni jesteśmy częstotliwości

        # Parametry
        self.hop_duration = hop_duration_ms / 1000.0  # sekund
        self.adapt_fast = 0.2  # gdy confident
        self.adapt_slow = 0.05  # gdy uncertain

    def get_theta_frequency(self, S_current, current_time):
        """Zwraca (frequency, confidence)"""

        # 1. Wykryj onset
        energy = np.sum(S_current**2)
        self.energy_history.append(energy)

        if len(self.energy_history) >= 2:
            energy_prev = self.energy_history[-2]
            energy_increase = energy - energy_prev

            # Adaptive threshold (na podstawie variance)
            if len(self.energy_history) > 10:
                energy_std = np.std(list(self.energy_history))
                threshold = 2.0 * energy_std
            else:
                threshold = 0.1

            onset_detected = energy_increase > threshold

            if onset_detected:
                self.onset_times.append(current_time)

        # 2. Oblicz częstotliwość z onsetów
        if len(self.onset_times) >= 3:
            # IOIs
            intervals = []
            for i in range(1, len(self.onset_times)):
                intervals.append(self.onset_times[i] - self.onset_times[i - 1])

            # Median i std
            median_ioi = np.median(intervals)
            std_ioi = np.std(intervals)

            # Confidence = jak regularne są intervaly
            # CV (coefficient of variation) = std/mean
            cv = std_ioi / (median_ioi + 1e-9)
            self.confidence = np.exp(-cv * 2)  # 0-1, wyższe gdy regularne

            # Measured frequency
            measured_freq = 1.0 / median_ioi
            measured_freq = np.clip(measured_freq, 0.5, 8.0)

            # Adapt rate depends on confidence
            adapt_rate = self.adapt_fast * self.confidence + self.adapt_slow * (
                1 - self.confidence
            )

            self.theta_freq += adapt_rate * (measured_freq - self.theta_freq)

        else:
            # Nie mamy jeszcze danych, decay confidence
            self.confidence *= 0.95

        return self.theta_freq, self.confidence
