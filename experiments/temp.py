"""
BioAgent5 - IMPROVED VERSION

Key changes:
1. Echoic memory stores RAW AUDIO (not features)
2. Long-window FFT (500ms) for better frequency resolution
3. Peak detection for sparse coding
4. Theta samples from recent window (not full buffer average)
"""

import numpy as np


class BioAgent5Improved:
    def __init__(self, echoic_memory_size=20, working_memory_size=7):
        self.n_sensors = 32
        self.theta = np.zeros(self.n_sensors)

        # === CHANGED: Echoic memory stores RAW AUDIO ===
        # 20 frames × 400 samples/frame = 8000 samples = 500ms @ 16kHz
        self.hop_size = 400  # 25ms @ 16kHz
        self.echoic_memory_size = echoic_memory_size
        self.echoic_memory_raw = np.zeros(echoic_memory_size * self.hop_size)

        # Working memory (stores features, not raw audio)
        self.working_memory_size = working_memory_size
        self.working_memory = np.zeros((working_memory_size, self.n_sensors))

        # State flags for processing
        self._theta_processed = True
        self._delta_processed = True

    def hear(self, frame):
        """
        Process audio frame with LONG-WINDOW FFT

        Changes from original:
        - Uses full echoic buffer (500ms) instead of single frame (25ms)
        - Frequency resolution: 2Hz instead of 8Hz
        - Peak detection for sparse coding

        Args:
            frame: audio samples (400 samples, 25ms @ 16kHz)

        Returns:
            activations: sparse peak-based frequency activations (32D)
        """
        # Pad if needed
        if len(frame) < self.hop_size:
            frame = np.pad(frame, (0, self.hop_size - len(frame)))

        # === 1. UPDATE RAW AUDIO BUFFER ===
        # Ring buffer: shift left, add new frame at end
        self.echoic_memory_raw = np.roll(self.echoic_memory_raw, shift=-self.hop_size)
        self.echoic_memory_raw[-self.hop_size :] = frame[: self.hop_size]

        # === 2. LONG-WINDOW FFT ===
        # Use FULL echoic buffer (8000 samples = 500ms)
        window = np.hamming(len(self.echoic_memory_raw))
        windowed = self.echoic_memory_raw * window

        # FFT with high resolution
        n_fft = 8192  # Zero-padding for interpolation
        spectrum = np.fft.rfft(windowed, n=n_fft)
        power = np.abs(spectrum) ** 2

        # Frequency resolution: 16000 / 8192 = 1.95 Hz per bin
        # 300Hz → bin 154, 600Hz → bin 308, 1000Hz → bin 513
        # Much better separation than before!

        # === 3. CONVERT TO FREQUENCY BANDS ===
        bins_per_band = len(power) // self.n_sensors
        n_bins = self.n_sensors * bins_per_band
        bands = power[:n_bins].reshape(self.n_sensors, bins_per_band).sum(axis=1)
        signal = np.log1p(bands)

        # === 4. ADAPTIVE THRESHOLD ===
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)

        if np.max(activations) == 0:
            return activations

        # === 5. PEAK DETECTION (Sparse Coding) ===
        min_prominence = 0.3  # Peak must be 30% above neighbors
        max_peaks = 6  # Keep top 6 peaks

        peaks = []
        for i in range(1, len(activations) - 1):
            # Check if local maximum
            if (
                activations[i] > activations[i - 1]
                and activations[i] > activations[i + 1]
            ):
                # Check prominence
                prominence = activations[i] - max(
                    activations[i - 1], activations[i + 1]
                )
                if prominence > min_prominence * np.max(activations):
                    peaks.append((i, activations[i]))

        # Sort by height, keep top max_peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        peaks = peaks[:max_peaks]

        # === 6. CREATE SPARSE ACTIVATION ===
        sparse = np.zeros_like(activations)

        for peak_idx, peak_val in peaks:
            # Activate peak
            sparse[peak_idx] = peak_val

            # Activate neighbors with lower strength (more realistic)
            if peak_idx > 0:
                sparse[peak_idx - 1] = peak_val * 0.3
            if peak_idx < len(sparse) - 1:
                sparse[peak_idx + 1] = peak_val * 0.3

        # Normalize
        if np.max(sparse) > 0:
            sparse /= np.max(sparse) + 1e-9

        return sparse

    def process_theta_frame(self):
        """
        Theta-gated sampling: capture snapshot from echoic memory

        CHANGED: Instead of averaging full 500ms buffer,
        sample from RECENT window (last 75ms)
        """
        from brain_rythm import RHYTHMS

        val = RHYTHMS["theta"].value
        if val > 0.95 and not self._theta_processed:
            self._theta_processed = True

            # === CHANGED: Sample RECENT echoic, not full average ===
            # Extract features from recent raw audio (last 3 hops = 75ms)
            recent_audio = self.echoic_memory_raw[-3 * self.hop_size :]

            # Quick FFT on recent window
            window = np.hamming(len(recent_audio))
            windowed = recent_audio * window
            spectrum = np.fft.rfft(windowed, n=2048)
            power = np.abs(spectrum) ** 2

            bins_per_band = len(power) // self.n_sensors
            n_bins = self.n_sensors * bins_per_band
            bands = power[:n_bins].reshape(self.n_sensors, bins_per_band).sum(axis=1)
            recent_features = np.log1p(bands)

            # Normalize
            if np.max(recent_features) > 0:
                recent_features /= np.max(recent_features)

            # Store in working memory
            self.working_memory = np.roll(self.working_memory, shift=-1, axis=0)
            self.working_memory[-1] = recent_features

        elif val < 0:
            self._theta_processed = False

    def process_gamma_frequency(self, frame):
        """
        This is now a no-op since echoic memory is updated in hear()

        Keep for compatibility with existing code
        """
        pass

    def process_delta_frequency(self):
        """
        Delta-gated consolidation (unchanged)
        """
        from brain_rythm import RHYTHMS

        val = RHYTHMS["delta"].value
        if val > 0.95 and not self._delta_processed:
            self._delta_processed = True

            # Collect non-zero vectors from working memory
            sequence = []
            for i in range(self.working_memory_size):
                vec = self.working_memory[i]
                if np.any(vec):
                    sequence.append(vec.copy())

            if len(sequence) > 1:
                print(f"[CONSOLIDATE] Sequence length: {len(sequence)}")

                # Learn sequence (requires hebbian_memory to be set)
                if hasattr(self, "hebbian_memory"):
                    self.hebbian_memory.learn_sequence(sequence)

                # Clear working memory
                self.working_memory.fill(0)

        elif val < 0:
            self._delta_processed = False


# Usage example:
#
# from bio_agent_5_improved import BioAgent5Improved
# from hebbian_memory_wta import HebbianMemoryWTA
#
# agent = BioAgent5Improved(echoic_memory_size=20, working_memory_size=7)
# agent.hebbian_memory = HebbianMemoryWTA(dim=32)
#
# # Then use as before:
# S_current = agent.hear(samples)
# agent.process_theta_frame()
# agent.process_delta_frequency()
