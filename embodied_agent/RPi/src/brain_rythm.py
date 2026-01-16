"""
Brain Rhythm - Oscillatory Control System
Generates sinusoidal rhythms at specified frequencies

Used for theta (4 Hz) and delta (0.5-1 Hz) timing control
"""

import numpy as np


class BrainRhythm:
    """
    Simple rhythm oscillator

    Generates sinusoidal oscillations at target frequency
    Used for gating theta and delta memory operations
    """

    def __init__(self, freq=0):
        """
        Args:
            freq: Initial frequency (Hz), 0 = not started
        """
        self.target_freq = freq
        self.current_freq = freq
        self.phase = 0
        self.value = 0
        self._started = False

    def start(self, freq):
        """
        Start oscillation at specified frequency

        Args:
            freq: Target frequency in Hz (e.g., 4.0 for theta)
        """
        self.target_freq = freq
        self.current_freq = freq
        self._started = True

    def is_started(self):
        """Check if rhythm is active"""
        return self._started

    def update(self):
        """
        Update phase and value

        Should be called at high frequency (e.g., 1000 Hz / 1ms intervals)
        to maintain smooth oscillation
        """
        if self._started:
            dt = 0.001  # 1ms time step
            self.phase += 2 * np.pi * self.current_freq * dt
            self.phase = self.phase % (2 * np.pi)
            self.value = np.sin(self.phase)
