import numpy as np
import matplotlib.pyplot as plt


def main():
    duration = 1.0  # seconds
    fs = 1000  # sampling rate (Hz) for smooth plotting
    t = np.linspace(0, duration, int(duration * fs))

    # Frequencies
    freq_gamma = 20.0
    freq_theta = 5.0
    freq_delta = 2.0

    # Generate signals
    # Using sin(2 * pi * f * t) as in the BioAgent implementation
    gamma = np.sin(2 * np.pi * freq_gamma * t)
    theta = np.sin(2 * np.pi * freq_theta * t)
    delta = np.sin(2 * np.pi * freq_delta * t)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, gamma, color="purple")
    plt.title(f"Gamma Rhythm ({freq_gamma} Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(0, duration)  # Full 10 seconds might be crowded for Gamma, but requested.

    plt.subplot(3, 1, 2)
    plt.plot(t, theta, color="blue")
    plt.title(f"Theta Rhythm ({freq_theta} Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(0, duration)

    plt.subplot(3, 1, 3)
    plt.plot(t, delta, color="green")
    plt.title(f"Delta Rhythm ({freq_delta} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(0, duration)

    plt.tight_layout()
    plt.savefig("brain_rhythms_10s.png")
    print("Plot saved to brain_rhythms_10s.png")
    plt.show()


if __name__ == "__main__":
    main()
