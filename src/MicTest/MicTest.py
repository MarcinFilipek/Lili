import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

FS = 44100
CHUNK = 2048

plt.ion()
fig, ax = plt.subplots()
x_freq = np.fft.rfftfreq(CHUNK, 1/FS)
line, = ax.plot(x_freq, np.zeros(len(x_freq)), color='magenta')

rms_text = ax.text(0.75, 0.9, '', transform=ax.transAxes,
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

ax.set_title("Analiza spectrum mikrofonu (fft)")
ax.set_xlim(0, 5000)
ax.set_ylim(0, 50)
ax.set_xlabel("Hz")
ax.set_ylabel("Amplituda")


def audio_callback(indata, fremes, time, status):
    if status:
        print(status)

    rms = np.sqrt(np.mean(indata**2))
    if rms < 0.001:
        print(f"RMS: {rms}")


    magnitude = np.abs(np.fft.rfft(indata[:, 0]))
    
    line.set_ydata(magnitude)

    rms_text.set_text(f"RMS: {rms:.4f}")

try:
    with sd.InputStream(channels=1, samplerate=FS, blocksize=CHUNK, callback=audio_callback):
        print("Nagrywanie...")
        while plt.fignum_exists(fig.number):
            plt.pause(0.01)
except KeyboardInterrupt:
    print("Zatrzymano")
except Exception as e:
    print(e)


# export QT_QPA_PLATFORM=xcb