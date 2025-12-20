import numpy as np
import sounddevice as sd
import queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traceback
import time
import os
import json


class BioAgent2():
    def __init__(self, fs=16000):
        self.fs = fs
        self.n_sensors = 32
        self.n_associative = 64
        self.n_motor = 4

        self.files = {
            'W_in': 'brain_W_in.npy',
            'W_res': 'brain_W_res.npy',
            'W_out': 'brain_W_out.npy',
            'theta': 'brain_theta.npy'
        }

        self.leak_rate = 0.2
        self.lr_motor = 0.05
        self.lr_internal = 0.02

        # Weights (Init)
        self.W_in = np.random.normal(
            0, 0.2, (self.n_sensors, self.n_associative))
        self.W_res = np.random.normal(
            0, 0.1, (self.n_associative, self.n_associative))
        self.W_out = np.random.normal(
            0, 0.1, (self.n_associative, self.n_motor))
        self.theta = np.zeros(self.n_sensors)

        self.load_brain()


        # Jeśli wagi są nowe, robimy skalowanie spektralne (tylko dla W_res)
        if not os.path.exists(self.files['W_res']):
            # Rzadkie polaczenia w mozgu (Sparsity 80%)
            mask = np.random.rand(self.n_associative, self.n_associative) > 0.2
            self.W_res[mask] = 0.0
            # Skalowanie spektralne(Echo state property)
            rho = max(abs(np.linalg.eigvals(self.W_res)))
            self.W_res = self.W_res / (rho + 1e-5) * 0.90

    
        # Stany
        self.state = np.zeros(self.n_associative)
        self.motor_actions = ["FORWARD", "BACK", "LEFT", "RIGHT"]
        self.eligibility = np.zeros_like(self.W_out)

        # Baseline głosu
        self.baseline = {
            'cent_mean': 1800.0,
            'zcr_mean': 0.15,
            'pitch_mean': 200.0
        }
        self.skip_baseline_calibration = self.load_voice_baseline()
        
        # Audio / Emocje
        self.energy_treshold = 0.02
        self.is_recording = False
        self.dynamic_buffer = []
        self.silence_counter = 0
        self.max_silence_frames = 10    # Ile klatek ciszy konczy nagrywanie

        # Wizualizacja
        self.last_sensors = np.zeros(self.n_sensors)
        self.last_motors = np.zeros(self.n_motor)

    def load_voice_baseline(self):
        if os.path.exists('voice_baseline.json'):
            with open('voice_baseline.json', 'r') as f:
                self.baseline = json.load(f)
            print(">>> Wczytano charakterystykę głosu (Baseline).")
            return True
        else:
            print(">>> Brak baseline głosu! Używam wartości domyślnych (zalecana kalibracja).")
            return False

    def save_brain(self):
        """Zapisuje aktualny stan sieci do plików .npy"""
        try:
            np.save(self.files['W_in'], self.W_in)
            np.save(self.files['W_res'], self.W_res)
            np.save(self.files['W_out'], self.W_out)
            np.save(self.files['theta'], self.theta)
            print(">>> Pamięć długotrwała zaktualizowana (zapisano wagi).")
        except Exception as e:
            print(f"Błąd zapisu pamięci: {e}")

    def load_brain(self):
        """Wczytuje wagi, jeśli pliki istnieją"""
        if os.path.exists(self.files['W_out']):
            try:
                self.W_in = np.load(self.files['W_in'])
                self.W_res = np.load(self.files['W_res'])
                self.W_out = np.load(self.files['W_out'])
                self.theta = np.load(self.files['theta'])
                print(">>> Wczytano wyuczoną pamięć z plików .npy")
            except Exception as e:
                print(f"Błąd podczas wczytywania pamięci: {e}")
        else:
            print(">>> Brak zapisanej pamięci. Inicjalizacja nowego mózgu.")

    def _get_pitch(self, signal):
        """Oblicza częstotliwość podstawową (F0) metodą autokorelacji."""
        if len(signal) < 250: return 0
        
        # Detrending (usuwamy składową stałą)
        sig = signal - np.mean(signal)
        
        # Autokorelacja - np.correlate jest szybkie dla małych okien
        corr = np.correlate(sig, sig, mode='full')
        corr = corr[len(corr)//2:]
        
        # Zakres mowy ludzkiej dla 16kHz:
        # 80Hz -> peak przy 200 próbce
        # 400Hz -> peak przy 40 próbce
        d_min, d_max = 40, 200
        
        if len(corr) < d_max: return 0
        
        # Szukamy lokalnego maksimum w zakresie mowy
        search_range = corr[d_min:d_max]
        peak = np.argmax(search_range) + d_min
        
        return self.fs / peak

    def collect_baseline(self, audio_q, duration_sec=30):
        """
        Zbiera próbki głosu użytkownika przez określony czas, 
        aby wyliczyć neutralny punkt odniesienia (baseline).
        """
        print(f"\n[KALIBRACJA] Rozpoczynam zbieranie charakterystyki głosu...")
        print(f"[KALIBRACJA] Mów naturalnie przez najbliższe {duration_sec} sekund.")
        
        start_time = time.time()
        collected_stats = {'cent': [], 'zcr': [], 'pitch': []}
        
        # Obliczamy ile ramek musimy zebrać (np. HOP_MS = 80ms)
        # duration_sec / 0.080
        
        try:
            while time.time() - start_time < duration_sec:
                if not audio_q.empty():
                    chunk = audio_q.get().flatten()
                    
                    # Liczymy tylko jeśli klatka nie jest ciszą (RMS > próg)
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > self.energy_treshold:
                        # 1. Centroid
                        fft_vals = np.abs(np.fft.rfft(chunk))
                        freqs = np.fft.rfftfreq(len(chunk), 1/self.fs)
                        cent = np.sum(fft_vals * freqs) / (np.sum(fft_vals) + 1e-9)
                        
                        # 2. ZCR
                        zcr = np.mean(np.abs(np.diff(np.sign(chunk))) > 0)
                        
                        # 3. Pitch (Autokorelacja)
                        pitch = self._get_pitch(chunk)
                        
                        if pitch > 0: # Zapisujemy tylko poprawne detekcje pitchu
                            collected_stats['cent'].append(cent)
                            collected_stats['zcr'].append(zcr)
                            collected_stats['pitch'].append(pitch)
                            
                            print(f"Postęp: {int((time.time()-start_time)/duration_sec*100)}% | Zebrano: {len(collected_stats['pitch'])} próbek", end='\r')

            # Wyliczamy średnie
            if len(collected_stats['cent']) > 0:
                self.baseline['cent_mean'] = np.mean(collected_stats['cent'])
                self.baseline['zcr_mean'] = np.mean(collected_stats['zcr'])
                self.baseline['pitch_mean'] = np.mean(collected_stats['pitch'])
                
                # Zapisujemy do pliku, żeby nie kalibrować co chwilę
                print("Zapisuję charakterystykę głosu do pliku 'voice_baseline.json'...")
                with open('voice_baseline.json', 'w') as f:
                    json.dump(self.baseline, f)
                
                print(f"\n[KALIBRACJA] Sukces!")
                print(f"Średni Centroid: {self.baseline['cent_mean']:.1f} Hz")
                print(f"Średni Pitch: {self.baseline['pitch_mean']:.1f} Hz")
                print(f"Średni ZCR: {self.baseline['zcr_mean']:.3f}")
            else:
                print("\n[BŁĄD] Nie zebrano wystarczającej ilości danych. Czy mikrofon działa?")

        except Exception as e:
            print(f"\n[BŁĄD] W trakcie kalibracji: {e}")

    def _hear(self, frame):
        '''
        1. Analiza spektralna
        - Sygnal jest poddawany transformacie Fouriera (FFT)
        - Zamiast setek prazkow czestotliwosci, grupujemy je w N_BANDS (np. 32 pasma)
        - Wynik: Wektor 32 liczb, gdzie kazda liczba mowi "jak glosno jest w danym pasmie"
        (basy, srednie, soprany)
        '''
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=512)
        power = np.abs(spectrum)**2
        bins_per_band = len(power) // self.n_sensors
        bands = power[:self.n_sensors * bins_per_band].reshape(self.n_sensors, bins_per_band).sum(axis=1)
        
        signal = np.log1p(bands)

        '''
        2. Aktywacja nuronow i progowanie
        - Od energi kazdego pasma odejmujemy prog theta (bands - theta)
        - Stosujemy funkcje ReLU (np.maximum(0, ...)). Jesli energia jest ponizej progu (szum),
        neuron wysyla 0. Jesli powyzej, wysyla sygnal dodatni.
        - Cel: odszumienie, agent ignoruje ciche dzwieki tla.
        '''
        activations = np.maximum(0, signal - self.theta)
        self.theta += 0.01 * (signal - self.theta)  # Adaptacja
        if np.max(activations) > 0:
            activations /= (np.max(activations) + 1e-9)
        return activations

    def _think(self, sensors_input):
        input_stim = np.dot(sensors_input, self.W_in)
        internal_stim = np.dot(self.state, self.W_res)
        new_state = np.tanh(input_stim + internal_stim)
        self.state = (1 - self.leak_rate) * self.state + \
            self.leak_rate * new_state

        # Samoorganizacja (Hebbian)
        if np.random.rand() < 0.02:
            mask = self.W_res != 0
            outer = np.outer(self.state, self.state)
            delta_w = outer - (self.state**2).reshape(-1, 1) * self.W_res
            self.W_res[mask] += self.lr_internal * delta_w[mask]

        return self.state

    def _act(self, internal_state):
        motor_activations = np.dot(internal_state, self.W_out)
        # Softmax dla stabilnosci
        exp_vals = np.exp(motor_activations - np.max(motor_activations))
        probs = exp_vals / np.sum(exp_vals)
        action_idx = np.random.choice(len(probs), p=probs)
        # Zwrraca prawdopodobienstwo do wizualizacji
        return action_idx, probs

    # def analyze_emotional_feedback(self, audio_data):
    #     y = audio_data
    #     rms = np.sqrt(np.mean(y**2))
    
    #     if rms < self.energy_treshold: 
    #         return 0.0
    
    #     # Szybki ZCR (Zero Crossing Rate) w NumPy
    #     zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
    
    #     # Szybki Centroid (Jasność) w NumPy
    #     fft_vals = np.abs(np.fft.rfft(y))
    #     freqs = np.fft.rfftfreq(len(y), 1/self.fs)
    #     cent_mean = np.sum(fft_vals * freqs) / (np.sum(fft_vals) + 1e-9)
    #     bandwidth = np.sqrt(np.sum(fft_vals * (freqs - cent_mean)**2) / (np.sum(fft_vals) + 1e-9))

    #     # Logika BioAgenta
    #     print(f"FRAZA -> RMS: {rms:.4f}, ZCR: {zcr:.2f}, CENT: {cent_mean:.0f}, BAND: {bandwidth:.2f}")
    #     if cent_mean > 2500 and zcr < 0.2: return 1.0  # Pochwała
    #     if (cent_mean < 1500) or (zcr > 0.3 and rms > 0.05): return -1.0 # Nagana
    #     return 0.0

    def analyze_emotional_feedback(self, audio_data):
        """
        Wykorzystuje zapisany baseline do obliczenia ciągłej wartości emocji.
        Zwraca wartość od -1.0 (kara) do 1.0 (nagroda).
        """
        y = audio_data
        rms = np.sqrt(np.mean(y**2))
        
        # Zabezpieczenie przed ciszą (aby nie analizować szumu)
        if rms < self.energy_treshold:
            return 0.0

        # 1. Pobranie parametrów bieżących
        # --- Centroid (Jasność/Barwa) ---
        fft_vals = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1/self.fs)
        current_cent = np.sum(fft_vals * freqs) / (np.sum(fft_vals) + 1e-9)
        
        # --- ZCR (Szorstkość/Syk) ---
        current_zcr = np.mean(np.abs(np.diff(np.sign(y))) > 0)
        
        # --- Pitch Slope (Kontur wysokości) ---
        mid = len(y) // 2
        p1 = self._get_pitch(y[:mid])
        p2 = self._get_pitch(y[mid:])
        # Obliczamy nachylenie tylko jeśli oba fragmenty mają wykrywalny pitch
        pitch_slope = (p2 - p1) if (p1 > 0 and p2 > 0) else 0

        # 2. Obliczanie znormalizowanych odchyleń (Delt)
        # Wzór: (Wartość_Bieżąca - Baseline) / Baseline
        cent_delta_norm = (current_cent - self.baseline['cent_mean']) / (self.baseline['cent_mean'] + 1e-9)
        zcr_delta_norm = (current_zcr - self.baseline['zcr_mean']) / (self.baseline['zcr_mean'] + 1e-9)
        
        # Pitch slope normalizujemy względem bazowego pitchu (spodziewany zakres zmian)
        pitch_slope_norm = pitch_slope / (self.baseline['pitch_mean'] + 1e-9)

        # 3. Formuła EMOTION
        # w1, w2 - wagi dla pozytywnych cech (jasność, rosnąca intonacja)
        # w3 - waga dla negatywnej cechy (szorstkość/ZCR)
        w1, w2, w3 = 0.5, 0.4, 0.3
        
        raw_score = (w1 * cent_delta_norm) + (w2 * pitch_slope_norm) - (w3 * zcr_delta_norm)

        # 4. Gęstość mowy (Arousal)
        # Sprawdzamy jak dużo energii jest w sygnale ponad tłem
        speech_density = np.sum(np.abs(y) > self.energy_treshold) / len(y)

        # 5. Finalny wynik z nasyceniem (tanh)
        # Mnożnik 5.0-10.0 to czułość (Sensitivity)
        emotion = np.tanh(raw_score * speech_density * 8.0)

        # Opcjonalny monitoring w konsoli
        # print(f"EMO: {emotion:+.2f} [C:{cent_delta_norm:+.2f} P:{pitch_slope_norm:+.2f} Z:{zcr_delta_norm:+.2f}]")

        return emotion

    def _collect_feedback(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk**2))

        if not self.is_recording:
            # Oczekiwanie na start slowa
            if rms > self.energy_treshold:
                print('Slucham Cię ...')
                self.is_recording = True
                self.dynamic_buffer = [audio_chunk]
                self.silence_counter = 0
        else:
            # Nagrywamy trwajace slowo
            self.dynamic_buffer.append(audio_chunk)

            if rms < self.energy_treshold:
                self.silence_counter += 1
            else:
                self.silence_counter = 0

            if self.silence_counter > self.max_silence_frames:
                print('Analizuję co powiedziałeś ...')
                # Laczymy zebrane klatki w jeden sygnal
                recorded_audio = np.concatenate(self.dynamic_buffer)
                # Analizujemy tylko jesli nagranie nie jest za krotkie (np. min 0.2s)
                if len(recorded_audio) > self.fs * 0.2:
                    reward = self.analyze_emotional_feedback(recorded_audio)
                    print(f'>> REWARD: {reward} <<')
                    if reward != 0:
                        self.apply_reward(reward)

                self.is_recording = False
                self.dynamic_buffer = []
                self.silence_counter = 0

    def apply_reward(self, reward):
        '''
        Modulacja wag o oparciu o zebrane eligibility traces.
        '''
        self.W_out += self.lr_motor * reward * self.eligibility
        self.eligibility *= 0.0
        self.save_brain()

    def process_frame(self, raw_frame_audio):
            # Sensory
            sensors_vec = self._hear(raw_frame_audio)
            # Zapis dla wizualizacji
            self.last_sensors = sensors_vec

            # Asocjacja
            brain_state = self._think(sensors_vec)

            # Motoryka
            action_idx, motor_vals = self._act(brain_state)
            # Zapis dla wizualizacji
            self.last_motors = motor_vals

            # Modulacja
            output_spike = np.zeros(self.n_motor)
            output_spike[action_idx] = 1.0
            self.eligibility = 0.9 * self.eligibility + \
                np.outer(brain_state, output_spike)

            # Zbieranie feedbacku
            self._collect_feedback(raw_frame_audio)

            return action_idx


def main():
    # Audio params
    FS = 16000
    HOP_MS = 80
    USE_GUI = False
    SHOW_PROCESS_TIME = False

    hop_size = int(FS * HOP_MS / 1000)

    audio_q = queue.Queue()
    agent = BioAgent2(fs=FS)

    def audio_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    def collect_baseline(self, audio_q, duration_sec=30):
        """
        Zbiera próbki głosu użytkownika przez określony czas, 
        aby wyliczyć neutralny punkt odniesienia (baseline).
        """
        print(f"\n[KALIBRACJA] Rozpoczynam zbieranie charakterystyki głosu...")
        print(f"[KALIBRACJA] Mów naturalnie przez najbliższe {duration_sec} sekund.")
    
        start_time = time.time()
        collected_stats = {'cent': [], 'zcr': [], 'pitch': []}
    
        # Obliczamy ile ramek musimy zebrać (np. HOP_MS = 80ms)
        # duration_sec / 0.080
    
        try:
            while time.time() - start_time < duration_sec:
                if not audio_q.empty():
                    chunk = audio_q.get().flatten()
                    
                    # Liczymy tylko jeśli klatka nie jest ciszą (RMS > próg)
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > self.energy_treshold:
                        # 1. Centroid
                        fft_vals = np.abs(np.fft.rfft(chunk))
                        freqs = np.fft.rfftfreq(len(chunk), 1/self.fs)
                        cent = np.sum(fft_vals * freqs) / (np.sum(fft_vals) + 1e-9)
                        
                        # 2. ZCR
                        zcr = np.mean(np.abs(np.diff(np.sign(chunk))) > 0)
                        
                        # 3. Pitch (Autokorelacja)
                        pitch = get_pitch(chunk)
                        
                        if pitch > 0: # Zapisujemy tylko poprawne detekcje pitchu
                            collected_stats['cent'].append(cent)
                            collected_stats['zcr'].append(zcr)
                            collected_stats['pitch'].append(pitch)
                            
                            print(f"Postęp: {int((time.time()-start_time)/duration_sec*100)}% | Zebrano: {len(collected_stats['pitch'])} próbek", end='\r')

            # Wyliczamy średnie
            if len(collected_stats['cent']) > 0:
                self.baseline['cent_mean'] = np.mean(collected_stats['cent'])
                self.baseline['zcr_mean'] = np.mean(collected_stats['zcr'])
                self.baseline['pitch_mean'] = np.mean(collected_stats['pitch'])
                
                # Zapisujemy do pliku, żeby nie kalibrować co chwilę
                with open('voice_baseline.json', 'w') as f:
                    json.dump(self.baseline, f)
                
                print(f"\n[KALIBRACJA] Sukces!")
                print(f"Średni Centroid: {self.baseline['cent_mean']:.1f} Hz")
                print(f"Średni Pitch: {self.baseline['pitch_mean']:.1f} Hz")
                print(f"Średni ZCR: {self.baseline['zcr_mean']:.3f}")
            else:
                print("\n[BŁĄD] Nie zebrano wystarczającej ilości danych. Czy mikrofon działa?")

        except Exception as e:
            print(f"\n[BŁĄD] W trakcie kalibracji: {e}")

    if USE_GUI:
        plt.ion()
        fig = plt.figure(figsize=(14, 6))
        fig.canvas.manager.set_window_title('BioAgent Neural Monitor')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.5, 1])

        # Wykres sensoryczny
        ax_sens = plt.subplot(gs[0])
        ax_sens.set_title("1. Sensory (ucho)", color='blue')
        ax_sens.set_ylim(0, 1)
        bar_sens = ax_sens.bar(range(agent.n_sensors), np.zeros(agent.n_sensors), color='blue')
        ax_sens.set_xlabel("Czestoliwosc (Low -> High)")

        # Wykres mozgu
        ax_brain = plt.subplot(gs[1])
        ax_brain.set_title("2. Asocjacja (Wewn. stan)", color="purple")
        # Obraz 8x8 (bo 64 neurony)
        brain_img = ax_brain.imshow(np.zeros((8, 8)), cmap='inferno', vmin=-1, vmax=1)
        plt.colorbar(brain_img, ax=ax_brain, label='Aktywacja')
        ax_brain.axis("off")

        # Wykres motoryczny
        ax_motor = plt.subplot(gs[2])
        ax_motor.set_title("3. Motoryka (Miesnie)", color='green')
        ax_motor.set_ylim(0, 1)
        bar_motor = ax_motor.bar(agent.motor_actions, np.zeros(agent.n_motor), color='green')

        # Text statusu
        status_text = fig.text(0.5, 0.02, "Slucham...", ha='center', fontsize=12)

    print("Start systemu... (Ctrl + C aby przerwac)")
    try:
        with sd.InputStream(samplerate=FS, channels=1, callback=audio_callback, blocksize=hop_size, dtype='float32'):
            if not agent.skip_baseline_calibration:
                agent.collect_baseline(audio_q, duration_sec=30)
            
            while True:
                if not audio_q.empty():
                    q_size = audio_q.qsize()
                    if q_size > 10:
                        print(f"OSTRZEŻENIE: Kolejka rośnie! Rozmiar: {q_size}")


                    chunk = audio_q.get().flatten()

                    start_time = time.time()
                    action_idx = agent.process_frame(chunk)
                    if SHOW_PROCESS_TIME:
                        print(f"Czas procesowania: {(time.time() - start_time)*1000:.2f} ms")

                    # WIZUALIZACJA
                    if USE_GUI:
                        # Update sensorow
                        for rect, h in zip(bar_sens, agent.last_sensors):
                            rect.set_height(h)

                        # Update stanu mozgu
                        brain_grid = agent.state.reshape(8, 8)
                        brain_img.set_data(brain_grid)

                        # Update motoryki
                        for i, rect in enumerate(bar_motor):
                            # Podswietl wybrana akcja na czerwono
                            rect.set_height(agent.last_motors[i])
                            rect.set_color('red' if i == action_idx else 'green')

                        status_text.set_text(f"Akcja: {agent.motor_actions[action_idx]}")

                        # Renderowanie klatki
                        fig.canvas.flush_events()
                        plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nZatrzymano.")
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
