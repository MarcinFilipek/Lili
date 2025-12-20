import numpy as np
import sounddevice as sd
import queue
import sys


class BioAgent():
    def __init__(self, fs=16000):
        self.fs = fs
        self.buffer_k = 10
        self.n_bands = 32
        self.n_motor = 4
        self.theta = np.zeros(self.n_bands)
        self.eligibility_decay = 0.95
        self.time_buffer = np.zeros((self.buffer_k, self.n_bands))
        self.lr = 0.05

        # Struktura neuro
        self.input_dim = self.n_bands * self.buffer_k
        self.weights = np.random.normal(
            0, 0.01, size=(self.input_dim, self.n_motor))
        self.eligibility = np.zeros_like(self.weights)

        # Bufor emocji
        self.feedback_cursor = 0
        self.feedback_buffer_size = fs * 1
        self.feedback_audio = np.zeros(self.feedback_buffer_size)

    def _frame_to_bands(self, frame):
        # Uproszczone FFT -> Energy pasm
        windowed = frame * np.hamming(len(frame))
        spectrum = np.fft.rfft(windowed, n=512)
        power = np.abs(spectrum)**2
        bins_per_band = len(power) // self.n_bands
        bands = np.array([np.sum(power[i*bins_per_band:(i+1)*bins_per_band])
                         for i in range(self.n_bands)])
        return np.log1p(bands)

    def process_frame(self, raw_frame_audio):
        '''
        1. Analiza spektralna
        - Sygnal jest poddawany transformacie Fouriera (FFT)
        - Zamiast setek prazkow czestotliwosci, grupujemy je w N_BANDS (np. 32 pasma)
        - Wynik: Wektor 32 liczb, gdzie kazda liczba mowi "jak glosno jest w danym pasmie" 
        (basy, srednie, soprany)
        '''
        bands = self._frame_to_bands(raw_frame_audio)

        '''
        2. Aktywacja nuronow i progowanie
        - Od energi kazdego pasma odejmujemy prog theta (bands - theta)
        - Stosujemy funkcje ReLU (np.maximum(0, ...)). Jesli energia jest ponizej progu (szum),
        neuron wysyla 0. Jesli powyzej, wysyla sygnal dodatni.
        - Cel: odszumienie, agent ignoruje ciche dzwieki tla.
        '''
        activations = np.maximum(0, bands - self.theta)
        if np.max(activations) > 0:
            activations /= (np.max(activations) * 1e-8)

        '''
        3. Aktualizacja bufora. Pojedyncza ramka to za malo do rozpoznania slowa. Potrzeba
        kontekstu, co bylo przed chwila.
        - Macierz time_buffer to "pamiec" ostatnich 10 ramek
        - np.roll(..., -1) przesowamy cala pamiec w "gore" (zapominamy najstarsza ramke)
        - Wstawiamy biezace aktywacje (activations) na sam koniec
        '''
        self.time_buffer = np.roll(self.time_buffer, -1, axis=0)
        self.time_buffer[-1, :] = activations

        '''
        4. Splaszenie widoku
        - Agent nie widzi macierzy 2D. Zmieniamy buffor (np. 10 ramek x 32 pasma) 
        w jeden dlugi wektor (320 liczb)
        - Wazne: Dzieki temu siec "widzi" roznice miedzy sekwencja "A potem B",
        a "B potem A". Kazdy monet w czasie ma swoje wlasne polaczenie synaptyczne.
        '''
        input_vector = self.time_buffer.flatten()

        '''
        5. Obliczenie aktywacji silnikow
        - Mnozymy wektor wejsciowy przez macierz wag weights.
        - Kazdy silnik (FORWARD, LEFT, RIGHT, STOP) sumuje sygnaly, ktore do niego docieraja
        '''
        motor_activations = np.dot(input_vector, self.weights)

        '''
        6. Wybor akcji
        Stosujemy strategie Epsilon-Greedy
        - Rzucamy kostka. Z malym prawdopodobienstwem 10%, wybieramy losowa akcje.
        - Zazwyczaj wybieramy akcje o najsilnieszym sygnale (np.argmax)
        '''
        if np.random.rand() < 0.1:
            action_idx = np.random.randint(0, self.n_motor)
        else:
            action_idx = np.argmax(motor_activations)

        '''
        7. Przygotowanie do nauki (Eligibility Traces)
        Kluczowy moment dla uczenia ze wzmocnieniem. Agent jeszcze nie wie, czy zrobil
        dobrze, ale zapamietuje, co zrobil.
        Aktualizacja sladu (eligibility)
        - Tworzymy "zdjecie" sytuacji: ktore wyjscia byly aktywne i ktora akcja zostala 
        wybrana
        - Wzor: eligibility = decay * eligibility + (Wejscie * Wyjscie)
        - Stopniowo zapominamy stare decyzje (decay), a mocno zaznaczamy te wlasnie podjeta
        - Sens: Jesli za 0.5 sekundy przyjdzie nagroda, agent wzmocni dokladnie te wagi, 
        ktore sa zapisane w eligibility
        '''
        output_spike = np.zeros(self.n_motor)
        output_spike[action_idx] = 1.0
        self.eligibility = self.eligibility_decay * \
            self.eligibility + np.outer(input_vector, output_spike)

        '''
        8. Homeostaza, adaptacja, aktualizacja progow Theta
        - Jesli neurony w danym pasmie byly aktywne, leciutko podnosimy ich prog theta
        - Jesli milczy, leciutko obnizamy prog
        - Cel: Zapobiega to sytuacji, w ktorej agent "slepnie" w halasie lub staje sie 
        nadwrazliwy na cisze
        '''
        self.theta += 0.001 * (activations - 0.1)

        '''
        9. Analiza emocji
        - Buforowanie audio, surowe audio jest doklejane do duzego bufora (np. na 1 sekunde)
        Warunkowa naliza:
        NIE: 
        - nic nie rob, czekaj na wiecej danych
        TAK: 
        - Uruchom analyze_emotion_feedback(), policz pitch, jasnosc dzwieku
        - Jesli wykryto POCHWALE, wywolaj apply_reward(+1). Wagi zostana zaaktualizowane
        w oparciu o slad
        - Jesli wykryto NAGANE, wywolaj apply_award(-1)
        - Wyczysc bufor
        '''
        len_chunk = len(raw_frame_audio)
        if self.feedback_cursor + len_chunk <= self.feedback_buffer_size:
            self.feedback_audio[self.feedback_cursor: self.feedback_cursor +
                                len_chunk] = raw_frame_audio
            self.feedback_cursor += len_chunk
        else:
            reward = self.analyze_emotional_feedback()

            # Jesli mamy ocene, aplikujemy nagrode
            if reward != 0:
                self.apply_reward(reward)

            # Reset bufora (przesowamy np. o 50% lub zerujemy)
            self.feedback_audio = np.zeros(self.feedback_buffer_size)
            self.feedback_cursor = 0

    def apply_reward(self, reward):
        '''
        Modulacja wag o oparciu o zebrane eligibility traces.
        '''
        print(f"\n >>> OTRZYMANO NAGRODE: {reward} <<<")
        self.weights += self.lr * reward * self.eligibility
        # Reset sladu po silnym wzmocnieniu
        if abs(reward) > 0.5:
            self.eligibility *= 0.1

    def analyze_emotional_feedback(self):
        '''
        Analizuje 1 sekunde audio pod katem cech prozodyncznych
        '''
        return 0.0


def main():
    # Audio params
    FS = 16000
    HOP_MS = 10

    hop_size = int(FS * HOP_MS / 1000)

    audio_q = queue.Queue()
    agent = BioAgent(fs=FS)

    def audio_callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    print("Startuje nasluch... (Ctrl + C aby przerwac)")
    try:
        with sd.InputStream(samplerate=FS, channels=1, callback=audio_callback, blocksize=hop_size):
            while True:
                if not audio_q.empty():
                    new_samples = audio_q.get().flatten()

                    agent.process_frame(new_samples)

    except KeyboardInterrupt:
        print("\nZatrzymano.")
    except Exception as e:
        print("\nError: {e}")


if __name__ == "__main__":
    main()
