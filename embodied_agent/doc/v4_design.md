# Lili v4 — projekt

Bio-inspirowany agent embodied. Ten dokument opisuje etap 1 i decyzje, które go kształtują.
Poprzednie wersje: tagi `rpi-v3-final`, `rpi-v3-latest`.

## Sprzęt

| Element | Opis |
|---|---|
| Komputer | Raspberry Pi 5 Model B Rev 1.1 |
| Kamera | Pi Camera (ov5647), Picamera2, zamontowana do góry nogami (obraz obracany o 180° w ISP) |
| Głowa | serwa pan/tilt przez STM32 + PCA9685, protokół UART: [protocol.md](protocol.md) |
| Napęd (później) | gąsienice L/R, sterownik Cytron MDD3A na STM32 (nowe komendy do dopisania w protocol.md) |

Obiektyw jest odsunięty od osi pan o **9,9 cm**, a od osi tilt o **2,6 cm** (`calib/head.json`).

## Cel etapu 1: separacja obiektów z paralaksy ruchu

Przy czystym obrocie wokół środka optycznego każdy punkt obrazu przesuwa się tak samo,
niezależnie od odległości: obraz nie niesie informacji o głębi. Przy translacji kamery o T
punkt w odległości Z przesuwa się w obrazie o

    u ≈ f · T / Z

Po odjęciu składowej obrotowej z przepływu optycznego zostaje przepływ translacyjny, z którego
dostajemy przybliżone 1/Z. Skoki 1/Z to krawędzie zasłaniania, czyli prawdziwe kontury obiektów.

### Źródło translacji

1. **Ręczne przesuwanie** robota (5–10 cm, bez obrotu): mocny, czysty sygnał, T znane tylko
   w przybliżeniu → głębia względna. Pierwszy test całego łańcucha.
2. **Peering na serwach**: obrót pan o θ przy kamerze odsuniętej o r = 9,9 cm od osi daje
   translację T = 2r·sin(θ/2), dla θ = 10° to ok. 1,7 cm.
3. **Gąsienice**: później, ta sama percepcja, inne źródło (R, T).

Szacunek dla 320×240 (f ≈ 376 px z kalibracji), pan 10°, r = 9,9 cm:

| Odległość | Przesunięcie z paralaksy |
|---|---|
| 30 cm | ok. 22 px |
| 1 m | ok. 6,5 px |
| 3 m | ok. 2,2 px |

Obrót sam w sobie przesuwa obraz o ok. 66 px. Błąd znajomości kąta o 0,5° daje ok. 3,3 px
prawie stałego błędu w całym obrazie: przesuwa absolutne 1/Z, ale nie niszczy jego skoków,
więc krawędzie zasłaniania są odporne na niedokładną kalibrację.

### Kryterium sukcesu etapu 1

Na nagraniu 2–3 obiektów na stole w odległości 30–80 cm, z tłem dalej niż 1,5 m, wykryte
krawędzie zasłaniania pokrywają się z konturami obiektów. Najpierw ocena wzrokowa, potem
kilka ręcznie zaznaczonych masek i miara pokrycia (IoU).

## Zasady

- Python, numpy, OpenCV. Percepcja w 160×120 lub 320×240, musi działać na RPi.
- **Najpierw offline.** Surowe doświadczenie (klatki + stan serw/napędu + timestampy) idzie
  na dysk, algorytmy działają na nagraniach. Nagrania pobierane z RPi ręcznie (`rsync`).
- Pakiet `lili` (pyproject.toml), małe moduły, testy bez sprzętu.
- Biologia jako inspiracja tam, gdzie upraszcza, nie jako warstwa metafor w nazwach.

### Inspiracje biologiczne i ich odpowiedniki

| Biologia | W Lili |
|---|---|
| Kopia eferentna | `body/efference.py`: komenda motoryczna → przewidywana poza kamery (R, T) |
| Porównanie kopii eferentnej z reaferencją | kompensacja obrotu z kopii eferentnej, resztka = to, czego ruch własny nie tłumaczy |
| Peering szarańczy | `motor/programs.py`: kołysanie pan wokół punktu przy kamerze odsuniętej od osi |
| Ruch-stop-patrz (np. ptaki, sakkady) | klatki do percepcji tylko po ustaleniu się serwa: kopia eferentna zgadza się z rzeczywistością, brak rozmycia i efektu rolling shutter |

## Struktura

```
RPi/
├── src/lili/
│   ├── hw/                  # jedyne miejsce dotykające sprzętu
│   │   ├── stm_link/        # UART z STM32 (istniejący kod)
│   │   ├── camera.py        # Picamera2 (istniejący kod)
│   │   └── drive.py         # później: gąsienice
│   ├── body/                # model ciała, czysta matematyka
│   │   ├── geometry.py      # układy współrzędnych, łańcuch baza → pan → tilt → kamera
│   │   ├── camera_model.py  # K, projekcja, piksel ↔ promień
│   │   ├── servo_model.py   # bajt ↔ rzeczywisty kąt, czas dojazdu
│   │   └── efference.py     # komenda → przewidywana poza (R, T)
│   ├── record/              # surowe doświadczenie
│   │   ├── schema.py        # zawartość próbki i zdarzenia
│   │   ├── writer.py        # zapis sesji (RPi)
│   │   └── reader.py        # odczyt sesji (wszędzie)
│   ├── motor/
│   │   ├── programs.py      # peering, skan, ruch-stop-patrz
│   │   └── runner.py        # wykonanie programu na hw + zapis przez record/
│   ├── perception/          # tylko dane, zero sprzętu
│   │   ├── preprocess.py    # rozdzielczość, szarość
│   │   ├── rotation.py      # obrót kamery z obrazu (kalibracja serw, doprecyzowanie kopii eferentnej)
│   │   ├── flow.py          # przepływ optyczny, jeden interfejs
│   │   ├── derotation.py    # kompensacja obrotu
│   │   ├── parallax.py      # przepływ resztkowy + T → 1/Z i pewność
│   │   └── occlusion.py     # skoki 1/Z → krawędzie zasłaniania
│   └── pipeline.py          # perception/ dla jednej pary widoków
├── tools/                   # cienkie CLI: record_session, run_offline, view_session,
│                            # calibrate_camera, calibrate_servo, uart_stress, camera_preview
├── calib/                   # wyniki kalibracji (w repo)
├── data/                    # nagrania (gitignore)
└── tests/                   # w tym synthetic.py: znana mapa Z + poza → idealny przepływ
```

Zależności w jedną stronę: `perception` → `body`; `hw` używają tylko `motor/runner.py` i `tools/`.
Ruch zawsze opisany jako poza kamery (R, T), więc percepcja nie wie, czy ruszały się serwa,
gąsienice czy ręka.

## Pipeline dla pary widoków A → B

1. **preprocess:** szarość, docelowa rozdzielczość.
2. **efference:** z kątów serw (i później odometrii) poza B względem A: R, T.
3. **derotation:** przekształcenie B homografią obrotu H = K·Rᵀ·K⁻¹. Po tym kroku zostaje
   tylko mały przepływ translacyjny, co ułatwia pracę algorytmowi przepływu. Homografia działa
   tylko bez dystorsji, więc jeden `cv2.remap` robi naraz korekcję dystorsji i obrót (A dostaje
   samą korekcję), a dalsze kroki pracują w geometrii otworkowej o tej samej K.
4. **flow:** przepływ optyczny A → B' (oraz B' → A do sprawdzenia spójności).
5. **parallax:** rzut przepływu resztkowego na kierunek wynikający z T → 1/Z na piksel
   + niepewność 1/Z (składowa przepływu niezgodna z T, spójność w przód i w tył, odległość
   od epipolu). Wzór dokładny, bez przybliżenia u ≈ f·T/Z: u = ρ·d/(1 − ρ·Tz).
6. **occlusion:** krawędzie z gradientu 1/Z oraz obszary niespójnego przepływu
   (odsłonięcia i zasłonięcia) → kontury obiektów.

Jeśli kopia eferentna okaże się zbyt niedokładna, R można doprecyzować z obrazu
(homografia dopasowana do dalekiego tła), z kopią eferentną jako punktem startowym.

## Format nagrania

```
data/sessions/<YYYYMMDD-HHMMSS>_<etykieta>/
├── meta.json        # wersja kodu (git hash), rozdzielczość, ustawienia kamery,
│                    # program ruchu i parametry, pliki kalibracji
├── frames/
│   └── chunk_0000.npz   # frames (N,H,W) uint8, t (N,) ns, index (N,)
└── state.jsonl      # zdarzenia w kolejności czasu, jedna linia = jedno zdarzenie
```

Zdarzenia w `state.jsonl` (wszystkie z `t` w ns):
- `cmd`: wysłana komenda (pan/tilt w stopniach i bajtach, później napęd),
- `stm`: odpowiedź STM32 (potwierdzony kąt),
- `phase`: znacznik fazy ruch-stop-patrz (`move`, `settle`, `look`; `end` zamyka ostatni `look`).

Klatki: 320×240, szarość (kanał Y z YUV420), 30 fps, nagrywane ciągle przez całą sesję,
także w trakcie ruchu (przydatne do kalibracji czasu dojazdu serw).

Ustalenia:
- Klatki bez kompresji stratnej (kompresja wideo psuje przepływ optyczny).
- Na czas sesji stała ekspozycja i balans bieli: zmiany jasności wyglądają dla przepływu jak ruch.
- Jeden zegar dla wszystkiego (`time.monotonic_ns`). Timestamp sensora z metadanych Picamera2
  zapisywany obok, synchronizacja zegarów do sprawdzenia przy kalibracji.

## Kalibracja

- **Kamera:** K z szachownicy, osobno dla każdej używanej rozdzielczości i trybu sensora.
- **Serwa:** zależność zadany bajt → rzeczywisty kąt (tilt ma zakres impulsów 1000–1600 µs,
  ograniczony konstrukcją głowy, więc „stopnie” z protokołu nie są stopniami fizycznymi) i czas
  dojazdu, który wyznacza długość fazy `settle`. Mierzone kamerą: program `sweep` przejeżdża jedną
  oś tam i z powrotem, obrót między postojami liczony jest z obrazu daleko położonej sceny.
  Wynik: skala, oś obrotu w układzie kamery, histereza, opóźnienie i czas dojazdu (`calib/servo.json`).
  Źródła błędów po stronie firmware: niedokładny zegar PCA9685 (absorbuje go pomiar), krok PCA
  4,88 µs większy niż jednostka komendy tilt (3,3 µs), martwa strefa i luz przekładni.
- **Geometria:** odsunięcie obiektywu od osi pan (9,9 cm) i tilt (2,6 cm), pomiar linijką.
  Model zakłada obiektyw na wprost przed osiami. Osie z kalibracji serw: tilt dokładnie wzdłuż osi x
  kamery, pan odchylony o ok. 10° od osi y (przy tilt 0 kamera nie jest prostopadła do osi pan).
  Serwa są lokalnie nieliniowe (pan: 0→+10 to 8,7°, 0→−10 to 9,5°) i mają histerezę 0,3–0,4°
  (ok. 2–2,5 px), dlatego kopia eferentna korzysta z tabeli kątów, a obrót warto doprecyzować z obrazu.

## Kamienie milowe

| | Zakres | Gotowe, gdy |
|---|---|---|
| M0 | Struktura pakietu, `stm_link` i `camera` przeniesione do `hw/` | testy przechodzą, narzędzia działają na RPi |
| M1 | `record/` + `tools/record_session.py` z prostym programem ruch-stop-patrz | sesja zapisuje się na RPi i odczytuje na Macu |
| M2 | Kalibracja kamery i serw | `calib/` z K i modelem serw |
| M3 | `synthetic.py`, `derotation`, `flow`, `parallax` | na scenach syntetycznych czysty obrót daje resztkę ≈ 0, a odzyskane 1/Z zgadza się z prawdziwym |
| M4 | Nagrania z ręczną translacją, `occlusion`, `view_session` | widoczne krawędzie obiektów na stole |
| M5 | Peering na serwach | kryterium sukcesu etapu 1 |
| później | Gąsienice: MDD3A na STM32, `hw/drive.py`, kopia eferentna napędu | — |

## Wyniki M3 (sceny syntetyczne, `tests/synthetic.py`)

- Czysty obrót po derotacji: resztkowy przepływ DIS ok. 0,01 px (mediana), także z dystorsją z kalibracji.
- Peering pan ±10°, 15° z prawdziwej kalibracji (T ≈ 1,5 cm): mediana błędu 1/Z na obiektach 0,35–3 m
  poniżej 1,5%. Błąd kąta 0,5° przesuwa 1/Z o ok. 0,6 1/m, a skoki 1/Z między obiektami zostają (±6%).
- Obiekty, które po obrocie wychodzą poza wspólne pole widzenia, dostają dużą niepewność.
- DIS zawodzi przy przepływie porównywalnym z rozmiarem obiektu (ręczna translacja 5 cm, obiekt
  0,35 m: ok. 54 px przy obiekcie szerokim na 86 px): piramida gubi mały obiekt na grubych poziomach.
  Do M4: albo translacje ok. 2 cm, albo przeszukiwanie wzdłuż linii epipolarnych (T jest znane).

## Otwarte pytania

- Dokładne odsunięcie kamery od osi pan i tilt (zmierzyć przed M5).
- 160×120 czy 320×240 dla percepcji: decyzja po M4 na podstawie jakości i czasu na RPi.
- Kiedy kopia eferentna wystarczy, a kiedy potrzebne doprecyzowanie R z obrazu.
