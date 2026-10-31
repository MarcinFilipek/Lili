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

Kamera jest odsunięta od osi pan o ok. **8 cm** (do dokładnego pomiaru, gdy będzie potrzebny).

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
2. **Peering na serwach**: obrót pan o θ przy kamerze odsuniętej o r = 8 cm od osi daje
   translację T = 2r·sin(θ/2), dla θ = 10° to ok. 1,4 cm.
3. **Gąsienice**: później, ta sama percepcja, inne źródło (R, T).

Szacunek dla 320×240 (f ≈ 310 px), pan 10°, r = 8 cm:

| Odległość | Przesunięcie z paralaksy |
|---|---|
| 30 cm | ok. 14 px |
| 1 m | ok. 4 px |
| 3 m | ok. 1,4 px |

Obrót sam w sobie przesuwa obraz o ok. 55 px. Błąd znajomości kąta o 0,5° daje ok. 2,7 px
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
   tylko mały przepływ translacyjny, co ułatwia pracę algorytmowi przepływu.
4. **flow:** przepływ optyczny A → B' (oraz B' → A do sprawdzenia spójności).
5. **parallax:** rzut przepływu resztkowego na kierunek wynikający z T → 1/Z na piksel
   + pewność (tekstura, spójność w przód i w tył).
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
  więc „stopnie” z protokołu nie są stopniami fizycznymi) i czas dojazdu, który wyznacza
  długość fazy `settle`.
- **Geometria:** odsunięcie kamery od osi pan (ok. 8 cm) i tilt (do zmierzenia).

## Kamienie milowe

| | Zakres | Gotowe, gdy |
|---|---|---|
| M0 | Struktura pakietu, `stm_link` i `camera` przeniesione do `hw/` | testy przechodzą, narzędzia działają na RPi |
| M1 | `record/` + `tools/record_session.py` z prostym programem ruch-stop-patrz | sesja zapisuje się na RPi i odczytuje na Macu |
| M2 | Kalibracja kamery i serw | `calib/` z K i modelem serw |
| M3 | `synthetic.py`, `derotation`, `parallax` | na scenach syntetycznych czysty obrót daje resztkę ≈ 0, a odzyskane 1/Z zgadza się z prawdziwym |
| M4 | Nagrania z ręczną translacją, `occlusion`, `view_session` | widoczne krawędzie obiektów na stole |
| M5 | Peering na serwach | kryterium sukcesu etapu 1 |
| później | Gąsienice: MDD3A na STM32, `hw/drive.py`, kopia eferentna napędu | — |

## Otwarte pytania

- Dokładne odsunięcie kamery od osi pan i tilt (zmierzyć przed M5).
- 160×120 czy 320×240 dla percepcji: decyzja po M4 na podstawie jakości i czasu na RPi.
- Kiedy kopia eferentna wystarczy, a kiedy potrzebne doprecyzowanie R z obrazu.
