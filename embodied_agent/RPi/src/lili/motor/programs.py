from dataclasses import dataclass

LIMIT = 90.0


@dataclass(frozen=True)
class Step:
    """Krok ruch-stop-patrz: jedź do (pan, tilt), odczekaj settle_s na dojazd serw, patrz przez look_s."""

    pan: float
    tilt: float
    settle_s: float
    look_s: float


def peering(amplitude=10.0, cycles=3, center=0.0, tilt=0.0, settle_s=0.5, look_s=0.5):
    """Kołysanie pan wokół center, jak peering szarańczy: center, +a, center, -a, center, ...

    Kamera jest odsunięta od osi pan, więc obrót daje też translację, czyli paralaksę.
    """
    _check_range(center + amplitude, center - amplitude, tilt)

    offsets = [0.0] + [o for _ in range(cycles) for o in (amplitude, 0.0, -amplitude, 0.0)]
    return [Step(center + o, tilt, settle_s, look_s) for o in offsets]


def sweep(axis, limit=30.0, step=5.0, fixed=0.0, settle_s=1.5, look_s=0.5):
    """Kalibracja serwa: oś axis ("pan"/"tilt") od -limit do +limit co step i z powrotem.

    Druga oś stoi w fixed. Przejazd w obie strony pokazuje histerezę (luz przekładni, martwą strefę).
    """
    if axis not in ("pan", "tilt"):
        raise ValueError(f"Nieznana oś: {axis}")
    _check_range(limit, -limit, fixed)

    n = int(2 * limit // step)
    up = [-limit + i * step for i in range(n + 1)]
    angles = up + up[-2::-1]
    if axis == "pan":
        return [Step(a, fixed, settle_s, look_s) for a in angles]
    return [Step(fixed, a, settle_s, look_s) for a in angles]


def _check_range(*angles):
    for angle in angles:
        if abs(angle) > LIMIT:
            raise ValueError(f"Kąt {angle} poza zakresem [-{LIMIT}, {LIMIT}]")
