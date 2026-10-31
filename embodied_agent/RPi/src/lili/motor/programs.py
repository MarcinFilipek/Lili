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
    for angle in (center + amplitude, center - amplitude, tilt):
        if abs(angle) > LIMIT:
            raise ValueError(f"Kąt {angle} poza zakresem [-{LIMIT}, {LIMIT}]")

    offsets = [0.0] + [o for _ in range(cycles) for o in (amplitude, 0.0, -amplitude, 0.0)]
    return [Step(center + o, tilt, settle_s, look_s) for o in offsets]
