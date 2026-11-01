"""Model serwa: komenda (jednostki protokołu, czyli "stopnie" po stronie Pythona) -> rzeczywisty obrót kamery."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ServoModel:
    name: str  # "pan" / "tilt"
    channel: int
    deg_per_unit: float  # rzeczywiste stopnie obrotu na jednostkę komendy
    axis: list  # oś obrotu kamery (wektor jednostkowy w układzie kamery) przy rosnącej komendzie
    table: list  # [[komenda, kąt względem komendy 0 [deg]], ...], rosnąco po komendzie
    latency_ms: float  # od ACK z STM do początku ruchu obrazu (mediana)
    settle_ms: float  # od ACK z STM do zatrzymania obrazu (95. percentyl)
    info: dict = field(default_factory=dict)

    def angle(self, command):
        """Rzeczywisty kąt [deg] względem komendy 0. Poza tabelą liniowo z deg_per_unit."""
        commands, angles = np.array(self.table, dtype=float).T
        if command < commands[0]:
            return float(angles[0] + self.deg_per_unit * (command - commands[0]))
        if command > commands[-1]:
            return float(angles[-1] + self.deg_per_unit * (command - commands[-1]))
        return float(np.interp(command, commands, angles))


def load_servo_models(path):
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    return {name: ServoModel(**d) for name, d in data.items()}


def save_servo_model(path, model):
    """Zapisuje model pod jego nazwą, zostawiając w pliku modele pozostałych serw."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[model.name] = asdict(model)
    path.write_text(json.dumps(data, indent=2))


@dataclass
class AxisFit:
    axis: np.ndarray
    deg_per_unit: float
    cumulative: np.ndarray  # kąt każdego kroku względem kroku 0 [deg]
    residual_rms: float  # odchyłka od prostej kąt(komenda) [deg]
    off_axis_rms: float  # obrót poza wspólną osią na krok [deg], miara chybotania
    hysteresis: float  # średnia |różnica| kąta przy tej samej komendzie: dojazd od dołu minus od góry [deg]
    table: list


def fit_axis(commands, rvecs):
    """Dopasowanie osi i skali serwa.

    commands: (n,) komendy kolejnych kroków. rvecs: (n-1, 3) obroty kamery między kolejnymi
    krokami jako wektory Rodriguesa [rad].
    """
    commands = np.asarray(commands, dtype=float)
    rvecs = np.asarray(rvecs, dtype=float)
    deltas = np.diff(commands)
    moving = deltas != 0

    axis = (rvecs[moving] * np.sign(deltas[moving])[:, None]).sum(axis=0)
    axis /= np.linalg.norm(axis)
    along = rvecs @ axis
    off_axis = np.linalg.norm(rvecs - np.outer(along, axis), axis=1)

    cumulative = np.degrees(np.concatenate([[0.0], np.cumsum(along)]))
    slope, intercept = np.polyfit(commands, cumulative, 1)
    residual = cumulative - (slope * commands + intercept)

    direction = np.concatenate([[0.0], np.sign(deltas)])
    differences, table = [], []
    for command in np.unique(commands):
        at = commands == command
        up, down = at & (direction > 0), at & (direction < 0)
        if up.any() and down.any():
            differences.append(cumulative[up].mean() - cumulative[down].mean())
        table.append([float(command), float(cumulative[at].mean())])
    zero = np.interp(0.0, *np.array(table).T)

    return AxisFit(
        axis=axis,
        deg_per_unit=float(slope),
        cumulative=cumulative,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        off_axis_rms=float(np.degrees(np.sqrt(np.mean(off_axis[moving] ** 2)))),
        hysteresis=float(np.mean(np.abs(differences))) if differences else float("nan"),
        table=[[c, a - zero] for c, a in table],
    )


def motion_times(times, from_previous, to_final, t_ack, threshold=0.5):
    """(opóźnienie ruchu, czas dojazdu) względem ACK z STM [s], None gdy ruchu nie wykryto.

    from_previous: odległość [px] każdej klatki od nieruchomego obrazu przed ruchem,
    to_final: od nieruchomego obrazu po ruchu. Dojazd kończy się na pierwszej klatce,
    po której obraz już stale jest bliżej końcowego niż threshold.
    """
    after = times >= t_ack
    moved = after & (from_previous > threshold)
    if not moved.any():
        return None, None
    latency = times[moved][0] - t_ack

    unsettled = np.flatnonzero(after & (to_final > threshold))
    if len(unsettled) == 0:
        return latency, latency
    last = unsettled[-1]
    if last + 1 >= len(times):
        return latency, None
    return latency, times[last + 1] - t_ack
