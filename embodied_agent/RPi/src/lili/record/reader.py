import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import schema as s


class Frame(NamedTuple):
    index: int
    t: int
    t_sensor: int
    image: np.ndarray


class Frames(NamedTuple):
    images: np.ndarray  # (N, H, W)
    t: np.ndarray
    t_sensor: np.ndarray
    index: np.ndarray


class Session:
    """Odczyt sesji zapisanej przez SessionWriter."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / s.META_FILE).read_text())
        self._chunks = sorted((self.path / s.FRAMES_DIR).glob("chunk_*.npz"))

    def events(self, kind=None):
        with open(self.path / s.STATE_FILE) as f:
            events = [json.loads(line) for line in f if line.strip()]
        return events if kind is None else [e for e in events if e["type"] == kind]

    def iter_frames(self):
        for chunk in self._chunks:
            with np.load(chunk) as data:
                for i, t, ts, image in zip(data[s.INDEX], data[s.T], data[s.T_SENSOR], data[s.FRAMES]):
                    yield Frame(int(i), int(t), int(ts), image)

    def frame_times(self):
        """(t, t_sensor) wszystkich klatek, bez wczytywania obrazów."""
        t, ts = [], []
        for chunk in self._chunks:
            with np.load(chunk) as data:
                t.append(data[s.T])
                ts.append(data[s.T_SENSOR])
        return _concat(t, np.int64), _concat(ts, np.int64)

    def load_frames(self):
        parts = {key: [] for key in (s.FRAMES, s.T, s.T_SENSOR, s.INDEX)}
        for chunk in self._chunks:
            with np.load(chunk) as data:
                for key, arrays in parts.items():
                    arrays.append(data[key])
        return Frames(
            images=_concat(parts[s.FRAMES], np.uint8),
            t=_concat(parts[s.T], np.int64),
            t_sensor=_concat(parts[s.T_SENSOR], np.int64),
            index=_concat(parts[s.INDEX], np.int64),
        )


def list_sessions(root):
    """Katalogi sesji w root, od najstarszej."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / s.META_FILE).is_file())


def _concat(arrays, dtype):
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)
