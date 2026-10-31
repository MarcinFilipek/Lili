import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CameraModel:
    """Model kamery otworkowej z dystorsją (konwencja OpenCV): K 3x3, dist (k1, k2, p1, p2, k3)."""

    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    rms: float = float("nan")  # błąd reprojekcji kalibracji [px]
    info: dict = field(default_factory=dict)

    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @property
    def cx(self):
        return float(self.K[0, 2])

    @property
    def cy(self):
        return float(self.K[1, 2])

    def fov_deg(self):
        """(poziomy, pionowy) kąt widzenia w stopniach."""
        return (
            math.degrees(2 * math.atan(self.width / (2 * self.fx))),
            math.degrees(2 * math.atan(self.height / (2 * self.fy))),
        )

    def scaled(self, width, height):
        """Ten sam obiektyw w innej rozdzielczości (np. 160x120 z kalibracji w 320x240)."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        # Środek piksela w (0.5, 0.5): skalowanie dotyczy krawędzi pikseli, nie ich indeksów
        K[0, 2] = (self.cx + 0.5) * sx - 0.5
        K[1, 2] = (self.cy + 0.5) * sy - 0.5
        return CameraModel(width, height, K, self.dist.copy(), self.rms, dict(self.info))

    def save(self, path):
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "rms": self.rms,
            "info": self.info,
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(
            width=data["width"],
            height=data["height"],
            K=np.array(data["K"], dtype=np.float64),
            dist=np.array(data["dist"], dtype=np.float64),
            rms=data["rms"],
            info=data.get("info", {}),
        )
