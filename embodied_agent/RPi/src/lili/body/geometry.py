"""Kinematyka głowy: baza -> pan -> tilt -> kamera.

Układ świata = układ kamery w pozie odniesienia (pan 0, tilt 0): x w prawo, y w dół, z wzdłuż
osi optycznej, jednostki w metrach. Osie przegubów są podane w tej pozie (iloczyn eksponencjałów).
"""

from dataclasses import dataclass

import cv2
import numpy as np

FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Pose:
    R: np.ndarray  # osie kamery w układzie odniesienia (kolumny)
    c: np.ndarray  # środek optyczny kamery w układzie odniesienia [m]

    def relative_to(self, other):
        """Ta poza wyrażona w układzie kamery other: obrót i przesunięcie kamery od other do self."""
        return Pose(other.R.T @ self.R, other.R.T @ (self.c - other.c))

    def to_camera(self, points):
        """Punkty (N, 3) z układu odniesienia do układu tej kamery."""
        return (points - self.c) @ self.R


@dataclass(frozen=True)
class Joint:
    axis: np.ndarray  # kierunek osi obrotu (jednostkowy), dodatni kąt = obrót prawoskrętny
    point: np.ndarray  # dowolny punkt na osi

    def transform(self, angle_deg):
        """(R, t) obrotu wokół osi: x -> R x + t."""
        R = cv2.Rodrigues(self.axis * np.radians(angle_deg))[0]
        return R, self.point - R @ self.point


def joint_behind_camera(axis, lever):
    """Przegub, którego oś leży lever [m] za obiektywem, na wprost wzdłuż osi optycznej."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    forward = FORWARD - (FORWARD @ axis) * axis
    forward /= np.linalg.norm(forward)
    return Joint(axis, -lever * forward)


@dataclass(frozen=True)
class Head:
    pan: Joint
    tilt: Joint

    def pose(self, pan_deg, tilt_deg):
        """Poza kamery dla rzeczywistych kątów przegubów [deg] względem pozy odniesienia."""
        # Tilt siedzi na ogniwie pan: najpierw obrót tilt, potem całość obraca pan
        R_pan, t_pan = self.pan.transform(pan_deg)
        R_tilt, t_tilt = self.tilt.transform(tilt_deg)
        return Pose(R_pan @ R_tilt, R_pan @ t_tilt + t_pan)
