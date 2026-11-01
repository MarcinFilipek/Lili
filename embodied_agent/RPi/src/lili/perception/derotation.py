"""Kompensacja obrotu: obraz z kamery -> widok otworkowy (bez dystorsji) w orientacji odniesienia.

Dla pary A -> B z ruchem kamery Pose(R, T) (R = osie kamery B w układzie A): A dostaje samą korekcję
dystorsji, B korekcję i obrót R. Po tym kroku B' wygląda jak widok z kamery przesuniętej o T bez
obrotu, a oba obrazy mają tę samą macierz K bez dystorsji. Mapy zależą tylko od R, więc dla
powtarzalnego ruchu (peering) wystarczy policzyć je raz.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rectification:
    map_x: np.ndarray  # (H, W) float32: kolumna źródłowa dla każdego piksela wyjścia
    map_y: np.ndarray
    valid: np.ndarray  # (H, W) bool: piksel wyjścia ma źródło wewnątrz obrazu
    K: np.ndarray  # macierz kamery otworkowej wyjścia

    def apply(self, image):
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def max_distorted_radius(camera):
    """Największy promień znormalizowany (bez dystorsji), który jeszcze trafia w kadr.

    Model wielomianowy dystorsji poza kadrem zawraca, więc dalsze promienie rzutowałyby się
    z powrotem do środka obrazu.
    """
    corners = np.array([[-0.5, -0.5], [camera.width - 0.5, -0.5], [-0.5, camera.height - 0.5],
                        [camera.width - 0.5, camera.height - 0.5]])
    xy = cv2.undistortPoints(corners.reshape(-1, 1, 2), camera.K, camera.dist).reshape(-1, 2)
    return float(np.linalg.norm(xy, axis=1).max())


def rectification(camera, R=None, K=None):
    """Mapy dla obrazu kamery obróconej o R względem orientacji odniesienia (None = bez obrotu).

    K: macierz wyjścia, domyślnie ta sama co kamery. Przy dystorsji beczkowatej brzegi obrazu
    źródłowego (zwłaszcza rogi) wypadają wtedy poza kadr wyjścia.
    """
    K = camera.K if K is None else np.asarray(K, float)
    w, h = camera.width, camera.height
    xs, ys = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    # Piksel wyjścia -> promień w orientacji odniesienia -> ten sam promień w układzie kamery
    rays = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ np.linalg.inv(K).T
    if R is not None:
        rays = rays @ R  # R^T r dla każdego promienia
    xy = rays[..., :2] / rays[..., 2:]
    in_lens = (rays[..., 2] > 0) & (np.linalg.norm(xy, axis=-1) < max_distorted_radius(camera))

    points = np.concatenate([xy, np.ones_like(xy[..., :1])], axis=-1).reshape(-1, 3)
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.K, camera.dist)
    pixels = pixels.reshape(h, w, 2)
    valid = (
        in_lens
        & (pixels[..., 0] >= -0.5) & (pixels[..., 0] <= w - 0.5)
        & (pixels[..., 1] >= -0.5) & (pixels[..., 1] <= h - 0.5)
    )
    pixels[~valid] = -1  # poza kadrem: BORDER_REPLICATE, a valid mówi, czego nie ufać
    return Rectification(pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32), valid, K)


def derotate_pair(a, b, camera, R, K=None):
    """(A', B', valid): A bez dystorsji, B bez dystorsji i obrotu R, wspólna maska ważnych pikseli."""
    rect_a = rectification(camera, None, K)
    rect_b = rectification(camera, R, K)
    return rect_a.apply(a), rect_b.apply(b), rect_a.valid & rect_b.valid
