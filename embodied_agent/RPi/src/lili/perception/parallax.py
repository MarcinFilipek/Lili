"""Przepływ resztkowy po derotacji + przesunięcie kamery T -> odwrotna głębia 1/Z.

Po derotacji kamera B' jest przesunięta o T względem A bez obrotu. Punkt o współrzędnych
znormalizowanych (x, y) i odwrotnej głębi rho = 1/Z przesuwa się wtedy o

    u = rho * d / (1 - rho * Tz),   d = (fx (x Tz - Tx), fy (y Tz - Ty))  [px]

Kierunek d zależy tylko od T (linia od epipolu), więc z rzutu przepływu na d, s = u.d / |d|^2,
wychodzi dokładnie rho = s / (1 + s Tz). Składowa prostopadła do d to błąd: ani obrót, ani T
jej nie tłumaczą. Jeśli T jest znane z dokładnością do skali, rho jest w tej samej skali.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Parallax:
    rho: np.ndarray  # (H, W) odwrotna głębia [1/m] (lub 1/jednostkę T), nan tam, gdzie nie da się jej wyznaczyć
    gain: np.ndarray  # (H, W) |d|: przepływ w px na jednostkę rho; 0 w epipolu
    residual: np.ndarray  # (H, W) składowa przepływu prostopadła do d [px]


def epipolar_directions(K, T, shape):
    """Kierunki d (H, W, 2) [px na jednostkę rho] dla obrazu otworkowego o macierzy K."""
    h, w = shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xs, ys = np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
    Tx, Ty, Tz = T
    return np.stack([fx * (xs * Tz - Tx), fy * (ys * Tz - Ty)], axis=-1)


def parallax(flow, K, T, min_gain=1e-3):
    """Odwrotna głębia z przepływu między obrazami po derotacji (wspólne K, bez dystorsji)."""
    T = np.asarray(T, float)
    d = epipolar_directions(K, T, flow.shape[:2])
    gain = np.linalg.norm(d, axis=-1)
    usable = gain > min_gain * max(gain.max(), 1e-12)
    safe = np.where(usable, gain, 1.0)
    s = np.sum(flow * d, axis=-1) / safe**2
    residual = np.abs(flow[..., 0] * d[..., 1] - flow[..., 1] * d[..., 0]) / safe
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = s / (1 + s * T[2])
    rho[~usable] = np.nan
    return Parallax(rho, np.where(usable, gain, 0.0), residual)


def rho_sigma(par, flow_noise=0.1, fb_error=None):
    """Przybliżona niepewność rho [1/m]: niepewność przepływu [px] podzielona przez gain.

    Do szumu przepływu (flow_noise) dochodzą błąd prostopadły do kierunku paralaksy
    i, jeśli podany, błąd spójności w przód i w tył.
    """
    flow_sigma = np.hypot(flow_noise, par.residual)
    if fb_error is not None:
        flow_sigma = np.hypot(flow_sigma, fb_error)
    with np.errstate(divide="ignore"):
        return flow_sigma / par.gain
