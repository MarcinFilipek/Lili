"""Kalibracja kamery z ujęć szachownicy (OpenCV)."""

import cv2
import numpy as np

from .camera_model import CameraModel

PATTERN = (9, 6)  # narożniki wewnętrzne, jak w calib/checkerboard_9x6_25mm.pdf


def find_corners(image, pattern=PATTERN, refine=True):
    """Narożniki szachownicy (N, 2) w kolejności OpenCV albo None, jeśli nie znaleziono."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(image, pattern, flags=flags)
    if not found:
        return None
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners = cv2.cornerSubPix(image, corners, (3, 3), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def board_points(pattern, square):
    """Narożniki w układzie tablicy (z = 0), w tej samej kolejności co find_corners."""
    cols, rows = pattern
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return np.hstack([grid, np.zeros((len(grid), 1))]).astype(np.float32)


def view_signature(corners, image_size, pattern=PATTERN):
    """Krótki opis ujęcia: położenie, wielkość i pochylenie tablicy w kadrze.

    Służy do odrzucania ujęć zbyt podobnych do już zebranych. Pochylenie ma wagę 3,
    bo nawet 20° daje mały stosunek boków, a pochylone ujęcia najbardziej pomagają kalibracji.
    """
    cols, rows = pattern
    w, h = image_size
    grid = corners.reshape(rows, cols, 2)
    top = np.linalg.norm(grid[0, -1] - grid[0, 0])
    bottom = np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    left = np.linalg.norm(grid[-1, 0] - grid[0, 0])
    right = np.linalg.norm(grid[-1, -1] - grid[0, -1])
    cx, cy = corners.mean(axis=0)
    tilt = 3 * np.log([top / bottom, left / right])
    return np.array([cx / w, cy / h, (top + bottom) / (2 * w), *tilt])


def calibrate(corner_sets, image_size, pattern=PATTERN, square=25.0):
    """Zwraca (CameraModel, błąd reprojekcji każdego ujęcia [px]). square w mm."""
    if len(corner_sets) < 3:
        raise ValueError(f"Za mało ujęć: {len(corner_sets)}, potrzeba co najmniej 3")

    obj = [board_points(pattern, square)] * len(corner_sets)
    img = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corner_sets]
    # k3 wyłączone: przy kilkudziesięciu ujęciach w niskiej rozdzielczości łatwo się przeucza
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=cv2.CALIB_FIX_K3)

    per_view = []
    for o, i, rvec, tvec in zip(obj, img, rvecs, tvecs):
        projected, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((projected - i) ** 2, axis=2)))))

    model = CameraModel(image_size[0], image_size[1], K, dist.ravel(), float(rms))
    return model, per_view
