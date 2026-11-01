"""Obrót kamery między dwoma ujęciami z dopasowanych punktów obrazu (model czystego obrotu)."""

import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_points(a, b, max_points=400, fb_threshold=1.0):
    """Punkty (N, 2) w a i odpowiadające im w b. Piramidalny Lucas-Kanade ze sprawdzeniem w przód i w tył."""
    pa = cv2.goodFeaturesToTrack(a, maxCorners=max_points, qualityLevel=0.01, minDistance=7)
    if pa is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    pb, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pa, None, **LK_PARAMS)
    pa_back, status_back, _ = cv2.calcOpticalFlowPyrLK(b, a, pb, None, **LK_PARAMS)
    fb_error = np.linalg.norm(pa - pa_back, axis=2).ravel()
    ok = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb_error < fb_threshold)
    return pa.reshape(-1, 2)[ok], pb.reshape(-1, 2)[ok]


def pixels_to_rays(points, camera):
    """Piksele (N, 2) -> jednostkowe promienie (N, 3) w układzie kamery, z korekcją dystorsji."""
    xy = cv2.undistortPoints(points.reshape(-1, 1, 2).astype(np.float64), camera.K, camera.dist).reshape(-1, 2)
    rays = np.column_stack([xy, np.ones(len(xy))])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def fit_rotation(a, b):
    """R minimalizujące sum |b - R a|^2 (algorytm Kabscha). a, b: (N, 3) wektory jednostkowe."""
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def ransac_rotation(a, b, threshold, iterations=200, rng=None):
    """Odporne dopasowanie obrotu b = R a: (R, maska inlierów). threshold: maks. błąd kątowy [rad]."""
    rng = rng or np.random.default_rng(0)
    best = np.zeros(len(a), dtype=bool)
    for _ in range(iterations):
        sample = rng.choice(len(a), 2, replace=False)
        R = fit_rotation(a[sample], b[sample])
        inliers = np.linalg.norm(b - a @ R.T, axis=1) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        raise ValueError("Nie znaleziono spójnego obrotu")

    R = fit_rotation(a[best], b[best])
    inliers = np.linalg.norm(b - a @ R.T, axis=1) < threshold
    return fit_rotation(a[inliers], b[inliers]), inliers


def estimate_rotation(a, b, camera, threshold_px=1.0):
    """Obrót kamery w ujęciu b względem a i liczba punktów zgodnych z tym obrotem.

    Zwraca macierz R_cam (osie kamery b wyrażone w układzie kamery a). Zakłada czysty obrót,
    czyli scenę daleką w porównaniu z przesunięciem kamery.
    """
    pa, pb = track_points(a, b)
    if len(pa) < 10:
        raise ValueError(f"Za mało dopasowanych punktów: {len(pa)}")
    rays_a, rays_b = pixels_to_rays(pa, camera), pixels_to_rays(pb, camera)
    R_rays, inliers = ransac_rotation(rays_a, rays_b, threshold_px / camera.fx)
    # Kierunek punktu sceny w układzie b: rays_b = R_cam^T rays_a
    return R_rays.T, int(inliers.sum())


def image_shift(a, b, window=None):
    """Przesunięcie (dx, dy) obrazu b względem a [px], korelacja fazowa.

    Okno Hanninga tłumi brzegi obrazu, które bez niego przekłamują wynik o ułamki piksela.
    Przy wielu wywołaniach warto przekazać gotowe window.
    """
    if window is None:
        window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    return dx, dy
