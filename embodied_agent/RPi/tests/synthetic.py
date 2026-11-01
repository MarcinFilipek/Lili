"""Sceny syntetyczne: teksturowane prostokąty w 3D, render z dowolnej pozy i prawdziwy przepływ.

Układ świata jak w body/geometry.py (x w prawo, y w dół, z do przodu, metry). Poza kamery to
geometry.Pose: R = osie kamery w świecie, c = środek optyczny.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from lili.body.camera_model import CameraModel
from lili.body.geometry import Pose

IDENTITY = Pose(np.eye(3), np.zeros(3))
# Kamery testowe: parametry zbliżone do calib/camera_320x240.json
PINHOLE = CameraModel(320, 240, np.array([[376.0, 0, 160.0], [0, 376.0, 120.0], [0, 0, 1]]), np.zeros(5))
DISTORTED = CameraModel(320, 240, PINHOLE.K, np.array([-0.43, 0.2, -0.003, 0.0003, 0.0]))
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    return cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(degrees))[0]


def noise_texture(width, height, seed):
    """Losowa tekstura (height, width) float32 0..255: plamy dwóch skal, dużo narożników."""
    rng = np.random.default_rng(seed)
    texture = rng.random((height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 3) + 0.7 * cv2.GaussianBlur(texture, (0, 0), 1.0)
    return cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)


@dataclass(frozen=True)
class Rect:
    center: np.ndarray  # środek prostokąta [m]
    right: np.ndarray  # krawędź wzdłuż wierszy tekstury (cała długość) [m]
    down: np.ndarray  # krawędź wzdłuż kolumn tekstury [m]
    texture: np.ndarray  # (H, W) float32, pokrywa cały prostokąt

    @classmethod
    def facing(cls, center, width, height, texels_per_m, seed):
        """Prostokąt prostopadły do osi z, tekstura o zadanej gęstości."""
        texture = noise_texture(round(width * texels_per_m), round(height * texels_per_m), seed)
        return cls(np.asarray(center, float), np.array([width, 0.0, 0.0]), np.array([0.0, height, 0.0]), texture)


@dataclass(frozen=True)
class Scene:
    rects: tuple  # id obiektu = indeks w krotce


def table_scene(f=376.0):
    """Trzy obiekty 0.3-0.8 m przed kamerą i tło 3 m. Teksel ok. 1 px w odległości obiektu."""
    def facing(center, width, height, seed):
        return Rect.facing(center, width, height, f / center[2], seed)

    return Scene((
        facing([0.0, 0.0, 3.0], 8.0, 6.0, 0),
        facing([-0.10, 0.02, 0.35], 0.08, 0.12, 1),
        facing([0.06, -0.03, 0.55], 0.14, 0.10, 2),
        facing([0.18, 0.06, 0.80], 0.20, 0.25, 3),
    ))


def pixel_grid(width, height, supersample=1):
    """Środki próbek (H*s, W*s, 2) w pikselach; środek piksela (i, j) ma współrzędne (j, i)."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    xs = (np.arange(width)[:, None] + offsets).ravel()
    ys = (np.arange(height)[:, None] + offsets).ravel()
    return np.stack(np.meshgrid(xs, ys), axis=-1)


def pixels_to_normalized(pixels, camera):
    """Piksele (..., 2) -> współrzędne znormalizowane (..., 2), z korekcją dystorsji."""
    shape = pixels.shape
    points = pixels.reshape(-1, 1, 2).astype(np.float64)
    if not np.any(camera.dist):
        xy = cv2.undistortPoints(points, camera.K, None)
    elif hasattr(cv2, "undistortPointsIter"):  # OpenCV 4
        xy = cv2.undistortPointsIter(points, camera.K, camera.dist, None, None, UNDISTORT_CRITERIA)
    else:  # OpenCV 5: criteria w undistortPoints
        xy = cv2.undistortPoints(points, camera.K, camera.dist, None, None, None, UNDISTORT_CRITERIA)
    return xy.reshape(shape)


def project(points_cam, camera):
    """Punkty (..., 3) w układzie kamery -> piksele (..., 2), z dystorsją."""
    shape = points_cam.shape
    pixels, _ = cv2.projectPoints(points_cam.reshape(-1, 3), np.zeros(3), np.zeros(3), camera.K, camera.dist)
    return pixels.reshape(shape[:-1] + (2,))


@dataclass
class Hits:
    depth: np.ndarray  # (H, W) Z w układzie kamery [m], inf gdy pusto
    ids: np.ndarray  # (H, W) indeks prostokąta, -1 gdy pusto
    points: np.ndarray  # (H, W, 3) punkty w układzie świata
    uv: np.ndarray  # (H, W, 2) współrzędne w teksturze trafionego prostokąta [0..1]


def cast(scene, camera, pose, pixels):
    """Promienie przez piksele (H, W, 2) kamery w pozie pose: najbliższe trafienie."""
    xy = pixels_to_normalized(pixels, camera)
    rays = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1) @ pose.R.T  # z_kam = 1
    depth = np.full(xy.shape[:-1], np.inf)
    ids = np.full(xy.shape[:-1], -1)
    uv = np.zeros(xy.shape)
    for i, rect in enumerate(scene.rects):
        normal = np.cross(rect.right, rect.down)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Parametr promienia = Z w układzie kamery, bo składowa z promienia w kamerze to 1
            t = ((rect.center - pose.c) @ normal) / (rays @ normal)
        hit = pose.c + t[..., None] * rays - rect.center
        a = hit @ rect.right / (rect.right @ rect.right) + 0.5
        b = hit @ rect.down / (rect.down @ rect.down) + 0.5
        closer = (t > 0) & (t < depth) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        depth[closer] = t[closer]
        ids[closer] = i
        uv[closer] = np.stack([a, b], axis=-1)[closer]
    points = pose.c + np.where(np.isfinite(depth), depth, 0)[..., None] * rays
    return Hits(depth, ids, points, uv)


def shade(scene, hits, background=0.0):
    """Jasność próbek z tekstur trafionych prostokątów (interpolacja dwuliniowa)."""
    values = np.full(hits.depth.shape, background, np.float32)
    for i, rect in enumerate(scene.rects):
        mask = hits.ids == i
        if not mask.any():
            continue
        h, w = rect.texture.shape
        # Krawędzie tekstury w 0 i 1, środki tekseli w (k + 0.5) / w
        tx = np.clip(hits.uv[..., 0][mask] * w - 0.5, 0, w - 1)
        ty = np.clip(hits.uv[..., 1][mask] * h - 0.5, 0, h - 1)
        values[mask] = bilinear(rect.texture, tx, ty)
    return values


def bilinear(image, x, y):
    """Wartości image w punktach (x, y) wewnątrz obrazu. cv2.remap nie przyjmuje map dłuższych niż 32767."""
    x0 = np.minimum(x.astype(int), image.shape[1] - 2)
    y0 = np.minimum(y.astype(int), image.shape[0] - 2)
    fx, fy = x - x0, y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def render(scene, camera, pose=IDENTITY, supersample=3):
    """Obraz uint8 (H, W) z kamery w pozie pose. Nadpróbkowanie tłumi aliasing na krawędziach."""
    s = supersample
    hits = cast(scene, camera, pose, pixel_grid(camera.width, camera.height, s))
    values = shade(scene, hits)
    values = values.reshape(camera.height, s, camera.width, s).mean(axis=(1, 3))
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def ground_truth(scene, camera, pose=IDENTITY):
    """Trafienia w środkach pikseli: mapa Z, id obiektów, punkty 3D."""
    return cast(scene, camera, pose, pixel_grid(camera.width, camera.height))


def true_flow(scene, camera, pose_a, pose_b):
    """Prawdziwy przepływ A -> B (H, W, 2) [px] i maska pikseli A widocznych w B."""
    hits_a = ground_truth(scene, camera, pose_a)
    in_b = pose_b.to_camera(hits_a.points.reshape(-1, 3)).reshape(hits_a.points.shape)
    pixels_b = project(in_b, camera)
    flow = pixels_b - pixel_grid(camera.width, camera.height)

    # Widoczny, jeśli promień z B przez rzut punktu trafia w ten sam punkt (nic go nie zasłania)
    hits_b = cast(scene, camera, pose_b, pixels_b)
    inside = (
        (pixels_b[..., 0] > -0.5) & (pixels_b[..., 0] < camera.width - 0.5)
        & (pixels_b[..., 1] > -0.5) & (pixels_b[..., 1] < camera.height - 0.5)
    )
    visible = (
        np.isfinite(hits_a.depth) & (in_b[..., 2] > 0) & inside
        & (np.abs(hits_b.depth - in_b[..., 2]) < 1e-6 * in_b[..., 2])
    )
    return flow.astype(np.float32), visible


def boundaries(ids):
    """Piksele (H, W) bool, których sąsiad z 4-sąsiedztwa należy do innego obiektu (obie strony granicy)."""
    edge = np.zeros(ids.shape, bool)
    horizontal = ids[:, 1:] != ids[:, :-1]
    vertical = ids[1:] != ids[:-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    return edge


def edge_scores(detected, truth, region, tolerance=2):
    """(dokładność, kompletność) krawędzi w obszarze region, z tolerancją położenia w pikselach."""
    to_truth = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 3)
    to_detected = cv2.distanceTransform((~detected).astype(np.uint8), cv2.DIST_L2, 3)
    precision = np.mean(to_truth[detected & region] <= tolerance)
    recall = np.mean(to_detected[truth & region] <= tolerance)
    return precision, recall
