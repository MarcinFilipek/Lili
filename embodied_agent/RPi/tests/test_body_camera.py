import cv2
import numpy as np
import pytest

from lili.body.camera_calibration import PATTERN, board_points, calibrate, find_corners, view_signature
from lili.body.camera_model import CameraModel

SIZE = (320, 240)
K_TRUE = np.array([[300.0, 0, 158.0], [0, 300.0, 121.0], [0, 0, 1]])
SQUARE = 25.0


def pose(rx, ry, rz, distance):
    """Tablica przed kamerą: obrót (stopnie) i odległość (mm), środek tablicy na osi optycznej."""
    rvec = np.radians([rx, ry, rz])
    R, _ = cv2.Rodrigues(rvec)
    cols, rows = PATTERN
    center = np.array([(cols - 1) * SQUARE / 2, (rows - 1) * SQUARE / 2, 0])
    tvec = np.array([0, 0, distance]) - R @ center
    return rvec, tvec


def render_board(rvec, tvec, px_per_mm=8, supersample=4):
    """Syntetyczne zdjęcie szachownicy (jak z wydruku: pola 10x7, biały margines) z kamerą K_TRUE.

    Render w supersample razy większej rozdzielczości i uśrednienie, jak przy zbieraniu światła przez piksel.
    """
    cols, rows = PATTERN
    margin = SQUARE
    w_mm, h_mm = (cols + 1) * SQUARE + 2 * margin, (rows + 1) * SQUARE + 2 * margin
    texture = np.full((int(h_mm * px_per_mm), int(w_mm * px_per_mm)), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x, y = int((margin + c * SQUARE) * px_per_mm), int((margin + r * SQUARE) * px_per_mm)
                s = int(SQUARE * px_per_mm)
                texture[y : y + s, x : x + s] = 0

    # Środek piksela tekstury -> mm w układzie tablicy (początek w pierwszym narożniku wewnętrznym)
    offset = 0.5 / px_per_mm - margin - SQUARE
    tex_to_board = np.array([[1 / px_per_mm, 0, offset], [0, 1 / px_per_mm, offset], [0, 0, 1]])
    R, _ = cv2.Rodrigues(rvec)
    # Kamera w rozdzielczości supersample: ten sam obiektyw, środki pikseli przesunięte o pół piksela
    s = supersample
    K_big = K_TRUE.copy()
    K_big[:2] *= s
    K_big[:2, 2] += (s - 1) / 2
    board_to_image = K_big @ np.column_stack([R[:, 0], R[:, 1], tvec])
    big = cv2.warpPerspective(
        texture, board_to_image @ tex_to_board, (SIZE[0] * s, SIZE[1] * s), flags=cv2.INTER_LINEAR, borderValue=180
    )
    return cv2.resize(big, SIZE, interpolation=cv2.INTER_AREA)


def project(rvec, tvec):
    points, _ = cv2.projectPoints(board_points(PATTERN, SQUARE), rvec, tvec, K_TRUE, np.zeros(5))
    return points.reshape(-1, 2)


POSES = [
    pose(rx, ry, rz, d)
    for rx, ry, rz, d in [
        (0, 0, 0, 450), (20, 0, 0, 450), (-20, 0, 0, 450), (0, 25, 0, 450), (0, -25, 0, 450),
        (15, 15, 10, 400), (-15, 20, -10, 500), (10, -20, 5, 380), (-25, -10, 0, 520), (0, 0, 30, 470),
        (20, -15, -15, 430), (-10, 10, 20, 410),
    ]
]


def test_find_corners_on_synthetic_image():
    rvec, tvec = POSES[5]
    corners = find_corners(render_board(rvec, tvec))
    assert corners is not None and corners.shape == (54, 2)

    truth = project(rvec, tvec)
    # Kolejność OpenCV może zaczynać się od innego rogu, porównujemy najbliższe punkty
    nearest = np.min(np.linalg.norm(corners[:, None] - truth[None], axis=2), axis=1)
    assert nearest.max() < 0.3


def test_find_corners_without_board():
    assert find_corners(np.full((240, 320), 128, np.uint8)) is None


def test_calibrate_recovers_intrinsics():
    rng = np.random.default_rng(0)
    corner_sets = [project(r, t) + rng.normal(0, 0.1, (54, 2)) for r, t in POSES]

    model, per_view = calibrate(corner_sets, SIZE, PATTERN, SQUARE)

    assert model.fx == pytest.approx(300, rel=0.01)
    assert model.fy == pytest.approx(300, rel=0.01)
    assert model.cx == pytest.approx(158, abs=1.5)
    assert model.cy == pytest.approx(121, abs=1.5)
    assert model.rms < 0.2
    assert len(per_view) == len(POSES)


def test_calibrate_needs_three_views():
    with pytest.raises(ValueError):
        calibrate([project(*POSES[0])] * 2, SIZE)


def test_view_signature_distinguishes_position_and_tilt():
    straight = view_signature(project(*POSES[0]), SIZE)
    tilted = view_signature(project(*POSES[1]), SIZE)
    assert straight[:2] == pytest.approx([0.5, 0.5], abs=0.03)
    assert abs(straight[3]) < 0.01
    assert np.linalg.norm(tilted - straight) > 0.1


def test_coverage_counts_corners_per_region():
    corners = np.array([[10, 10], [310, 10], [160, 120], [160, 125], [310, 230]], dtype=np.float32)
    counts = coverage([corners], SIZE)
    assert counts.tolist() == [[1, 0, 1], [0, 2, 0], [0, 0, 1]]
    assert missing_coverage(counts, min_corners=1) == [
        "górny środek", "lewy brzeg", "prawy brzeg", "lewy dolny", "dolny środek",
    ]


def test_camera_model_save_load(tmp_path):
    model = CameraModel(320, 240, K_TRUE.copy(), np.array([0.1, -0.2, 0, 0, 0]), 0.15, {"views": 20})
    model.save(tmp_path / "cam.json")
    loaded = CameraModel.load(tmp_path / "cam.json")
    np.testing.assert_allclose(loaded.K, model.K)
    np.testing.assert_allclose(loaded.dist, model.dist)
    assert loaded.rms == 0.15 and loaded.info == {"views": 20}


def test_camera_model_fov_and_scaling():
    model = CameraModel(320, 240, K_TRUE.copy(), np.zeros(5))
    fov_h, _ = model.fov_deg()
    assert fov_h == pytest.approx(2 * np.degrees(np.arctan(160 / 300)))

    half = model.scaled(160, 120)
    assert half.fx == pytest.approx(150)
    assert half.cx == pytest.approx((158 + 0.5) / 2 - 0.5)
    assert half.fov_deg() == pytest.approx(model.fov_deg())
