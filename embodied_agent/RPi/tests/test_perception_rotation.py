import cv2
import numpy as np
import pytest

from lili.body.camera_model import CameraModel
from lili.perception.rotation import estimate_rotation, fit_rotation, image_shift, ransac_rotation

SIZE = (320, 240)
CAMERA = CameraModel(320, 240, np.array([[376.0, 0, 160.0], [0, 376.0, 120.0], [0, 0, 1]]), np.zeros(5))


def scene_texture(seed=0):
    """Daleka scena jako obraz z szerszej kamery o tym samym f: plamy różnej wielkości, dużo narożników."""
    rng = np.random.default_rng(seed)
    texture = rng.random((720, 960)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 3) + 0.7 * cv2.GaussianBlur(texture, (0, 0), 1.0)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    return texture.astype(np.uint8)


TEXTURE = scene_texture()
K_TEXTURE = np.array([[376.0, 0, 480.0], [0, 376.0, 360.0], [0, 0, 1]])


def render(R_cam):
    """Widok kamery CAMERA obróconej o R_cam (osie kamery w układzie sceny) na scenę w nieskończoności."""
    # Piksel widoku -> promień w układzie kamery -> promień w układzie sceny -> piksel tekstury
    view_to_texture = K_TEXTURE @ R_cam @ np.linalg.inv(CAMERA.K)
    return cv2.warpPerspective(TEXTURE, view_to_texture, SIZE, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    return cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(degrees))[0]


@pytest.mark.parametrize("axis, degrees", [([0, 1, 0], 5.0), ([1, 0, 0], -4.0), ([0.2, 1, 0.1], 6.0)])
def test_estimate_rotation_recovers_camera_rotation(axis, degrees):
    R_true = rotation(axis, degrees)
    R, inliers = estimate_rotation(render(np.eye(3)), render(R_true), CAMERA)

    error = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_true.T)[0]))
    assert error < 0.05
    assert inliers >= 30  # próg, przy którym calibrate_servo ostrzega


def test_fit_rotation_exact():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R = rotation([1, 2, 3], 30)
    np.testing.assert_allclose(fit_rotation(a, a @ R.T), R, atol=1e-12)


def test_ransac_rotation_ignores_outliers():
    rng = np.random.default_rng(2)
    a = np.column_stack([rng.uniform(-0.4, 0.4, (200, 2)), np.ones(200)])
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R = rotation([0, 1, 0], 5)
    b = a @ R.T
    b[:40] = np.roll(b[:40], 1, axis=0)  # 20% błędnych dopasowań
    R_fit, inliers = ransac_rotation(a, b, threshold=0.003)
    np.testing.assert_allclose(R_fit, R, atol=1e-9)
    assert inliers[40:].all() and inliers.sum() < 200


def test_image_shift():
    a = TEXTURE[:240, :320]
    b = TEXTURE[3:243, 5:325]  # scena przesunięta o (-5, -3) px
    dx, dy = image_shift(a, b)
    assert (dx, dy) == pytest.approx((-5, -3), abs=0.1)
