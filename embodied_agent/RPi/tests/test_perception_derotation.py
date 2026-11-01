import cv2
import numpy as np
import pytest

from lili.body.geometry import Pose
from lili.perception.derotation import derotate_pair, rectification
from lili.perception.flow import dense_flow
from synthetic import DISTORTED, PINHOLE, render, rotation, table_scene

SCENE = table_scene()


def interior(mask, margin=15):
    return cv2.erode(mask.astype(np.uint8), np.ones((margin, margin))).astype(bool)


def test_no_rotation_no_distortion_is_identity():
    rect = rectification(PINHOLE)
    xs, ys = np.meshgrid(np.arange(320), np.arange(240))
    np.testing.assert_allclose(rect.map_x, xs, atol=1e-4)
    np.testing.assert_allclose(rect.map_y, ys, atol=1e-4)
    assert rect.valid.all()


def test_distortion_crops_source_corners():
    # Dystorsja beczkowata ściska brzegi: widok otworkowy z tym samym K mieści się w kadrze,
    # a rogi obrazu źródłowego zostają poza nim
    rect = rectification(DISTORTED)
    assert rect.valid.all()
    assert rect.map_x[0, 0] > 10 and rect.map_y[0, 0] > 10


@pytest.mark.parametrize("camera", [PINHOLE, DISTORTED], ids=["pinhole", "distorted"])
@pytest.mark.parametrize("axis, degrees", [([0, 1, 0], 10), ([1, 0, 0], -8), ([0.2, 1, 0.1], 6)])
def test_pure_rotation_leaves_no_flow(camera, axis, degrees):
    """Kryterium M3: obrót wokół środka optycznego po derotacji to (prawie) zerowy przepływ."""
    R = rotation(axis, degrees)
    a, b = render(SCENE, camera), render(SCENE, camera, Pose(R, np.zeros(3)))
    a2, b2, valid = derotate_pair(a, b, camera, R)
    residual = np.linalg.norm(dense_flow(a2, b2), axis=-1)[interior(valid)]
    assert np.median(residual) < 0.05
    assert np.percentile(residual, 95) < 0.15
    assert interior(valid).mean() > 0.4


def test_derotation_removes_rotation_but_keeps_translation():
    R, T = rotation([0, 1, 0], 8), np.array([0.01, 0, 0])
    a, b = render(SCENE, DISTORTED), render(SCENE, DISTORTED, Pose(R, T))
    a2, b2, valid = derotate_pair(a, b, DISTORTED, R)
    # Po derotacji para wygląda jak sama translacja wyrenderowana kamerą otworkową
    a_ref, b_ref = render(SCENE, PINHOLE), render(SCENE, PINHOLE, Pose(np.eye(3), T))
    inner = interior(valid)
    assert np.median(np.abs(a2.astype(float) - a_ref)[inner]) < 3
    assert np.median(np.abs(b2.astype(float) - b_ref)[inner]) < 3
    assert np.median(np.abs(b2.astype(float) - a_ref)[inner]) > 10  # translacja zostaje
