import cv2
import numpy as np
import pytest

from lili.body.geometry import Pose
from synthetic import (
    DISTORTED, IDENTITY, PINHOLE, Rect, Scene, ground_truth, pixel_grid, pixels_to_normalized, project, render,
    rotation, table_scene, true_flow,
)


def test_ground_truth_depth_of_facing_plane():
    scene = Scene((Rect.facing([0, 0, 2.0], 10, 10, 10, 0),))
    hits = ground_truth(scene, DISTORTED)
    np.testing.assert_allclose(hits.depth, 2.0)
    assert (hits.ids == 0).all()


def test_nearer_rect_occludes():
    scene = table_scene()
    hits = ground_truth(scene, PINHOLE)
    # Środek obiektu 1 (0.35 m) przed tłem (3 m)
    center = project(scene.rects[1].center, PINHOLE)
    assert hits.ids[round(center[1]), round(center[0])] == 1
    assert set(np.unique(hits.ids)) == {0, 1, 2, 3}


def test_distortion_round_trip():
    grid = pixel_grid(DISTORTED.width, DISTORTED.height)
    xy = pixels_to_normalized(grid, DISTORTED)
    back = project(np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], axis=-1), DISTORTED)
    np.testing.assert_allclose(back, grid, atol=1e-6)


def test_pure_rotation_flow_is_homography():
    R = rotation([0.1, 1, 0], 8)
    flow, visible = true_flow(table_scene(), PINHOLE, IDENTITY, Pose(R, np.zeros(3)))
    grid = pixel_grid(PINHOLE.width, PINHOLE.height)
    # Piksel A -> piksel B: K R^T K^-1, niezależnie od głębi
    H = PINHOLE.K @ R.T @ np.linalg.inv(PINHOLE.K)
    expected = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(grid.shape) - grid
    np.testing.assert_allclose(flow[visible], expected[visible], atol=1e-3)


def test_translation_flow_is_f_t_over_z():
    T = 0.02
    scene = table_scene()
    flow, visible = true_flow(scene, PINHOLE, IDENTITY, Pose(np.eye(3), np.array([T, 0, 0])))
    depth = ground_truth(scene, PINHOLE).depth
    np.testing.assert_allclose(flow[..., 0][visible], -PINHOLE.fx * T / depth[visible], rtol=1e-6)
    np.testing.assert_allclose(flow[..., 1][visible], 0, atol=1e-6)
    # Bliski obiekt zasłania część tła, które było widoczne w A
    assert 0.8 < visible.mean() < 1


def test_render_matches_true_flow():
    """Jasność piksela A = jasność B w miejscu wskazanym przez prawdziwy przepływ."""
    scene = table_scene()
    pose_b = Pose(rotation([0, 1, 0], 6), np.array([0.01, 0.002, 0.003]))
    a, b = render(scene, DISTORTED), render(scene, DISTORTED, pose_b)
    flow, visible = true_flow(scene, DISTORTED, IDENTITY, pose_b)
    xs, ys = np.meshgrid(np.arange(320, dtype=np.float32), np.arange(240, dtype=np.float32))
    warped = cv2.remap(b.astype(np.float32), xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR)
    # borderValue=0: przycina też brzegi kadru, gdzie przepływ wskazuje tuż za obraz B
    inner = cv2.erode(visible.astype(np.uint8), np.ones((5, 5)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    diff = np.abs(warped - a)[inner.astype(bool)]
    # Na 0..255 resztka z interpolacji i zaokrągleń; przepływ błędny o 0,5 px daje ok. 5 / 13
    assert np.median(diff) < 2.5 and np.percentile(diff, 90) < 6
    assert np.median(np.abs(np.roll(a, 3, axis=1).astype(float) - a)) > 20  # a tekstura jest wyraźna


@pytest.mark.parametrize("camera", [PINHOLE, DISTORTED], ids=["pinhole", "distorted"])
def test_render_shape_and_range(camera):
    image = render(table_scene(), camera)
    assert image.shape == (240, 320) and image.dtype == np.uint8
    assert image.std() > 30
