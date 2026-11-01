import numpy as np
import pytest

from lili.body.geometry import Pose
from lili.perception.flow import dense_flow, forward_backward_error, warp_by_flow
from synthetic import IDENTITY, PINHOLE, noise_texture, render, table_scene, true_flow

TEXTURE = noise_texture(400, 300, 5).astype(np.uint8)


@pytest.mark.parametrize("method", ["dis", "farneback"])
def test_dense_flow_recovers_shift(method):
    a = TEXTURE[20:260, 20:340]
    b = TEXTURE[23:263, 15:335]  # treść przesunięta o (+5, -3) px
    flow = dense_flow(a, b, method)
    assert flow.shape == (240, 320, 2) and flow.dtype == np.float32
    inner = flow[20:-20, 20:-20].reshape(-1, 2)
    np.testing.assert_allclose(np.median(inner, axis=0), [5, -3], atol=0.05)


def test_warp_by_flow_undoes_shift():
    a = TEXTURE[20:260, 20:340]
    b = TEXTURE[23:263, 15:335]
    flow = np.zeros((240, 320, 2), np.float32)
    flow[...] = [5, -3]
    np.testing.assert_array_equal(warp_by_flow(b, flow)[10:-10, 10:-10], a[10:-10, 10:-10])


def test_forward_backward_error_marks_occlusions():
    scene = table_scene()
    pose_b = Pose(np.eye(3), np.array([0.015, 0, 0]))
    a, b = render(scene, PINHOLE), render(scene, PINHOLE, pose_b)
    flow, visible = true_flow(scene, PINHOLE, IDENTITY, pose_b)
    in_frame = np.abs(np.arange(320) + flow[..., 0] - 159.5) < 160
    occluded = ~visible & in_frame  # tło zasłonięte w B przez bliższe obiekty
    fb = forward_backward_error(dense_flow(a, b), dense_flow(b, a))
    assert np.median(fb[visible]) < 0.05
    assert np.median(fb[occluded]) > 0.5
