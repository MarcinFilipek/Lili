from pathlib import Path

import cv2
import numpy as np
import pytest

from lili.body.camera_model import CameraModel
from lili.body.efference import Efference
from lili.body.geometry import Pose
from lili.perception.derotation import derotate_pair
from lili.perception.flow import dense_flow, forward_backward_error
from lili.perception.parallax import epipolar_directions, estimate_translation, parallax, rho_sigma
from synthetic import IDENTITY, PINHOLE, ground_truth, render, rotation, table_scene, true_flow

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
SCENE = table_scene()


@pytest.mark.parametrize("T", [[0.02, 0, 0], [-0.01, 0.005, 0], [0.005, 0, 0.03], [0, 0, -0.02]])
def test_exact_flow_gives_exact_inverse_depth(T):
    flow, visible = true_flow(SCENE, PINHOLE, IDENTITY, Pose(np.eye(3), np.array(T)))
    par = parallax(flow, PINHOLE.K, T)
    rho_true = 1 / ground_truth(SCENE, PINHOLE).depth
    ok = visible & np.isfinite(par.rho)
    assert ok.mean() > 0.7
    np.testing.assert_allclose(par.rho[ok], rho_true[ok], rtol=1e-5)
    assert par.residual[visible].max() < 1e-3


def test_gain_vanishes_at_epipole():
    # Ruch do przodu: epipol w punkcie głównym, tam paralaksy nie ma
    T = np.array([0, 0, 0.03])
    gain = np.linalg.norm(epipolar_directions(PINHOLE.K, T, (240, 320)), axis=-1)
    assert gain[120, 160] == pytest.approx(0, abs=1e-9)
    assert gain[0, 0] == pytest.approx(np.hypot(160, 120) * 0.03)  # |p - c| * Tz
    par = parallax(np.zeros((240, 320, 2), np.float32), PINHOLE.K, T)
    assert np.isnan(par.rho[120, 160]) and rho_sigma(par)[120, 160] == np.inf


@pytest.mark.parametrize("T", [[0.02, 0.003, 0.004], [-0.015, 0.0, -0.005], [0.002, 0.018, 0.0]])
def test_estimate_translation_direction_from_exact_flow(T):
    T = np.array(T)
    flow, visible = true_flow(SCENE, PINHOLE, IDENTITY, Pose(np.eye(3), T))
    guess = np.sign(T) * (np.abs(T) == np.abs(T).max()) * 0.02  # tylko główna oś, jak deklaracja ręki
    estimated = estimate_translation(flow, PINHOLE.K, guess, visible)
    np.testing.assert_allclose(estimated / np.linalg.norm(estimated), T / np.linalg.norm(T), atol=1e-4)
    assert np.linalg.norm(estimated) == pytest.approx(0.02)


def test_relative_scale():
    """T znane z dokładnością do skali: rho w tej samej skali."""
    T = np.array([0.02, 0, 0])
    flow, visible = true_flow(SCENE, PINHOLE, IDENTITY, Pose(np.eye(3), T))
    np.testing.assert_allclose(parallax(flow, PINHOLE.K, T / 0.02).rho[visible], parallax(flow, PINHOLE.K, T).rho[visible] * 0.02, rtol=1e-6)


def interior(mask, margin):
    return cv2.erode(mask.astype(np.uint8), np.ones((margin, margin))).astype(bool)


def run_chain(motion, R_assumed=None):
    """Render A, B kamerą z kalibracji -> derotacja -> DIS -> rho. Prawda w układzie A po korekcji dystorsji."""
    camera = CameraModel.load(CALIB_DIR / "camera_320x240.json")
    a, b = render(SCENE, camera), render(SCENE, camera, motion)
    a2, b2, valid = derotate_pair(a, b, camera, motion.R if R_assumed is None else R_assumed)
    flow, back = dense_flow(a2, b2), dense_flow(b2, a2)
    par = parallax(flow, camera.K, motion.c)
    sigma = rho_sigma(par, fb_error=forward_backward_error(flow, back))

    pinhole = CameraModel(camera.width, camera.height, camera.K, np.zeros(5))
    truth = ground_truth(SCENE, pinhole)
    _, visible = true_flow(SCENE, pinhole, IDENTITY, Pose(np.eye(3), motion.c))
    return par.rho, sigma, 1 / truth.depth, truth.ids, interior(valid, 15) & visible


def object_masks(ids, usable):
    """Wnętrza obiektów: bez pasa 3 px przy krawędziach zasłaniania."""
    return [interior(ids == i, 7) & usable for i in range(len(SCENE.rects))]


@pytest.mark.parametrize("cmd_b", [(10, 0), (-10, 0), (15, 0)])
def test_peering_recovers_inverse_depth(cmd_b):
    """Kryterium M3: pan z prawdziwej kalibracji (obrót + translacja z ramienia 9,9 cm) -> 1/Z."""
    motion = Efference.from_calibration(CALIB_DIR).motion((0, 0), cmd_b)
    rho, sigma, rho_true, ids, usable = run_chain(motion)
    confident = sigma < 0.1
    for i, mask in enumerate(object_masks(ids, usable)):
        mask &= confident
        if mask.sum() < 500:
            continue  # obiekt poza wspólnym polem widzenia (np. wyjechał za brzeg po obrocie)
        rel = (rho[mask] - rho_true[mask]) / rho_true[mask]
        assert abs(np.median(rel)) < 0.03, f"obiekt {i}"
        assert np.percentile(np.abs(rel), 90) < 0.1, f"obiekt {i}"
    # Na pikselach uznanych za pewne odzyskana głębia zgadza się z niepewnością
    covered = usable & confident
    assert covered.mean() > 0.4
    assert np.median(np.abs(rho - rho_true)[covered] / sigma[covered]) < 1


def test_uncertain_where_objects_leave_shared_view():
    """Pan -10: obiekt przy lewym brzegu wjeżdża w pas bez odpowiednika w B' i dostaje dużą niepewność."""
    motion = Efference.from_calibration(CALIB_DIR).motion((0, 0), (-10, 0))
    rho, sigma, rho_true, ids, usable = run_chain(motion)
    wrong = usable & (np.abs(rho - rho_true) > 0.3 * rho_true)
    assert wrong.sum() > 1000
    assert np.median(sigma[wrong]) > 0.3


def test_rotation_error_keeps_depth_jumps():
    """Błąd kąta 0,5 deg przesuwa rho prawie o stałą, ale skoki między obiektami i tłem zostają."""
    motion = Efference.from_calibration(CALIB_DIR).motion((0, 0), (10, 0))
    R_wrong = motion.R @ rotation([0, 1, 0], 0.5)
    rho, _, rho_true, ids, usable = run_chain(motion, R_wrong)
    medians = [np.median(rho[m]) for m in object_masks(ids, usable)]
    medians_true = [np.median(rho_true[m]) for m in object_masks(ids, usable)]
    offset = np.array(medians) - medians_true
    assert np.abs(offset).max() > 0.2  # absolutne rho wyraźnie przesunięte
    jumps, jumps_true = np.diff(medians), np.diff(medians_true)
    np.testing.assert_allclose(jumps, jumps_true, rtol=0.15)
