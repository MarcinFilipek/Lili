from pathlib import Path

import numpy as np
import pytest

from lili.body.efference import Efference
from lili.body.geometry import Head, Joint, Pose, joint_behind_camera
from lili.body.servo_model import ServoModel

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

# Oś pan pionowo (y w dół, więc obrót w prawo to -y), tilt wokół x
PAN = joint_behind_camera([0, -1, 0], 0.1)
TILT = joint_behind_camera([-1, 0, 0], 0.03)


def test_joint_behind_camera_places_axis_behind_lens():
    np.testing.assert_allclose(PAN.point, [0, 0, -0.1])
    # Oś pochylona względem osi optycznej: punkt i tak leży 0.1 m od obiektywu, prostopadle do osi
    tilted = joint_behind_camera([0, -0.98, -0.18], 0.1)
    assert np.linalg.norm(tilted.point) == pytest.approx(0.1)
    assert tilted.point @ tilted.axis == pytest.approx(0)


def test_zero_pose_is_reference():
    pose = Head(PAN, TILT).pose(0, 0)
    np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pose.c, 0, atol=1e-12)


def test_pan_moves_camera_along_arc():
    pose = Head(PAN, TILT).pose(10, 0)
    # Cięciwa łuku o promieniu 0.1 m
    assert np.linalg.norm(pose.c) == pytest.approx(2 * 0.1 * np.sin(np.radians(5)))
    # Obiektyw dalej 0.1 m od osi
    assert np.linalg.norm(pose.c - PAN.point) == pytest.approx(0.1)
    # Kąt między osiami optycznymi = kąt obrotu
    assert np.degrees(np.arccos(pose.R[:, 2] @ [0, 0, 1])) == pytest.approx(10)


def test_tilt_axis_follows_pan():
    head = Head(PAN, TILT)
    panned = head.pose(90, 0)
    both = head.pose(90, 10)
    # Tilt obraca kamerę wokół jej własnej osi x, niezależnie od pan
    relative = both.relative_to(panned)
    axis = np.cross([0, 0, 1], relative.R[:, 2])
    np.testing.assert_allclose(axis / np.linalg.norm(axis), [-1, 0, 0], atol=1e-12)


def test_relative_pose_maps_points_consistently():
    head = Head(PAN, TILT)
    a, b = head.pose(-5, 3), head.pose(8, -4)
    points = np.array([[0.1, -0.2, 1.0], [-0.5, 0.3, 3.0]])
    motion = b.relative_to(a)
    in_a = a.to_camera(points)
    expected = Pose(motion.R, motion.c).to_camera(in_a)
    np.testing.assert_allclose(b.to_camera(points), expected, atol=1e-12)


def test_efference_uses_servo_scale():
    pan = ServoModel("pan", 0, 0.9, [0, -1, 0], [[-90, -81], [90, 81]], 50, 200)
    tilt = ServoModel("tilt", 1, 0.3, [-1, 0, 0], [[-90, -27], [90, 27]], 50, 200)
    efference = Efference(Head(PAN, TILT), pan, tilt)
    motion = efference.motion((0, 0), (10, 0))
    assert np.degrees(np.arccos(motion.R[2, 2])) == pytest.approx(9)


def test_efference_from_repo_calibration():
    efference = Efference.from_calibration(CALIB_DIR)
    lever = efference.head.pan.point
    motion = efference.motion((0, 0), (10, 0))
    angle = np.arccos((np.trace(motion.R) - 1) / 2)
    # Sam pan: kamera jedzie po okręgu wokół osi pan, przesunięcie = cięciwa łuku
    assert 8 < np.degrees(angle) < 11
    assert np.linalg.norm(motion.c) == pytest.approx(2 * np.linalg.norm(lever) * np.sin(angle / 2))
