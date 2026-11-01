import numpy as np
import pytest

from lili.body.servo_model import ServoModel, fit_axis, load_servo_models, motion_times, save_servo_model
from lili.motor.programs import sweep


def sweep_commands(limit=30, step=5):
    return [st.pan for st in sweep("pan", limit, step)]


def test_fit_axis_recovers_scale_axis_and_hysteresis():
    commands = np.array(sweep_commands())
    axis = np.array([0.1, 0.99, 0.0])
    axis /= np.linalg.norm(axis)

    # Rzeczywisty kąt: 0.9 deg na jednostkę, dojazd od góry zostaje 0.4 deg z tyłu (luz)
    deltas = np.diff(commands)
    direction = np.concatenate([[1.0], np.sign(deltas)])
    angle = 0.9 * commands - 0.2 * direction
    rvecs = np.outer(np.radians(np.diff(angle)), axis)

    fit = fit_axis(commands, rvecs)
    np.testing.assert_allclose(fit.axis, axis, atol=1e-9)
    assert fit.deg_per_unit == pytest.approx(0.9, abs=0.01)
    assert fit.hysteresis == pytest.approx(0.4, abs=1e-6)
    assert fit.off_axis_rms == pytest.approx(0, abs=1e-9)
    assert dict(map(tuple, fit.table))[0.0] == pytest.approx(0)


def test_fit_axis_sign_follows_command():
    commands = [0, 10, 20]
    rvecs = np.outer(np.radians([-3, -3]), [0, 1, 0])  # obrót wokół -y przy rosnącej komendzie
    fit = fit_axis(commands, rvecs)
    np.testing.assert_allclose(fit.axis, [0, -1, 0])
    assert fit.deg_per_unit == pytest.approx(0.3)


def test_motion_times():
    times = np.arange(0, 1.0, 0.03)
    t_ack = 0.1
    # Ruch zaczyna się 60 ms po ACK, kończy po 300 ms
    position = np.clip((times - 0.16) / 0.24, 0, 1) * 20
    latency, settle = motion_times(times, position, 20 - position, t_ack)
    assert latency == pytest.approx(0.09, abs=0.031)
    assert settle == pytest.approx(0.3, abs=0.031)


def test_motion_times_without_motion():
    times = np.arange(0, 0.5, 0.03)
    assert motion_times(times, np.zeros_like(times), np.zeros_like(times), 0.1) == (None, None)


def test_servo_model_angle_and_save(tmp_path):
    pan = ServoModel("pan", 0, 0.9, [0, 1, 0], [[-10, -9.2], [0, 0], [10, 8.8]], 40.0, 300.0)
    assert pan.angle(5) == pytest.approx(4.4)
    assert pan.angle(20) == pytest.approx(8.8 + 9)

    path = tmp_path / "servo.json"
    save_servo_model(path, pan)
    save_servo_model(path, ServoModel("tilt", 1, 0.3, [1, 0, 0], [[0, 0]], 40.0, 200.0))
    models = load_servo_models(path)
    assert set(models) == {"pan", "tilt"}
    assert models["pan"] == pan
