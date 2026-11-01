from pathlib import Path

import numpy as np
import pytest

from lili.body.camera_model import CameraModel
from lili.body.efference import Efference
from lili.body.geometry import Pose
from lili.motor.programs import manual
from lili.pipeline import motion, process_pair, views
from lili.record import schema as s
from lili.record.reader import Session
from lili.record.writer import SessionWriter
from synthetic import ground_truth, render, table_scene

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
SCENE = table_scene()


def write_manual_session(root, camera, offsets_cm=(0, 2), frames_per_look=5, noise=2.0, true_shift=None):
    """Sesja jak z record_session manual: zaszumione klatki z każdego postoju i zdarzenia faz.

    true_shift: rzeczywiste położenia podstawy [m], jeśli inne niż deklarowane (niedokładna ręka).
    """
    rng = np.random.default_rng(0)
    steps = manual(offsets_cm=offsets_cm)
    t = 0
    with SessionWriter(root, "synt", {"program": {"name": "manual"}}) as writer:
        for i, step in enumerate(steps):
            position = np.array(step.offset if true_shift is None else true_shift[i])
            image = render(SCENE, camera, Pose(np.eye(3), position)).astype(float)
            writer.add_event(s.phase_event(t := t + 1, s.PHASE_MOVE, i))
            writer.add_event(s.cmd_event(t := t + 1, step.pan, step.tilt, 90, 90))
            writer.add_event(s.base_event(t := t + 1, step.offset))
            writer.add_frame(np.zeros_like(image, np.uint8), t := t + 1)  # klatka z ruchu, poza look
            writer.add_event(s.phase_event(t := t + 1, s.PHASE_SETTLE, i))
            writer.add_event(s.phase_event(t := t + 1, s.PHASE_LOOK, i))
            for _ in range(frames_per_look):
                frame = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).round().astype(np.uint8)
                writer.add_frame(frame, t := t + 1)
        writer.add_event(s.phase_event(t + 1, s.PHASE_END, len(steps)))
    return Session(writer.path)


@pytest.fixture(scope="module")
def camera():
    return CameraModel.load(CALIB_DIR / "camera_320x240.json")


def test_views_average_look_frames(tmp_path, camera):
    session = write_manual_session(tmp_path, camera, offsets_cm=(0, 2, 4))
    result = views(session)
    assert [v.step for v in result] == [0, 1, 2]
    assert all(v.frames == 5 for v in result)  # klatka z ruchu pominięta
    np.testing.assert_allclose([v.offset for v in result], [[0, 0, 0], [0.02, 0, 0], [0.04, 0, 0]])
    # Mediana 5 klatek z szumem 2 poziomy: blisko czystego obrazu
    clean = render(SCENE, camera)
    assert np.abs(result[0].image.astype(float) - clean).mean() < 1.5


def test_motion_combines_head_and_base(tmp_path, camera):
    efference = Efference.from_calibration(CALIB_DIR)
    a, b = views(write_manual_session(tmp_path, camera))
    pose = motion(a, b, efference)
    np.testing.assert_allclose(pose.R, np.eye(3), atol=1e-12)  # serwa stały
    np.testing.assert_allclose(pose.c, [0.02, 0, 0], atol=1e-12)


def test_manual_session_gives_inverse_depth_and_edges(tmp_path, camera):
    """Ręka przesunęła robota trochę inaczej, niż zadeklarowano: kierunek T z obrazu."""
    true_shift = [(0, 0, 0), (0.019, 0.002, 0.003)]
    a, b = views(write_manual_session(tmp_path, camera, true_shift=true_shift))
    declared = motion(a, b, Efference.from_calibration(CALIB_DIR))
    result = process_pair(a.image, b.image, camera, declared, refine_translation=True)

    T_true = np.array(true_shift[1])
    cos = result.T @ T_true / np.linalg.norm(result.T) / np.linalg.norm(T_true)
    assert np.degrees(np.arccos(min(cos, 1))) < 2

    truth = ground_truth(SCENE, CameraModel(camera.width, camera.height, camera.K, np.zeros(5)))
    # Długość z deklaracji (2 cm zamiast ok. 1.93 cm), a rho ~ 1 / |T|, więc rho zaniżone o ten stosunek.
    # Obiekt 1 przy lewym brzegu odjeżdża w stronę brzegu i DIS go gubi.
    scale = np.linalg.norm(T_true) / np.linalg.norm(result.T)
    for i in (0, 2, 3):
        mask = (truth.ids == i) & result.reliable
        ratio = np.median(result.rho[mask] * truth.depth[mask])
        assert ratio == pytest.approx(scale, rel=0.05), f"obiekt {i}"
    assert result.edges.sum() > 300
