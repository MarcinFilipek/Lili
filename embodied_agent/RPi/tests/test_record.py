import numpy as np
import pytest

from lili.record import schema as s
from lili.record.reader import Session, list_sessions
from lili.record.writer import SessionWriter


def write_session(root, n_frames, chunk_size=3):
    with SessionWriter(root, "test", {"note": "x"}, chunk_size=chunk_size) as writer:
        for i in range(n_frames):
            writer.add_frame(np.full((4, 6), i, dtype=np.uint8), t=1_000 + i * 10, t_sensor=5_000 + i)
        writer.add_event(s.phase_event(1_001, s.PHASE_LOOK, 0))
        writer.add_event(s.stm_event(1_002, 0, 12.0))
    return writer.path


def test_roundtrip_across_chunks(tmp_path):
    session = Session(write_session(tmp_path, 7))
    frames = session.load_frames()

    assert frames.images.shape == (7, 4, 6)
    np.testing.assert_array_equal(frames.images[:, 0, 0], np.arange(7))
    np.testing.assert_array_equal(frames.t, 1_000 + np.arange(7) * 10)
    np.testing.assert_array_equal(frames.t_sensor, 5_000 + np.arange(7))
    np.testing.assert_array_equal(frames.index, np.arange(7))
    assert len(list((session.path / s.FRAMES_DIR).glob("*.npz"))) == 3


def test_meta_and_events(tmp_path):
    session = Session(write_session(tmp_path, 7))

    assert session.meta["note"] == "x"
    assert session.meta["schema_version"] == s.SCHEMA_VERSION
    assert session.meta["frames"] == 7
    assert session.meta["events"] == 2
    assert session.meta["duration_s"] == pytest.approx(60e-9)
    assert session.events(kind="stm") == [s.stm_event(1_002, 0, 12.0)]
    assert len(session.events()) == 2


def test_iter_frames_and_times_match_load(tmp_path):
    session = Session(write_session(tmp_path, 5))
    frames = session.load_frames()

    iterated = list(session.iter_frames())
    assert [f.index for f in iterated] == list(frames.index)
    np.testing.assert_array_equal(np.stack([f.image for f in iterated]), frames.images)

    t, t_sensor = session.frame_times()
    np.testing.assert_array_equal(t, frames.t)
    np.testing.assert_array_equal(t_sensor, frames.t_sensor)


def test_session_without_frames(tmp_path):
    session = Session(write_session(tmp_path, 0))
    assert session.meta["frames"] == 0
    assert session.load_frames().images.size == 0
    assert list(session.iter_frames()) == []


def test_list_sessions(tmp_path):
    path = write_session(tmp_path, 1)
    (tmp_path / "not_a_session").mkdir()
    assert list_sessions(tmp_path) == [path]
    assert list_sessions(tmp_path / "missing") == []
