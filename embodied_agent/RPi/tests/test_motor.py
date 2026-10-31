import itertools
import threading

import numpy as np
import pytest

from lili.hw.stm_link import StmLink
from lili.motor.programs import Step, peering
from lili.motor.runner import FrameRecorder, run_program


class ListWriter:
    def __init__(self):
        self.events = []
        self.frames = []

    def add_event(self, event):
        self.events.append(event)

    def add_frame(self, image, t, t_sensor=0):
        self.frames.append((image, t, t_sensor))


class FakeCamera:
    def __init__(self, fail_after=None):
        self.count = 0
        self.fail_after = fail_after
        self.enough = threading.Event()

    def capture(self):
        if self.count == self.fail_after:
            raise RuntimeError("kamera padła")
        self.count += 1
        if self.count >= 5:
            self.enough.set()
        return np.zeros((2, 2), np.uint8), self.count, 0


def test_peering_positions():
    steps = peering(amplitude=10, cycles=2, center=5, tilt=-3)
    assert [st.pan for st in steps] == [5, 15, 5, -5, 5, 15, 5, -5, 5]
    assert all(st.tilt == -3 for st in steps)


def test_peering_rejects_out_of_range():
    with pytest.raises(ValueError):
        peering(amplitude=20, center=80)


def test_run_program_records_phases_commands_and_acks(ser):
    writer, slept = ListWriter(), []
    clock = itertools.count()

    run_program([Step(10.0, -5.0, 0.2, 0.3)], StmLink(serial_port=ser), writer, lambda: next(clock), slept.append)

    assert [(e["type"], e.get("phase")) for e in writer.events] == [
        ("phase", "move"),
        ("cmd", None),
        ("stm", None),
        ("stm", None),
        ("phase", "settle"),
        ("phase", "look"),
        ("phase", "end"),
    ]
    assert [e["t"] for e in writer.events] == sorted(e["t"] for e in writer.events)
    assert writer.events[1] | {"t": 0} == {
        "t": 0, "type": "cmd", "pan": 10.0, "tilt": -5.0, "pan_byte": 100, "tilt_byte": 85,
    }
    assert [e["position"] for e in writer.events if e["type"] == "stm"] == [10.0, -5.0]
    assert slept == [0.2, 0.3]
    assert ser.positions == {0: 100, 1: 85}


def test_frame_recorder_records_until_stopped():
    camera, writer = FakeCamera(), ListWriter()
    with FrameRecorder(camera, writer):
        assert camera.enough.wait(timeout=2)

    assert len(writer.frames) >= 5
    assert [t for _, t, _ in writer.frames] == list(range(1, len(writer.frames) + 1))


def test_frame_recorder_reraises_camera_error():
    recorder = FrameRecorder(FakeCamera(fail_after=3), ListWriter())
    recorder.start()
    with pytest.raises(RuntimeError, match="kamera"):
        recorder.stop()
