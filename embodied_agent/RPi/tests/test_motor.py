import itertools
import threading

import numpy as np
import pytest

from lili.hw.stm_link import StmLink
from lili.motor.programs import Step, manual, peering, sweep
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


def test_sweep_goes_there_and_back_on_one_axis():
    steps = sweep("tilt", limit=30, step=15, fixed=5)
    assert [st.tilt for st in steps] == [-30, -15, 0, 15, 30, 15, 0, -15, -30]
    assert all(st.pan == 5 for st in steps)


def test_sweep_rejects_bad_axis_and_range():
    with pytest.raises(ValueError):
        sweep("roll")
    with pytest.raises(ValueError):
        sweep("pan", limit=95)


def test_manual_holds_servos_and_declares_offsets():
    steps = manual(offsets_cm=(0, 2, 4), direction="lewo", pan=5, tilt=-3)
    assert [(st.pan, st.tilt) for st in steps] == [(5, -3)] * 3
    assert [st.offset for st in steps] == [(-0.0, 0.0, 0.0), (-0.02, 0.0, 0.0), (-0.04, 0.0, 0.0)]
    assert all(st.prompt for st in steps) and "lewo do 4 cm" in steps[2].prompt


def test_manual_rejects_bad_direction():
    with pytest.raises(ValueError):
        manual(direction="góra")


def test_run_program_waits_for_operator_and_logs_base(ser):
    writer, asked = ListWriter(), []
    clock = itertools.count()
    steps = manual(offsets_cm=(0, 2), settle_s=1.0, look_s=0.5)

    run_program(steps, StmLink(serial_port=ser), writer, lambda: next(clock), lambda _: None, asked.append)

    assert asked == [st.prompt for st in steps]
    kinds = [(e["type"], e.get("phase")) for e in writer.events]
    # Położenie podstawy zapisane po potwierdzeniu operatora, przed odczekaniem drgań
    assert kinds[:6] == [("phase", "move"), ("cmd", None), ("stm", None), ("stm", None), ("base", None), ("phase", "settle")]
    assert [e["offset"] for e in writer.events if e["type"] == "base"] == [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]


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
