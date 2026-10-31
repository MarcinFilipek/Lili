import threading
import time

from lili.hw.stm_link import CHANNEL_PAN, CHANNEL_TILT
from lili.hw.stm_link.protocol import degrees_to_byte
from lili.record import schema as s


def run_program(steps, link, writer, clock=time.monotonic_ns, sleep=time.sleep):
    """Wykonuje kroki ruch-stop-patrz, zapisując fazy, komendy i odpowiedzi STM32."""
    for i, step in enumerate(steps):
        writer.add_event(s.phase_event(clock(), s.PHASE_MOVE, i))
        writer.add_event(
            s.cmd_event(clock(), step.pan, step.tilt, degrees_to_byte(step.pan), degrees_to_byte(step.tilt))
        )
        for channel, degrees in ((CHANNEL_PAN, step.pan), (CHANNEL_TILT, step.tilt)):
            position = link.set_position(channel, degrees)
            writer.add_event(s.stm_event(clock(), channel, position))

        writer.add_event(s.phase_event(clock(), s.PHASE_SETTLE, i))
        sleep(step.settle_s)
        writer.add_event(s.phase_event(clock(), s.PHASE_LOOK, i))
        sleep(step.look_s)

    writer.add_event(s.phase_event(clock(), s.PHASE_END, len(steps)))


class FrameRecorder:
    """Ciągłe nagrywanie klatek z kamery do writera w osobnym wątku, od start() do stop()."""

    def __init__(self, camera, writer):
        self.camera = camera
        self.writer = writer
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise self._error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _run(self):
        try:
            while not self._stop.is_set():
                image, t, t_sensor = self.camera.capture()
                self.writer.add_frame(image, t, t_sensor)
        except Exception as e:
            self._error = e
