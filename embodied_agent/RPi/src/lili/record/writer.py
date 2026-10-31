import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from . import schema as s


class SessionWriter:
    """Zapis sesji na dysk. Całe I/O idzie w osobnym wątku, więc add_frame/add_event nie blokują."""

    def __init__(self, root, label, meta=None, chunk_size=100):
        now = datetime.now()
        self.path = Path(root) / f"{now:%Y%m%d-%H%M%S}_{label}"
        (self.path / s.FRAMES_DIR).mkdir(parents=True)

        self.meta = {
            "schema_version": s.SCHEMA_VERSION,
            "label": label,
            "created": now.isoformat(timespec="seconds"),
            **(meta or {}),
        }
        self._write_meta()

        self.chunk_size = chunk_size
        self._queue = queue.Queue()
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add_frame(self, image, t, t_sensor=0):
        self._queue.put(("frame", (image, t, t_sensor)))

    def add_event(self, event):
        self._queue.put(("event", event))

    def close(self):
        """Zapisuje resztę danych i podsumowanie w meta.json. Rzuca błąd z wątku zapisu, jeśli był."""
        self._queue.put(None)
        self._thread.join()
        self._write_meta()
        if self._error is not None:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self):
        images, times, sensor_times = [], [], []
        n_frames = n_chunks = n_events = 0
        t_first = t_last = None

        def flush():
            nonlocal n_chunks
            if not images:
                return
            np.savez_compressed(
                self.path / s.FRAMES_DIR / s.chunk_name(n_chunks),
                **{
                    s.FRAMES: np.stack(images),
                    s.T: np.array(times, dtype=np.int64),
                    s.T_SENSOR: np.array(sensor_times, dtype=np.int64),
                    s.INDEX: np.arange(n_frames - len(images), n_frames, dtype=np.int64),
                },
            )
            n_chunks += 1
            images.clear()
            times.clear()
            sensor_times.clear()

        closed = False
        try:
            with open(self.path / s.STATE_FILE, "w") as state:
                while (item := self._queue.get()) is not None:
                    kind, payload = item
                    if kind == "event":
                        state.write(json.dumps(payload) + "\n")
                        n_events += 1
                        continue

                    image, t, t_sensor = payload
                    images.append(image)
                    times.append(t)
                    sensor_times.append(t_sensor)
                    n_frames += 1
                    t_first = t if t_first is None else t_first
                    t_last = t
                    if len(images) == self.chunk_size:
                        flush()
                closed = True
                flush()
        except Exception as e:
            self._error = e
            # Opróżnij kolejkę do sygnału końca, żeby close() się nie zawiesił
            while not closed and self._queue.get() is not None:
                pass

        self.meta.update(
            frames=n_frames,
            chunks=n_chunks,
            events=n_events,
            duration_s=(t_last - t_first) / 1e9 if n_frames > 1 else 0.0,
        )

    def _write_meta(self):
        (self.path / s.META_FILE).write_text(json.dumps(self.meta, indent=2, default=str))
