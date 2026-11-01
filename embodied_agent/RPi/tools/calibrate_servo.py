"""Kalibracja serwa z sesji nagranej programem sweep (record_session.py sweep ...). Działa na RPi i na Macu.

Mierzy kamerą: rzeczywisty obrót na jednostkę komendy, oś obrotu w układzie kamery,
histerezę oraz czas od ACK z STM do początku ruchu i do zatrzymania obrazu.
"""

import argparse
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from lili.body.camera_model import CameraModel
from lili.body.servo_model import ServoModel, fit_axis, motion_times, save_servo_model
from lili.hw.stm_link import CHANNEL_PAN, CHANNEL_TILT
from lili.perception.rotation import estimate_rotation, image_shift
from lili.record.reader import Session, list_sessions

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = {"pan": CHANNEL_PAN, "tilt": CHANNEL_TILT}
SHIFT_THRESHOLD_PX = 0.5
BEFORE_ACK_NS = 100_000_000  # klatki sprzed ACK, żeby było widać obraz jeszcze nieruchomy


def step_intervals(events):
    """Dla każdego kroku (t_move, t_look, t_koniec_patrzenia)."""
    phases = [e for e in events if e["type"] == "phase"]
    moves = [e["t"] for e in phases if e["phase"] == "move"]
    looks = [e["t"] for e in phases if e["phase"] == "look"]
    ends = moves[1:] + [e["t"] for e in phases if e["phase"] == "end"]
    return list(zip(moves, looks, ends))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, nargs="?", help="domyślnie najnowsza sesja")
    ap.add_argument("--camera", type=Path, default=ROOT / "calib" / "camera_320x240.json")
    ap.add_argument("--out", type=Path, default=ROOT / "calib" / "servo.json")
    args = ap.parse_args()

    session = Session(args.session or list_sessions(ROOT / "data" / "sessions")[-1])
    program = session.meta.get("program", {})
    if program.get("name") != "sweep":
        raise SystemExit(f"{session.path.name} nie jest nagraniem sweep")
    name, channel = program["axis"], CHANNELS[program["axis"]]
    camera = CameraModel.load(args.camera)

    frames = session.load_frames()
    times = frames.t_sensor if np.all(frames.t_sensor > 0) else frames.t
    events = session.events()
    intervals = step_intervals(events)
    commands = [e[name] for e in events if e["type"] == "cmd"]
    acks = [e["t"] for e in events if e["type"] == "stm" and e["channel"] == channel]
    if not len(intervals) == len(commands) == len(acks):
        raise SystemExit("Niespójne zdarzenia w sesji")

    def frames_between(t0, t1):
        selected = (times >= t0) & (times < t1)
        return frames.images[selected], times[selected]

    # Nieruchomy obraz każdego kroku: średnia klatek z fazy patrzenia
    statics = []
    for i, (_, look, end) in enumerate(intervals):
        images, _ = frames_between(look, end)
        if len(images) == 0:
            raise SystemExit(f"Krok {i}: brak klatek w fazie patrzenia")
        statics.append(images.astype(np.float32).mean(axis=0).round().astype(np.uint8))

    rvecs, inliers = [], []
    for a, b in zip(statics, statics[1:]):
        R, n = estimate_rotation(a, b, camera)
        rvecs.append(cv2.Rodrigues(R)[0].ravel())
        inliers.append(n)
    fit = fit_axis(commands, rvecs)

    window = cv2.createHanningWindow((camera.width, camera.height), cv2.CV_32F)
    latencies, settles = [], []
    for i in range(1, len(intervals)):
        if commands[i] == commands[i - 1]:
            continue
        images, t = frames_between(acks[i] - BEFORE_ACK_NS, intervals[i][2])
        from_previous = np.array([np.hypot(*image_shift(statics[i - 1], im, window)) for im in images])
        to_final = np.array([np.hypot(*image_shift(statics[i], im, window)) for im in images])
        latency, settle = motion_times(t / 1e9, from_previous, to_final, acks[i] / 1e9, SHIFT_THRESHOLD_PX)
        if latency is not None:
            latencies.append(latency * 1000)
        if settle is not None:
            settles.append(settle * 1000)

    print(f"{session.path.name}: {name} (kanał {channel}), {len(commands)} kroków")
    print(f"\n{'krok':>4} {'komenda':>8} {'kąt [deg]':>10} {'punkty':>7}")
    for i, (command, angle) in enumerate(zip(commands, fit.cumulative)):
        print(f"{i:>4} {command:>8.0f} {angle:>10.2f} {inliers[i - 1] if i else '':>7}")

    print(f"\nskala: {fit.deg_per_unit:.4f} deg na jednostkę komendy")
    print(f"oś obrotu kamery (x, y, z): {np.array2string(fit.axis, precision=3)}")
    print(f"odchyłka od prostej: {fit.residual_rms:.2f} deg, chybotanie poza osią: {fit.off_axis_rms:.2f} deg")
    print(f"histereza (od dołu minus od góry): {fit.hysteresis:.2f} deg")
    if latencies and settles:
        print(f"ACK -> początek ruchu: mediana {np.median(latencies):.0f} ms, maks. {np.max(latencies):.0f} ms")
        print(
            f"ACK -> zatrzymanie: mediana {np.median(settles):.0f} ms, "
            f"95% {np.percentile(settles, 95):.0f} ms, maks. {np.max(settles):.0f} ms"
        )
    if min(inliers) < 30:
        print(f"UWAGA: mało zgodnych punktów (min. {min(inliers)}). Scena za mało szczegółowa albo za blisko?")

    model = ServoModel(
        name=name,
        channel=channel,
        deg_per_unit=fit.deg_per_unit,
        axis=fit.axis.tolist(),
        table=fit.table,
        latency_ms=float(np.median(latencies)) if latencies else float("nan"),
        settle_ms=float(np.percentile(settles, 95)) if settles else float("nan"),
        info={
            "date": date.today().isoformat(),
            "session": session.path.name,
            "program": program,
            "residual_rms_deg": fit.residual_rms,
            "off_axis_rms_deg": fit.off_axis_rms,
            "hysteresis_deg": fit.hysteresis,
            "min_inliers": int(min(inliers)),
        },
    )
    save_servo_model(args.out, model)
    print(f"\nZapisano {args.out} [{name}]")


if __name__ == "__main__":
    main()
