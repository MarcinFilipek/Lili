"""Nagranie sesji ruch-stop-patrz (peering pan) na RPi. Kamera nagrywa ciągle przez całą sesję."""

import argparse
import subprocess
from pathlib import Path

from lili.hw.camera import Camera
from lili.hw.stm_link import CHANNEL_PAN, CHANNEL_TILT, StmLink
from lili.motor.programs import peering
from lili.motor.runner import FrameRecorder, run_program
from lili.record.writer import SessionWriter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def code_version():
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="krótki opis sceny, trafia do nazwy katalogu")
    ap.add_argument("--amplitude", type=float, default=10.0, help="wychylenie pan w stopniach")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--center", type=float, default=0.0)
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--settle", type=float, default=0.5, help="czas na dojazd serw [s]")
    ap.add_argument("--look", type=float, default=0.5, help="czas patrzenia po ustaleniu [s]")
    ap.add_argument("--port", default="/dev/ttyAMA0")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    program = {
        "name": "peering",
        "amplitude": args.amplitude,
        "cycles": args.cycles,
        "center": args.center,
        "tilt": args.tilt,
        "settle_s": args.settle,
        "look_s": args.look,
    }
    steps = peering(args.amplitude, args.cycles, args.center, args.tilt, args.settle, args.look)

    with Camera() as cam, StmLink(args.port) as link:
        # Pozycja startowa przed blokadą ekspozycji, żeby automatyka ustaliła się na tej scenie
        link.set_position(CHANNEL_PAN, steps[0].pan)
        link.set_position(CHANNEL_TILT, steps[0].tilt)
        exposure = cam.lock_exposure()

        meta = {
            "code_version": code_version(),
            "camera": {**cam.info(), "controls": exposure},
            "program": program,
        }
        print(f"Nagrywanie: {len(steps)} kroków")
        with SessionWriter(args.out, args.label, meta) as writer, FrameRecorder(cam, writer):
            run_program(steps, link, writer)

    m = writer.meta
    print(f"{writer.path}\n{m['frames']} klatek, {m['duration_s']:.1f} s, {m['events']} zdarzeń")


if __name__ == "__main__":
    main()
