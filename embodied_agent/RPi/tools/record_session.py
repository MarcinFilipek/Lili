"""Nagranie sesji ruch-stop-patrz na RPi. Kamera nagrywa ciągle przez całą sesję.

    python tools/record_session.py peering kubek --amplitude 10
    python tools/record_session.py sweep pan_kal --axis pan --limit 30 --step 5
    python tools/record_session.py manual stol_listwa --offsets 0 2 4
"""

import argparse
import subprocess
from pathlib import Path

from lili.hw.camera import Camera
from lili.hw.stm_link import CHANNEL_PAN, CHANNEL_TILT, StmLink
from lili.motor.programs import manual, peering, sweep
from lili.motor.runner import FrameRecorder, run_program
from lili.record.writer import SessionWriter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def code_version():
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--exclude=*"],  # sam hash, bez nazw tagów
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build_program(args):
    """(kroki, opis programu do meta.json)"""
    if args.program == "peering":
        params = dict(amplitude=args.amplitude, cycles=args.cycles, center=args.center, tilt=args.tilt)
        steps = peering(**params, settle_s=args.settle, look_s=args.look)
    elif args.program == "sweep":
        params = dict(axis=args.axis, limit=args.limit, step=args.step, fixed=args.fixed)
        steps = sweep(**params, settle_s=args.settle, look_s=args.look)
    else:
        params = dict(offsets_cm=args.offsets, direction=args.direction, pan=args.pan, tilt=args.tilt)
        steps = manual(**params, settle_s=args.settle, look_s=args.look)
    return steps, {"name": args.program, **params, "settle_s": args.settle, "look_s": args.look}


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("label", help="krótki opis sceny, trafia do nazwy katalogu")
    common.add_argument("--port", default="/dev/ttyAMA0")
    common.add_argument("--out", type=Path, default=DATA_DIR)
    common.add_argument("--look", type=float, default=0.5, help="czas patrzenia po ustaleniu [s]")

    ap = argparse.ArgumentParser()
    programs = ap.add_subparsers(dest="program", required=True)

    p = programs.add_parser("peering", parents=[common], help="kołysanie pan wokół punktu")
    p.add_argument("--amplitude", type=float, default=10.0, help="wychylenie pan w stopniach")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--center", type=float, default=0.0)
    p.add_argument("--tilt", type=float, default=0.0)
    p.add_argument("--settle", type=float, default=0.5, help="czas na dojazd serw [s]")

    s = programs.add_parser("sweep", parents=[common], help="kalibracja serwa: przejazd osi tam i z powrotem")
    s.add_argument("--axis", choices=["pan", "tilt"], required=True)
    s.add_argument("--limit", type=float, default=30.0)
    s.add_argument("--step", type=float, default=5.0)
    s.add_argument("--fixed", type=float, default=0.0, help="pozycja drugiej osi")
    s.add_argument("--settle", type=float, default=1.5, help="czas na dojazd serw [s]")

    m = programs.add_parser("manual", parents=[common], help="ręczne przesuwanie robota wzdłuż listwy, serwa stoją")
    m.add_argument("--offsets", type=float, nargs="+", default=[0.0, 2.0, 4.0], help="położenia wzdłuż listwy [cm]")
    m.add_argument("--direction", choices=["prawo", "lewo"], default="prawo", help="kierunek przesuwu względem kamery")
    m.add_argument("--pan", type=float, default=0.0)
    m.add_argument("--tilt", type=float, default=0.0)
    m.add_argument("--settle", type=float, default=1.0, help="czas na wygaśnięcie drgań po zabraniu rąk [s]")

    args = ap.parse_args()
    steps, program = build_program(args)

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
        duration = sum(st.settle_s + st.look_s for st in steps)
        print(f"Nagrywanie: {len(steps)} kroków, ok. {duration:.0f} s")
        with SessionWriter(args.out, args.label, meta) as writer, FrameRecorder(cam, writer):
            run_program(steps, link, writer)

    m = writer.meta
    print(f"{writer.path}\n{m['frames']} klatek, {m['duration_s']:.1f} s, {m['events']} zdarzeń")


if __name__ == "__main__":
    main()
