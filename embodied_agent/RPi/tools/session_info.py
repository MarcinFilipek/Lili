"""Podsumowanie sesji: klatki, FPS, zgubione klatki, zdarzenia. Bez argumentu bierze najnowszą sesję."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from lili.record.reader import Session, list_sessions

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, nargs="?")
    args = ap.parse_args()

    path = args.session
    if path is None:
        sessions = list_sessions(DATA_DIR)
        if not sessions:
            raise SystemExit(f"Brak sesji w {DATA_DIR}")
        path = sessions[-1]

    session = Session(path)
    meta = session.meta
    print(f"{session.path.name}  (kod {meta.get('code_version')}, schemat v{meta['schema_version']})")
    print(f"program: {meta.get('program')}")

    t, _ = session.frame_times()
    print(f"klatki: {len(t)}, czas: {meta['duration_s']:.2f} s")
    if len(t) > 1:
        dt_ms = np.diff(t) / 1e6
        period = np.median(dt_ms)
        dropped = int(np.sum(np.round(dt_ms / period) - 1))
        print(f"FPS: {1000 / period:.1f}, odstęp maks.: {dt_ms.max():.1f} ms, zgubione klatki: ~{dropped}")

    events = session.events()
    print("zdarzenia:", dict(Counter(e["type"] for e in events)))

    # Czas od wysłania komendy do ACK obu serw, czyli opóźnienie samego UART
    cmd_t = [e["t"] for e in events if e["type"] == "cmd"]
    stm_t = [e["t"] for e in events if e["type"] == "stm"][1::2]
    if cmd_t and len(cmd_t) == len(stm_t):
        latency_ms = (np.array(stm_t) - np.array(cmd_t)) / 1e6
        print(f"UART komenda -> ACK (pan+tilt): śr. {latency_ms.mean():.1f} ms, maks. {latency_ms.max():.1f} ms")


if __name__ == "__main__":
    main()
