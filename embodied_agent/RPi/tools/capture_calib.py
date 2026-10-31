"""Zbieranie ujęć szachownicy do kalibracji kamery (RPi, monitor niepotrzebny).

Trzymaj tablicę przed kamerą i zmieniaj ujęcie: blisko i daleko, w rogach i na brzegach kadru,
pochylona w różne strony. Skrypt zapisuje tylko ujęcia ostre (tablica nieruchoma przez
dwie klatki) i wyraźnie inne od już zebranych.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lili.body.camera_calibration import PATTERN, find_corners, view_signature
from lili.hw.camera import Camera

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "calib"

STILL_PX = 0.5  # maks. średnie przesunięcie narożników między klatkami, żeby uznać tablicę za nieruchomą
MIN_DIFFERENCE = 0.1  # min. odległość sygnatury ujęcia od już zebranych


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or DATA_DIR / f"{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    print(f"Zapis do {out}. Ruszaj tablicą, Ctrl+C kończy wcześniej.")

    signatures = []
    previous = None
    with Camera() as cam:
        try:
            while len(signatures) < args.count:
                image, _, _ = cam.capture()
                corners = find_corners(image, refine=False)
                still = (
                    corners is not None
                    and previous is not None
                    and np.mean(np.linalg.norm(corners - previous, axis=1)) < STILL_PX
                )
                previous = corners
                if not still:
                    continue

                signature = view_signature(corners, (cam.width, cam.height), PATTERN)
                if signatures and min(np.linalg.norm(signature - s) for s in signatures) < MIN_DIFFERENCE:
                    continue

                cv2.imwrite(str(out / f"{len(signatures):03d}.png"), image)
                signatures.append(signature)
                cx, cy, size = signature[:3]
                print(
                    f"{len(signatures)}/{args.count}  środek ({cx:.2f}, {cy:.2f}) kadru, "
                    f"szerokość {size:.2f} kadru"
                )
        except KeyboardInterrupt:
            pass

    print(f"Zebrano {len(signatures)} ujęć. Dalej: python tools/calibrate_camera.py {out}")


if __name__ == "__main__":
    main()
