"""Czas kroków percepcji dla pary klatek w 320x240 i 160x120 (do decyzji o rozdzielczości na RPi)."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lili.body.camera_model import CameraModel
from lili.body.efference import Efference
from lili.perception.derotation import rectification
from lili.perception.flow import dense_flow, forward_backward_error
from lili.perception.parallax import parallax, rho_sigma

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


def timed(fn, repeats):
    fn()  # rozgrzewka
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    return (time.perf_counter() - start) / repeats * 1000, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--repeats", type=int, default=20)
    args = ap.parse_args()

    full = CameraModel.load(CALIB_DIR / "camera_320x240.json")
    motion = Efference.from_calibration(CALIB_DIR).motion((0, 0), (10, 0))
    rng = np.random.default_rng(0)
    print(f"OpenCV {cv2.__version__}, wątki {cv2.getNumThreads()}, powtórzenia {args.repeats}")

    for width, height in [(320, 240), (160, 120)]:
        camera = full.scaled(width, height)
        noise = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 1.5)
        a = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        b = np.roll(a, 3, axis=1)

        t_maps, rect = timed(lambda: rectification(camera, motion.R), max(1, args.repeats // 5))
        t_remap, b2 = timed(lambda: rect.apply(b), args.repeats)
        t_flow, flow = timed(lambda: dense_flow(a, b2), args.repeats)
        back = dense_flow(b2, a)
        t_par, _ = timed(
            lambda: rho_sigma(parallax(flow, camera.K, motion.c), fb_error=forward_backward_error(flow, back)),
            args.repeats,
        )
        total = 2 * t_remap + 2 * t_flow + t_par
        print(
            f"{width}x{height}: mapy {t_maps:.1f} ms (raz na kąt), remap {t_remap:.2f} ms, "
            f"DIS {t_flow:.1f} ms, paralaksa+pewność {t_par:.1f} ms -> para A/B z przepływem wstecz {total:.1f} ms"
        )


if __name__ == "__main__":
    main()
