"""Kalibracja kamery z ujęć zebranych przez capture_calib.py. Działa na RPi i na Macu."""

import argparse
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from lili.body.camera_calibration import PATTERN, calibrate, find_corners

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "calib"
CALIB_DIR = ROOT / "calib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path, nargs="?", help="katalog z ujęciami, domyślnie najnowszy w data/calib")
    ap.add_argument("--square", type=float, default=25.0, help="zmierzony bok pola [mm]")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="numery ujęć do pominięcia")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    images_dir = args.images or sorted(p for p in DATA_DIR.iterdir() if p.is_dir())[-1]
    paths = [p for p in sorted(images_dir.glob("*.png")) if int(p.stem) not in args.drop]

    used, corner_sets, image_size = [], [], None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image_size = (image.shape[1], image.shape[0])
        corners = find_corners(image)
        if corners is None:
            print(f"{path.name}: nie znaleziono szachownicy, pomijam")
            continue
        used.append(path)
        corner_sets.append(corners)

    model, per_view = calibrate(corner_sets, image_size, PATTERN, args.square)

    median = np.median(per_view)
    print(f"\nUjęcia: {len(used)}, błąd reprojekcji RMS: {model.rms:.3f} px")
    for path, err in zip(used, per_view):
        flag = "  <- podejrzane, rozważ --drop" if err > 2 * median and err > 0.3 else ""
        print(f"  {path.stem}: {err:.3f} px{flag}")

    fov_h, fov_v = model.fov_deg()
    print(f"\nfx={model.fx:.1f} fy={model.fy:.1f} cx={model.cx:.1f} cy={model.cy:.1f} px")
    print(f"dist (k1, k2, p1, p2, k3) = {np.array2string(model.dist, precision=4)}")
    print(f"kąt widzenia: {fov_h:.1f}° x {fov_v:.1f}°")

    model.info = {
        "date": date.today().isoformat(),
        "images": str(images_dir.relative_to(ROOT) if images_dir.is_relative_to(ROOT) else images_dir),
        "views": len(used),
        "dropped": args.drop,
        "pattern": list(PATTERN),
        "square_mm": args.square,
        "per_view_rms": per_view,
    }
    out = args.out or CALIB_DIR / f"camera_{image_size[0]}x{image_size[1]}.json"
    model.save(out)
    print(f"\nZapisano {out}")


if __name__ == "__main__":
    main()
