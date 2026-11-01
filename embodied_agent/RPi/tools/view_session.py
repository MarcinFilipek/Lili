"""Podgląd percepcji dla pary postojów sesji: krawędzie zasłaniania i mapa 1/Z.

    python tools/view_session.py                     # najnowsza sesja, postoje 0 i 1
    python tools/view_session.py data/sessions/X --pair 0 2
    python tools/view_session.py --save wynik.png    # bez okna (RPi przez SSH)

Mozaika: A z krawędziami | odległość tam, gdzie pewna / odległość wszędzie | B po derotacji.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from lili.body.camera_model import CameraModel
from lili.body.efference import Efference
from lili.pipeline import motion, process_pair, views
from lili.record.reader import Session, list_sessions

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sessions"
CALIB_DIR = ROOT / "calib"
NEAR_M, FAR_M = 0.25, 3.0  # zakres kolorów: czerwony blisko, niebieski daleko


def load_camera(width, height):
    path = CALIB_DIR / f"camera_{width}x{height}.json"
    if path.exists():
        return CameraModel.load(path)
    return CameraModel.load(CALIB_DIR / "camera_320x240.json").scaled(width, height)


def depth_colors(rho, mask):
    """Kolor wg odległości w skali log (TURBO: czerwony blisko), czarny poza mask."""
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.clip(1 / rho, NEAR_M, FAR_M)
    level = np.log(FAR_M / distance) / np.log(FAR_M / NEAR_M)
    colors = cv2.applyColorMap(np.nan_to_num(level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colors[~(mask & np.isfinite(rho) & (rho > 0))] = 0
    return colors


def label(image, text):
    cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def mosaic(result, scale=2):
    gray = lambda img: cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # noqa: E731
    with_edges = gray(result.a)
    with_edges[result.edges] = (0, 0, 255)
    tiles = [
        label(with_edges, "A + krawedzie"),
        label(depth_colors(result.rho, result.reliable), f"odleglosc pewna ({NEAR_M}-{FAR_M} m)"),
        label(depth_colors(result.rho, result.valid), "odleglosc wszedzie"),
        label(gray(result.b), "B po derotacji"),
    ]
    tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for t in tiles]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, nargs="?")
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1], metavar=("A", "B"), help="numery postojów")
    ap.add_argument("--save", type=Path, help="zapisz mozaikę do pliku zamiast pokazywać okno")
    args = ap.parse_args()

    path = args.session
    if path is None:
        sessions = list_sessions(DATA_DIR)
        if not sessions:
            raise SystemExit(f"Brak sesji w {DATA_DIR}")
        path = sessions[-1]
    session = Session(path)
    all_views = views(session)
    print(f"{session.path.name}: {len(all_views)} postojów, program {session.meta.get('program', {}).get('name')}")
    for v in all_views:
        print(f"  {v.step}: pan {v.pan:g}, tilt {v.tilt:g}, podstawa {np.round(v.offset * 100, 1)} cm, {v.frames} klatek")

    a, b = (all_views[i] for i in args.pair)
    camera = load_camera(a.image.shape[1], a.image.shape[0])
    pose = motion(a, b, Efference.from_calibration(CALIB_DIR))
    manual = session.meta.get("program", {}).get("name") == "manual"
    result = process_pair(a.image, b.image, camera, pose, refine_translation=manual)

    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(pose.R)[0]))
    print(f"para {a.step} -> {b.step}: obrót {angle:.2f} deg, T = {np.round(result.T * 100, 2)} cm"
          + (" (kierunek z obrazu)" if manual else ""))
    reliable = result.reliable & result.valid
    print(f"pewne piksele: {reliable.sum() / result.valid.sum():.0%} wspólnego pola, krawędzie: {result.edges.sum()} px")
    if reliable.any():
        distances = 1 / result.rho[reliable]
        print("odległości pewnych pikseli [m], percentyle 5/50/95: "
              + " / ".join(f"{d:.2f}" for d in np.percentile(distances, [5, 50, 95])))

    image = mosaic(result)
    if args.save:
        cv2.imwrite(str(args.save), image)
        print(f"zapisano {args.save}")
    else:
        cv2.imshow(session.path.name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
