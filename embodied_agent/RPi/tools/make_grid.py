"""Siatka prostych linii na A4 (poziomo) do wzrokowego sprawdzenia dystorsji i jej korekcji."""

import argparse
from pathlib import Path

from make_checkerboard import CALIB_DIR, PAGE_H_MM, PAGE_W_MM, PT_PER_MM, write_pdf

MARGIN_MM = 6.0


def content_stream(step_mm, line_mm):
    def mm(v):
        return f"{v * PT_PER_MM:.3f}"

    # Pełna liczba kratek, siatka wyśrodkowana na stronie
    nx = int((PAGE_W_MM - 2 * MARGIN_MM) // step_mm)
    ny = int((PAGE_H_MM - 2 * MARGIN_MM) // step_mm)
    x0, y0 = (PAGE_W_MM - nx * step_mm) / 2, (PAGE_H_MM - ny * step_mm) / 2
    x1, y1 = x0 + nx * step_mm, y0 + ny * step_mm

    ops = ["0 0 0 RG", f"{mm(line_mm)} w", "2 J"]  # kwadratowe końce linii, żeby rogi siatki były pełne
    for i in range(nx + 1):
        x = x0 + i * step_mm
        ops.append(f"{mm(x)} {mm(y0)} m {mm(x)} {mm(y1)} l S")
    for j in range(ny + 1):
        y = y0 + j * step_mm
        ops.append(f"{mm(x0)} {mm(y)} m {mm(x1)} {mm(y)} l S")
    return "\n".join(ops).encode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=20.0, help="odstęp linii [mm]")
    ap.add_argument("--line", type=float, default=1.5, help="grubość linii [mm]")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or CALIB_DIR / f"grid_{args.step:g}mm.pdf"
    write_pdf(out, content_stream(args.step, args.line))
    print(out)


if __name__ == "__main__":
    main()
