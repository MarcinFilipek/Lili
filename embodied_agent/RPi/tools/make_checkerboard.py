"""Szachownica do kalibracji kamery jako PDF A4 (poziomo) w skali 1:1. Bez zależności poza stdlib."""

import argparse
from pathlib import Path

PAGE_W_MM, PAGE_H_MM = 297.0, 210.0
PT_PER_MM = 72 / 25.4

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


def content_stream(cols, rows, square_mm):
    """cols x rows narożników wewnętrznych -> (cols+1) x (rows+1) pól."""
    board_w, board_h = (cols + 1) * square_mm, (rows + 1) * square_mm
    if board_w > PAGE_W_MM - 20 or board_h > PAGE_H_MM - 35:
        raise SystemExit(f"Szachownica {board_w:.0f}x{board_h:.0f} mm nie mieści się na A4")

    x0 = (PAGE_W_MM - board_w) / 2
    y0 = PAGE_H_MM - 8 - board_h  # 8 mm od górnej krawędzi, PDF liczy y od dołu

    def mm(v):
        return f"{v * PT_PER_MM:.3f}"

    ops = ["0 0 0 rg"]
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x, y = x0 + c * square_mm, y0 + r * square_mm
                ops.append(f"{mm(x)} {mm(y)} {mm(square_mm)} {mm(square_mm)} re f")

    # Linia kontrolna 100 mm z kreskami na końcach
    lx, ly = x0, y0 - 16
    ops += [
        "0.5 w",
        f"{mm(lx)} {mm(ly)} m {mm(lx + 100)} {mm(ly)} l S",
        f"{mm(lx)} {mm(ly - 2)} m {mm(lx)} {mm(ly + 2)} l S",
        f"{mm(lx + 100)} {mm(ly - 2)} m {mm(lx + 100)} {mm(ly + 2)} l S",
    ]

    text = [
        (y0 - 7, f"Szachownica {cols}x{rows} naroznikow wewnetrznych, pole {square_mm:g} mm. "
                 "Drukuj w skali 100% (bez dopasowania do strony)."),
        (ly - 6, "Kontrola skali: ta linia musi miec dokladnie 100 mm. Zmierz tez pole linijka."),
    ]
    for y, line in text:
        ops.append(f"BT /F1 9 Tf {mm(x0)} {mm(y)} Td ({line}) Tj ET")
    return "\n".join(ops).encode("ascii")


def write_pdf(path, stream):
    w, h = PAGE_W_MM * PT_PER_MM, PAGE_H_MM * PT_PER_MM
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] "
        f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"

    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    Path(path).write_bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="narożniki wewnętrzne w poziomie")
    ap.add_argument("--rows", type=int, default=6, help="narożniki wewnętrzne w pionie")
    ap.add_argument("--square", type=float, default=25.0, help="bok pola [mm]")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or CALIB_DIR / f"checkerboard_{args.cols}x{args.rows}_{args.square:g}mm.pdf"
    write_pdf(out, content_stream(args.cols, args.rows, args.square))
    print(out)


if __name__ == "__main__":
    main()
