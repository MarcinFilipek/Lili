#!/usr/bin/env python3
"""
Użycie:
    python summarize_session.py logs/agent_YYYYMMDD_HHMMSS.csv
    python summarize_session.py logs/  # ostatni plik w katalogu
"""

import sys
import csv
import math
from pathlib import Path
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "step":  int(r["step"]),
                "eb":    float(r["error_brightness"]),
                "ee":    float(r.get("error_edges", 0.0)),
                "avg":   float(r["avg_error"]),
                "frust": float(r["frustration"]),
                "pa":    float(r["pan_actual"]),
                "ta":    float(r["tilt_actual"]),
                "pe":    float(r["pan_estimated"]),
                "te":    float(r["tilt_estimated"]),
                "sl":    int(r["sleeping"]),
                "sc":    int(r["sleep_count"]),
            })
    return rows


def stats(vals):
    if not vals:
        return {}
    n = len(vals)
    mu = sum(vals) / n
    sigma = math.sqrt(sum((x - mu) ** 2 for x in vals) / n)
    s = sorted(vals)
    p = lambda q: s[int(q * (n - 1))]
    return {"n": n, "mean": mu, "std": sigma,
            "p10": p(.1), "p50": p(.5), "p90": p(.9), "max": s[-1]}


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}"


def summarize(path):
    rows = load(path)
    total = len(rows)
    wake  = [r for r in rows if r["sl"] == 0]
    n_sleeps = rows[-1]["sc"]

    # Epizody
    episodes = defaultdict(list)
    for r in wake:
        episodes[r["sc"]].append(r)

    ep_lines = []
    for ep_id in sorted(episodes):
        ep  = episodes[ep_id]
        eb  = [r["eb"] for r in ep]
        ee  = [r["ee"] for r in ep]
        s   = stats(eb)
        se  = stats(ee)
        pct = sum(1 for v in eb if v < 0.05) / len(eb) * 100
        ep_lines.append(
            f"  ep{ep_id+1:>2d} ({len(ep):>5d} kroków): "
            f"br={fmt(s['mean'])} ed={fmt(se['mean'])}  "
            f"p50={fmt(s['p50'])}  p90={fmt(s['p90'])}  "
            f"<0.05: {pct:.0f}%"
        )

    # Ogólne brightness
    all_eb  = [r["eb"] for r in wake]
    all_ee  = [r["ee"] for r in wake]
    ov_b    = stats(all_eb)
    ov_e    = stats(all_ee)
    cut     = max(1, len(wake) // 5)
    tb_early = sum(r["eb"] for r in wake[:cut]) / cut
    tb_late  = sum(r["eb"] for r in wake[-cut:]) / cut
    te_early = sum(r["ee"] for r in wake[:cut]) / cut
    te_late  = sum(r["ee"] for r in wake[-cut:]) / cut
    arrow_b  = "↓" if tb_late < tb_early * 0.8 else \
               "↑" if tb_late > tb_early * 1.2 else "→"
    arrow_e  = "↓" if te_late < te_early * 0.8 else \
               "↑" if te_late > te_early * 1.2 else "→"

    # Dryft
    dp = stats([abs(r["pa"] - r["pe"]) for r in wake])
    dt = stats([abs(r["ta"] - r["te"]) for r in wake])

    pan_vals  = [r["pa"] for r in wake]
    tilt_vals = [r["ta"] for r in wake]

    dur = total * 0.7 / 60

    lines = [
        f"=== SESSION SUMMARY  {Path(path).name} ===",
        f"Kroki: {total}  |  Sny: {n_sleeps}  |  Czas: ~{dur:.1f} min",
        "",
        "── Błąd brightness ──",
        f"  mean={fmt(ov_b['mean'])}  p50={fmt(ov_b['p50'])}  "
        f"p90={fmt(ov_b['p90'])}  max={fmt(ov_b['max'], 3)}",
        f"  trend (pierwsze 20% vs ostatnie 20%): "
        f"{fmt(tb_early)} → {fmt(tb_late)}  {arrow_b}",
        "",
        "── Błąd edges ──",
        f"  mean={fmt(ov_e['mean'])}  p50={fmt(ov_e['p50'])}  "
        f"p90={fmt(ov_e['p90'])}  max={fmt(ov_e['max'], 3)}",
        f"  trend (pierwsze 20% vs ostatnie 20%): "
        f"{fmt(te_early)} → {fmt(te_late)}  {arrow_e}",
        "",
        "── Per epizod ──",
    ] + ep_lines + [
        "",
        "── Dryft pozycji (|estimated − actual|) ──",
        f"  pan:  mean={fmt(dp['mean'], 2)}°  p90={fmt(dp['p90'], 2)}°  max={fmt(dp['max'], 1)}°",
        f"  tilt: mean={fmt(dt['mean'], 2)}°  p90={fmt(dt['p90'], 2)}°  max={fmt(dt['max'], 1)}°",
        "",
        "── Zakres eksploracji ──",
        f"  pan:  [{min(pan_vals):.0f}°, {max(pan_vals):.0f}°]",
        f"  tilt: [{min(tilt_vals):.0f}°, {max(tilt_vals):.0f}°]",
        "===",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Użycie: python summarize_session.py <plik.csv lub katalog>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        csvs = sorted(target.glob("agent_*.csv"))
        if not csvs:
            print(f"Brak plików agent_*.csv w {target}")
            sys.exit(1)
        target = csvs[-1]
        print(f"(ostatni plik: {target.name})\n")

    print(summarize(target))


if __name__ == "__main__":
    main()