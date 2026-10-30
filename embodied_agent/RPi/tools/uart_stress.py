"""Stres-test UART z STM32: SET/GET na obu kanałach, podsumowanie OK/ERR."""

import argparse

from lili.stm_link import CHANNEL_PAN, CHANNEL_TILT, ProtocolError, StmLink

TARGETS = {CHANNEL_PAN: -40.0, CHANNEL_TILT: -30.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--iterations", type=int, default=100)
    ap.add_argument("--port", default="/dev/ttyAMA0")
    args = ap.parse_args()

    stats = {f"{op} ch{ch}": {"ok": 0, "err": 0} for op in ("SET", "GET") for ch in TARGETS}

    def record(label, got, expected):
        stats[label]["ok" if got == expected else "err"] += 1

    with StmLink(args.port) as link:
        for i in range(args.iterations):
            print(f"\rIteracja {i + 1}/{args.iterations}", end="", flush=True)
            for op in ("SET", "GET"):
                for ch, target in TARGETS.items():
                    label = f"{op} ch{ch}"
                    try:
                        got = link.set_position(ch, target) if op == "SET" else link.get_position(ch)
                        record(label, got, target)
                    except ProtocolError as e:
                        print(f"\n{label}: {e}")
                        stats[label]["err"] += 1

    print(f"\n\n{'Frame':<10} {'OK':>6} {'ERR':>6} {'OK%':>7}")
    for label, s in stats.items():
        total = s["ok"] + s["err"]
        print(f"{label:<10} {s['ok']:>6} {s['err']:>6} {100.0 * s['ok'] / total:>6.1f}%")


if __name__ == "__main__":
    main()
