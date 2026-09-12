#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def fmt_num(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{int(value):,}"
    except ValueError:
        return value


def fmt_float(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} results.csv [last_n=40]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(path.open()))
    print(f"rows: {len(rows)}")
    if not rows:
        return 0
    shown = rows[-last_n:]
    cols = [
        ("gates", "gate_count"),
        ("sr", "sr"),
        ("r", "r"),
        ("m", "m"),
        ("x", "x"),
        ("trial", "trial"),
        ("mixed_gates", "mixed_gates"),
        ("status", "status"),
        ("conflicts", "conflicts"),
        ("solve_s", "solve_seconds"),
        ("mix_s", "mix_seconds"),
        ("cnf_s", "cnf_seconds"),
    ]
    print("| " + " | ".join(label for label, _ in cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for row in shown:
        vals = []
        for label, key in cols:
            value = row.get(key, "")
            if key in {"mixed_gates", "conflicts"}:
                value = fmt_num(value)
            elif key.endswith("_seconds"):
                value = fmt_float(value)
            vals.append(value)
        print("| " + " | ".join(vals) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
