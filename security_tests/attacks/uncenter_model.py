#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_hex_bits(text: str, count: int):
    if text.startswith(("0x", "0X")):
        text = text[2:]
    value = int(text or "0", 16)
    return [(value >> i) & 1 for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center-hex", required=True)
    parser.add_argument("--var-start", default=129, type=int)
    parser.add_argument("--var-count", default=128, type=int)
    args = parser.parse_args()

    center_bits = parse_hex_bits(args.center_hex, args.var_count)
    with args.model.open() as src, args.out.open("w") as dst:
        for line in src:
            if not line.startswith("v"):
                dst.write(line)
                continue
            out = ["v"]
            for raw in line.split()[1:]:
                lit = int(raw)
                if lit == 0:
                    out.append("0")
                    break
                var = abs(lit)
                idx = var - args.var_start
                if 0 <= idx < args.var_count and center_bits[idx]:
                    lit = -lit
                out.append(str(lit))
            dst.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
