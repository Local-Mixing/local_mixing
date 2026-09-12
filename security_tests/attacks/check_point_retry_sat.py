#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
VAL = {c: i for i, c in enumerate(CHARS)}
BASE = len(CHARS)


def parse_circuit(path: Path) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    gate: list[int] = []
    overflow = 0
    for ch in path.read_text():
        if ch == ";":
            if gate:
                if len(gate) != 3:
                    raise ValueError(f"bad gate arity {len(gate)} near gate {len(gates)}")
                gates.append((gate[0], gate[1], gate[2]))
                gate.clear()
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch in "\n\r\t ":
            continue
        else:
            if ch not in VAL:
                raise ValueError(f"bad circuit char {ch!r}")
            gate.append(VAL[ch] + BASE * overflow)
            overflow = 0
    if gate:
        raise ValueError(f"unterminated gate at end: {gate}")
    return gates


def evaluate(gates: list[tuple[int, int, int]], state: int) -> int:
    for target, c1, c2 in gates:
        if ((state >> c1) & 1) | (1 ^ ((state >> c2) & 1)):
            state ^= 1 << target
    return state


def extract_model(log: Path, bits: int) -> int | None:
    text = log.read_text(errors="ignore")
    if "s SATISFIABLE" not in text:
        return None
    seen = [None] * bits
    for line in text.splitlines():
        if not line.startswith("v "):
            continue
        for part in line.split()[1:]:
            lit = int(part)
            if lit == 0:
                break
            var = abs(lit)
            if 1 <= var <= bits:
                seen[var - 1] = lit > 0
    if any(x is None for x in seen):
        raise ValueError(f"{log}: SAT model missing one of variables 1..{bits}")
    value = 0
    for i, bit in enumerate(seen):
        if bit:
            value |= 1 << i
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=64)
    parser.add_argument("--flip", type=int, default=63)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    gates = parse_circuit(args.circuit)
    mask = (1 << args.bits) - 1
    found = False
    for log in args.logs:
        model = extract_model(log, args.bits)
        if model is None:
            continue
        found = True
        out = evaluate(gates, model)
        lower = out & mask
        expected = model ^ (1 << args.flip)
        ok = lower == expected
        print(f"log {log}")
        print(f"x 0x{model:0{args.bits // 4}x}")
        print(f"lower_output 0x{lower:0{args.bits // 4}x}")
        print(f"expected     0x{expected:0{args.bits // 4}x}")
        print(f"verified {str(ok).lower()}")
        if ok:
            return 0
    if not found:
        print("no SAT model found")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
