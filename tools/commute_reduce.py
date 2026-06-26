#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
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
                gates.append(tuple(gate))
                gate.clear()
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch in "\n\r\t ":
            continue
        else:
            if ch not in VAL:
                raise ValueError(f"unknown circuit character {ch!r}")
            gate.append(VAL[ch] + BASE * overflow)
            overflow = 0
    if gate:
        raise ValueError(f"unterminated gate at end of file: {gate}")
    return gates


def encode_wire(wire: int) -> str:
    q, r = divmod(wire, BASE)
    return "~" * q + CHARS[r]


def write_circuit(path: Path, gates: list[tuple[int, int, int]]) -> None:
    with path.open("w") as f:
        for target, c1, c2 in gates:
            f.write(encode_wire(target))
            f.write(encode_wire(c1))
            f.write(encode_wire(c2))
            f.write(";")


@functools.cache
def commute(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    if b < a:
        return commute(b, a)
    at, ac1, ac2 = a
    bt, bc1, bc2 = b
    if at == bt:
        return True
    if at != bc1 and at != bc2 and bt != ac1 and bt != ac2:
        return True

    wires = sorted(set(a) | set(b))
    idx = {wire: i for i, wire in enumerate(wires)}
    for mask in range(1 << len(wires)):
        state = [(mask >> i) & 1 for i in range(len(wires))]
        left = state.copy()
        right = state.copy()
        apply_gate(left, a, idx)
        apply_gate(left, b, idx)
        apply_gate(right, b, idx)
        apply_gate(right, a, idx)
        if left != right:
            return False
    return True


def apply_gate(
    state: list[int], gate: tuple[int, int, int], idx: dict[int, int]
) -> None:
    target, c1, c2 = gate
    if state[idx[c1]] or not state[idx[c2]]:
        state[idx[target]] ^= 1


def reduce_once(gates: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], int, int]:
    stack: list[tuple[int, int, int]] = []
    cancelled = 0
    scanned = 0

    for gate in gates:
        j = len(stack) - 1
        while j >= 0 and commute(gate, stack[j]):
            scanned += 1
            if gate == stack[j]:
                del stack[j]
                cancelled += 2
                break
            j -= 1
        else:
            stack.append(gate)
            continue

        if j < 0:
            continue

    return stack, cancelled, scanned


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--passes", type=int, default=20)
    args = parser.parse_args()

    gates = parse_circuit(args.input)
    print(f"input_gates {len(gates)} wires {max(max(g) for g in gates) + 1}")

    for pass_idx in range(1, args.passes + 1):
        before = len(gates)
        gates, cancelled, scanned = reduce_once(gates)
        print(
            f"pass {pass_idx} before {before} after {len(gates)} "
            f"cancelled {cancelled} scanned {scanned}"
        )
        if cancelled == 0:
            break

    write_circuit(args.output, gates)
    print(f"output_gates {len(gates)} output {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
