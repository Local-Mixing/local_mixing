#!/usr/bin/env python3
"""Decode a collision CNF model and verify H(x1) = H(x2) on the mpmct1 tape."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_model_bits(path: Path, start: int, count: int) -> list[bool]:
    """Read DIMACS variables [start, start+count) (1-based inclusive start)."""
    values: list[bool | None] = [None] * count
    satisfiable = False
    end = start + count - 1
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if line.strip() == "s SATISFIABLE":
                satisfiable = True
            if not line.startswith("v"):
                continue
            for token in line[1:].split():
                literal = int(token)
                if literal == 0:
                    continue
                variable = abs(literal)
                if start <= variable <= end:
                    idx = variable - start
                    value = literal > 0
                    previous = values[idx]
                    if previous is not None and previous != value:
                        raise ValueError(f"contradictory model for var {variable}")
                    values[idx] = value
    if not satisfiable:
        raise ValueError("solver output does not contain s SATISFIABLE")
    missing = [start + i for i, value in enumerate(values) if value is None]
    if missing:
        raise ValueError(f"solver model omits input variables: {missing[:16]}")
    return [bool(value) for value in values]


def evaluate_mpmct1(path: Path, state: int) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8") as source:
        header = source.readline().split()
        if len(header) != 3 or header[0] != "mpmct1":
            raise ValueError("invalid mpmct1 header")
        wires = int(header[1])
        expected_gates = int(header[2])
        gates = 0
        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            fields = [int(value) for value in line.split()]
            if len(fields) < 3:
                raise ValueError(f"short gate at line {line_number}")
            target, complemented, width = fields[:3]
            if (
                complemented not in (0, 1)
                or width < 0
                or len(fields) != 3 + 2 * width
                or not 0 <= target < wires
            ):
                raise ValueError(f"malformed gate at line {line_number}")
            fires = True
            seen: set[int] = set()
            for offset in range(width):
                control = fields[3 + 2 * offset]
                polarity = fields[4 + 2 * offset]
                if (
                    not 0 <= control < wires
                    or polarity not in (0, 1)
                    or control == target
                    or control in seen
                ):
                    raise ValueError(f"invalid control at line {line_number}")
                seen.add(control)
                fires &= ((state >> control) & 1) == polarity
            fires ^= bool(complemented)
            if fires:
                state ^= 1 << target
            gates += 1
        if gates != expected_gates:
            raise ValueError(
                f"gate count mismatch: header={expected_gates} parsed={gates}"
            )
    return state, wires, gates


def bits_to_int(bits: list[bool]) -> int:
    value = 0
    for index, bit in enumerate(bits):
        if bit:
            value |= 1 << index
    return value


def padded_hex(value: int, bits: int) -> str:
    return "0x" + format(value, f"0{(bits + 3) // 4}x")


def write_private_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(record, destination, indent=2, sort_keys=True)
            destination.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def mpmct_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as source:
        header = source.readline().split()
    if len(header) != 3 or header[0] != "mpmct1":
        raise ValueError("invalid mpmct1 header")
    return int(header[1]), int(header[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--solver-output", type=Path, required=True)
    parser.add_argument("--in-bits", type=int, default=64)
    parser.add_argument("--pad", type=int, default=32)
    parser.add_argument("--out-bits", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wires, gates = mpmct_header(args.circuit)
    width = args.in_bits + args.pad
    if wires < width:
        raise SystemExit(f"circuit width {wires} < in_bits+pad {width}")
    vars_per_copy = wires + gates

    x1_bits = read_model_bits(args.solver_output, 1, args.in_bits)
    x2_bits = read_model_bits(
        args.solver_output, vars_per_copy + 1, args.in_bits
    )
    x1 = bits_to_int(x1_bits)
    x2 = bits_to_int(x2_bits)
    if x1 == x2:
        raise SystemExit("model inputs are equal; not a collision")

    out1, _, _ = evaluate_mpmct1(args.circuit, x1)
    out2, _, _ = evaluate_mpmct1(args.circuit, x2)
    mask = (1 << args.out_bits) - 1
    d1 = out1 & mask
    d2 = out2 & mask
    ok = d1 == d2
    record = {
        "circuit": str(args.circuit),
        "in_bits": args.in_bits,
        "pad": args.pad,
        "out_bits": args.out_bits,
        "x1_hex": padded_hex(x1, args.in_bits),
        "x2_hex": padded_hex(x2, args.in_bits),
        "digest1_hex": padded_hex(d1, args.out_bits),
        "digest2_hex": padded_hex(d2, args.out_bits),
        "verified": ok,
    }
    write_private_json(args.out, record)
    if not ok:
        raise SystemExit("digest mismatch after independent evaluation")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
