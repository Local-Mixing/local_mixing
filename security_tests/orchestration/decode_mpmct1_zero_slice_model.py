#!/usr/bin/env python3
"""Decode and independently verify a Kissat zero-slice preimage model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_initial_model(path: Path, wires: int) -> list[bool]:
    values: list[bool | None] = [None] * wires
    satisfiable = False
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
                if variable <= wires:
                    value = literal > 0
                    previous = values[variable - 1]
                    if previous is not None and previous != value:
                        raise ValueError(
                            f"solver model contradicts itself for variable {variable}"
                        )
                    values[variable - 1] = value
    if not satisfiable:
        raise ValueError("solver output does not contain s SATISFIABLE")
    missing = [index + 1 for index, value in enumerate(values) if value is None]
    if missing:
        raise ValueError(f"solver model omits initial variables: {missing[:16]}")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--solver-output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--target-hex", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.final.open("r", encoding="utf-8") as source:
        header = source.readline().split()
    if len(header) != 3 or header[0] != "mpmct1":
        raise ValueError("invalid final-circuit header")
    total_wires = int(header[1])
    if total_wires != 4 * args.n:
        raise ValueError(f"expected 4n={4 * args.n} wires, got {total_wires}")

    model = read_initial_model(args.solver_output, total_wires)
    full_input = sum(int(value) << wire for wire, value in enumerate(model))
    logical_mask = (1 << args.n) - 1
    logical_input = full_input & logical_mask
    nonzero_padding_bits = (full_input >> args.n).bit_count()
    target = int(args.target_hex, 16)
    if target & ~logical_mask:
        raise ValueError("target is wider than n")

    full_output, parsed_wires, gate_count = evaluate_mpmct1(args.final, full_input)
    logical_output = (full_output >> args.n) & logical_mask
    verified = nonzero_padding_bits == 0 and logical_output == target
    record: dict[str, object] = {
        "sat": True,
        "verified": verified,
        "n": args.n,
        "total_wires": parsed_wires,
        "final_gate_count": gate_count,
        "logical_input_hex": padded_hex(logical_input, args.n),
        "logical_output_hex": padded_hex(logical_output, args.n),
        "target_logical_output_hex": padded_hex(target, args.n),
        "nonzero_padding_input_bits": nonzero_padding_bits,
        "input_convention": f"x on wires 0..{args.n - 1}; zero on wires {args.n}..{total_wires - 1}",
        "output_convention": f"C(x) on wires {args.n}..{2 * args.n - 1}",
    }
    write_private_json(args.out, record)
    print(json.dumps({"decoded_report": str(args.out), "verified": verified}, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
