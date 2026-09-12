#!/usr/bin/env python3
"""Create and independently verify one logical TDP challenge pair.

The public final circuit has 4n wires.  A logical input x occupies wires
0..n-1 and every other input wire is zero.  The logical output is read from
wires n..2n-1 and must equal the original n-wire g57 circuit C(x).

The script streams the potentially multi-million-gate mpmct1 file, evaluates
the source through a separate g57 parser, checks the exact selected pair, and
writes a private answer record and a safe-to-share public challenge record.
Secret input values are never printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path


WIRE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
WIRE_INDEX = {char: index for index, char in enumerate(WIRE_CHARS)}
BASE = len(WIRE_CHARS)


def parse_g57(path: Path) -> list[tuple[int, int, int]]:
    text = path.read_text(encoding="utf-8")
    gates: list[tuple[int, int, int]] = []
    for raw_gate in text.split(";"):
        raw_gate = raw_gate.strip()
        if not raw_gate:
            continue
        wires: list[int] = []
        overflow = 0
        for char in raw_gate:
            if char.isspace():
                continue
            if char == "~":
                overflow += 1
                continue
            if char not in WIRE_INDEX:
                raise ValueError(f"unknown g57 wire character {char!r}")
            wires.append(overflow * BASE + WIRE_INDEX[char])
            overflow = 0
        if overflow:
            raise ValueError("dangling g57 overflow marker")
        if len(wires) != 3:
            raise ValueError(f"g57 gate does not have three wires: {raw_gate!r}")
        target, control_x, control_y = wires
        if len({target, control_x, control_y}) != 3:
            raise ValueError(f"g57 gate does not use three distinct wires: {wires}")
        gates.append((target, control_x, control_y))
    return gates


def eval_g57(gates: list[tuple[int, int, int]], state: int) -> int:
    for target, control_x, control_y in gates:
        x = (state >> control_x) & 1
        y = (state >> control_y) & 1
        if x or not y:
            state ^= 1 << target
    return state


def eval_mpmct(path: Path, state: int) -> tuple[int, int, int, int]:
    max_seen = -1
    with path.open("r", encoding="utf-8") as source:
        header = source.readline().split()
        if len(header) != 3 or header[0] != "mpmct1":
            raise ValueError("missing or invalid mpmct1 header")
        wires = int(header[1])
        expected_gates = int(header[2])
        gate_count = 0
        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue
            fields = [int(value) for value in line.split()]
            if len(fields) < 3:
                raise ValueError(f"short gate at line {line_number}")
            target, comp, width = fields[:3]
            if comp not in (0, 1) or width < 0 or len(fields) != 3 + 2 * width:
                raise ValueError(f"malformed gate at line {line_number}")
            if not 0 <= target < wires:
                raise ValueError(f"target outside circuit at line {line_number}")
            fires = True
            seen_controls: set[int] = set()
            for offset in range(width):
                control = fields[3 + 2 * offset]
                polarity = fields[4 + 2 * offset]
                if not 0 <= control < wires or polarity not in (0, 1):
                    raise ValueError(f"invalid control at line {line_number}")
                if control == target or control in seen_controls:
                    raise ValueError(f"non-reversible gate shape at line {line_number}")
                seen_controls.add(control)
                fires &= ((state >> control) & 1) == polarity
                max_seen = max(max_seen, control)
            fires ^= bool(comp)
            if fires:
                state ^= 1 << target
            max_seen = max(max_seen, target)
            gate_count += 1
        if gate_count != expected_gates:
            raise ValueError(
                f"mpmct gate count mismatch: header={expected_gates} parsed={gate_count}"
            )
    return state, wires, gate_count, max_seen


def padded_hex(value: int, bits: int) -> str:
    return "0x" + format(value, f"0{(bits + 3) // 4}x")


def write_json(path: Path, record: dict[str, object], mode: int) -> None:
    """Atomically create one JSON artifact without ever printing its contents."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
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
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--input-hex")
    parser.add_argument("--answer-out", type=Path, required=True)
    parser.add_argument("--challenge-out", type=Path, required=True)
    args = parser.parse_args()

    if args.n < 3:
        raise ValueError("n must be at least 3")
    mask = (1 << args.n) - 1
    logical_input = (
        int(args.input_hex, 16) if args.input_hex is not None else secrets.randbits(args.n)
    )
    if logical_input & ~mask:
        raise ValueError("logical input is wider than n")

    source_gates = parse_g57(args.source)
    source_output = eval_g57(source_gates, logical_input) & mask
    source_roundtrip = eval_g57(list(reversed(source_gates)), source_output) & mask
    if source_roundtrip != logical_input:
        raise AssertionError("source circuit did not invert by gate reversal")

    # All non-logical input wires are zero because the full numeric state is x.
    full_output, total_wires, final_gates, max_seen = eval_mpmct(args.final, logical_input)
    if total_wires != 4 * args.n:
        raise AssertionError(f"expected 4n={4 * args.n} final wires, got {total_wires}")
    shifted_output = (full_output >> args.n) & mask
    if shifted_output != source_output:
        raise AssertionError("selected final-circuit pair violates the n-wire output shift")

    answer_record = {
        "circuit_id": args.circuit_id,
        "challenge_input_source": "fresh Python secrets.randbits(n) draw",
        "n": args.n,
        "total_wires": total_wires,
        "source_gate_count": len(source_gates),
        "final_gate_count": final_gates,
        "max_wire_seen": max_seen,
        "logical_input_wires": f"0..{args.n - 1}",
        "zero_input_wires": f"{args.n}..{total_wires - 1}",
        "logical_output_wires": f"{args.n}..{2 * args.n - 1}",
        "logical_input_hex": padded_hex(logical_input, args.n),
        "logical_output_hex": padded_hex(source_output, args.n),
        "full_input_hex": padded_hex(logical_input, total_wires),
        "full_output_hex": padded_hex(full_output, total_wires),
        "source_reverse_roundtrip_verified": True,
        "final_shift_pair_verified": True,
        "all_final_gates_structurally_reversible": True,
    }
    challenge_record = {
        "circuit_id": args.circuit_id,
        "n": args.n,
        "total_wires": total_wires,
        "final_gate_count": final_gates,
        "logical_input_wires": f"0..{args.n - 1}",
        "zero_input_wires": f"{args.n}..{total_wires - 1}",
        "logical_output_wires": f"{args.n}..{2 * args.n - 1}",
        "target_logical_output_hex": padded_hex(source_output, args.n),
        "task": "find the n-bit logical input whose zero-slice evaluation gives the target logical output",
    }
    write_json(args.answer_out, answer_record, 0o600)
    write_json(args.challenge_out, challenge_record, 0o644)
    print(
        json.dumps(
            {
                "circuit_id": args.circuit_id,
                "answer_written": str(args.answer_out),
                "challenge_written": str(args.challenge_out),
                "selected_pair_verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
