#!/usr/bin/env python3
"""Independently verify the zero-slice output layout of a mixed TDP circuit.

This evaluator is deliberately separate from the Rust verifier used during
generation.  It evaluates many inputs in parallel by storing one sample per
bit of a Python integer.  No sampled input or output value is printed or
written; the report contains only aggregate, non-secret verification counts.
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


def parse_g57(path: Path, n: int) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    for encoded in path.read_text(encoding="utf-8").split(";"):
        encoded = encoded.strip()
        if not encoded:
            continue
        wires: list[int] = []
        overflow = 0
        for char in encoded:
            if char == "~":
                overflow += 1
                continue
            try:
                wire = overflow * BASE + WIRE_INDEX[char]
            except KeyError as error:
                raise ValueError(f"invalid g57 character {char!r}") from error
            wires.append(wire)
            overflow = 0
        if overflow or len(wires) != 3:
            raise ValueError("malformed g57 gate")
        target, positive, negative = wires
        if len({target, positive, negative}) != 3 or max(wires) >= n:
            raise ValueError("invalid g57 wire tuple")
        gates.append((target, positive, negative))
    return gates


def apply_g57_lanes(
    state: list[int], gates: list[tuple[int, int, int]], lane_mask: int
) -> None:
    for target, positive, negative in gates:
        state[target] ^= state[positive] | (lane_mask ^ state[negative])


def eval_final_lanes(
    path: Path, state: list[int], lane_mask: int
) -> tuple[int, int, int]:
    max_wire_seen = -1
    with path.open("r", encoding="utf-8") as source:
        header = source.readline().split()
        if len(header) != 3 or header[0] != "mpmct1":
            raise ValueError("invalid mpmct1 header")
        wires = int(header[1])
        expected_gates = int(header[2])
        if len(state) != wires:
            raise ValueError("state/header wire-count mismatch")

        gate_count = 0
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

            fires = lane_mask
            seen_controls: set[int] = set()
            for offset in range(width):
                control = fields[3 + 2 * offset]
                polarity = fields[4 + 2 * offset]
                if (
                    not 0 <= control < wires
                    or polarity not in (0, 1)
                    or control == target
                    or control in seen_controls
                ):
                    raise ValueError(f"invalid control at line {line_number}")
                seen_controls.add(control)
                literal = state[control] if polarity else lane_mask ^ state[control]
                fires &= literal
                max_wire_seen = max(max_wire_seen, control)
            if complemented:
                fires ^= lane_mask
            state[target] ^= fires
            max_wire_seen = max(max_wire_seen, target)
            gate_count += 1

        if gate_count != expected_gates:
            raise ValueError(
                f"gate-count mismatch: header={expected_gates} parsed={gate_count}"
            )
    return wires, gate_count, max_wire_seen


def block_match_count(
    final_state: list[int], reference: list[int], base: int, n: int, samples: int
) -> int:
    mismatch_lanes = 0
    for offset in range(n):
        mismatch_lanes |= final_state[base + offset] ^ reference[offset]
    return samples - mismatch_lanes.bit_count()


def block_stats(state: list[int], start: int, width: int, lane_mask: int) -> dict[str, int]:
    fixed_zero = sum(state[wire] == 0 for wire in range(start, start + width))
    fixed_one = sum(state[wire] == lane_mask for wire in range(start, start + width))
    return {
        "width": width,
        "varying_wires_across_samples": width - fixed_zero - fixed_one,
        "fixed_zero_wires_across_samples": fixed_zero,
        "fixed_one_wires_across_samples": fixed_one,
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
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
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()

    if args.n < 3 or args.samples < 2:
        raise ValueError("n must be >=3 and samples must be >=2")
    lane_mask = (1 << args.samples) - 1
    sampled_inputs = [secrets.randbits(args.n) for _ in range(args.samples)]
    input_lanes = [
        sum(((value >> wire) & 1) << lane for lane, value in enumerate(sampled_inputs))
        for wire in range(args.n)
    ]

    source_gates = parse_g57(args.source, args.n)
    source_output = input_lanes.copy()
    apply_g57_lanes(source_output, source_gates, lane_mask)
    source_roundtrip = source_output.copy()
    apply_g57_lanes(source_roundtrip, list(reversed(source_gates)), lane_mask)
    if source_roundtrip != input_lanes:
        raise AssertionError("source gate reversal did not recover every sampled input")

    final_state = input_lanes + [0] * (3 * args.n)
    total_wires, final_gate_count, max_wire_seen = eval_final_lanes(
        args.final, final_state, lane_mask
    )
    if total_wires != 4 * args.n:
        raise AssertionError(f"expected {4 * args.n} final wires, got {total_wires}")

    matches_by_n_block = {
        f"{block * args.n}..{(block + 1) * args.n - 1}": block_match_count(
            final_state, source_output, block * args.n, args.n, args.samples
        )
        for block in range(4)
    }
    intended_block = f"{args.n}..{2 * args.n - 1}"
    if matches_by_n_block[intended_block] != args.samples:
        raise AssertionError("the intended shifted output block does not equal C(x)")

    report: dict[str, object] = {
        "circuit_id": args.circuit_id,
        "input_convention": {
            "x_wires": f"0..{args.n - 1}",
            "zero_wires": f"{args.n}..{total_wires - 1}",
        },
        "output_convention": {
            "ignored_low_junk_wires": f"0..{args.n - 1}",
            "verified_C_of_x_wires": intended_block,
            "ignored_high_junk_wires": f"{2 * args.n}..{total_wires - 1}",
        },
        "fresh_csprng_samples": args.samples,
        "source_gate_count": len(source_gates),
        "final_gate_count": final_gate_count,
        "max_wire_seen": max_wire_seen,
        "source_reverse_roundtrip_passed": True,
        "C_of_x_match_count_by_128_wire_output_block": matches_by_n_block,
        "intended_shift_passed": True,
        "ignored_low_block_observed_stats": block_stats(
            final_state, 0, args.n, lane_mask
        ),
        "ignored_high_2n_block_observed_stats": block_stats(
            final_state, 2 * args.n, 2 * args.n, lane_mask
        ),
        "note": "Junk means ignored/unconstrained by the TDP interface, not random.",
    }
    write_report(args.report_out, report)
    print(
        json.dumps(
            {
                "circuit_id": args.circuit_id,
                "report_written": str(args.report_out),
                "samples": args.samples,
                "intended_shift_passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
