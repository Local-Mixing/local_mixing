#!/usr/bin/env python3
"""Empirically check that nonzero padding does not expose the C(x) port.

The report contains aggregate match counts only.  Fresh sampled inputs and
outputs are kept in memory and are never printed or written.
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

if __package__:
    from . import verify_layout
else:
    import verify_layout

apply_g57_lanes = verify_layout.apply_g57_lanes
eval_final_lanes = verify_layout.eval_final_lanes
parse_g57 = verify_layout.parse_g57
write_report = verify_layout.write_report


def packed_wire_lanes(values: list[int], width: int) -> list[int]:
    return [
        sum(((value >> wire) & 1) << lane for lane, value in enumerate(values))
        for wire in range(width)
    ]


def subset_block_matches(
    final_state: list[int],
    reference: list[int],
    block_base: int,
    n: int,
    subset_mask: int,
) -> int:
    mismatch = 0
    for offset in range(n):
        mismatch |= final_state[block_base + offset] ^ reference[offset]
    return (subset_mask & ~mismatch).bit_count()


def nonzero_random(bits: int) -> int:
    return secrets.randbelow((1 << bits) - 1) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--samples-per-case", type=int, default=256)
    args = parser.parse_args()

    if args.n < 3 or args.samples_per_case < 2:
        raise ValueError("n must be >=3 and samples-per-case must be >=2")

    cases = ("inner_aux_only", "outer_gadget_aux_only", "both_aux_regions")
    total_samples = len(cases) * args.samples_per_case
    lane_mask = (1 << total_samples) - 1

    x_values: list[int] = []
    inner_values: list[int] = []
    outer_values: list[int] = []
    for case in cases:
        for _ in range(args.samples_per_case):
            x_values.append(secrets.randbits(args.n))
            inner_values.append(
                nonzero_random(args.n) if case in ("inner_aux_only", "both_aux_regions") else 0
            )
            outer_values.append(
                nonzero_random(2 * args.n)
                if case in ("outer_gadget_aux_only", "both_aux_regions")
                else 0
            )

    x_lanes = packed_wire_lanes(x_values, args.n)
    source_output = x_lanes.copy()
    source_gates = parse_g57(args.source, args.n)
    apply_g57_lanes(source_output, source_gates, lane_mask)

    final_state = (
        x_lanes
        + packed_wire_lanes(inner_values, args.n)
        + packed_wire_lanes(outer_values, 2 * args.n)
    )
    total_wires, final_gate_count, max_wire_seen = eval_final_lanes(
        args.final, final_state, lane_mask
    )
    if total_wires != 4 * args.n:
        raise AssertionError(f"expected {4 * args.n} wires, got {total_wires}")

    matches: dict[str, dict[str, int]] = {}
    for case_index, case in enumerate(cases):
        start = case_index * args.samples_per_case
        subset_mask = ((1 << args.samples_per_case) - 1) << start
        matches[case] = {
            f"{block * args.n}..{(block + 1) * args.n - 1}": subset_block_matches(
                final_state,
                source_output,
                block * args.n,
                args.n,
                subset_mask,
            )
            for block in range(4)
        }

    intended_block = f"{args.n}..{2 * args.n - 1}"
    clean_answer_matches = {case: counts[intended_block] for case, counts in matches.items()}
    report: dict[str, object] = {
        "circuit_id": args.circuit_id,
        "n": args.n,
        "samples_per_case": args.samples_per_case,
        "cases": {
            "inner_aux_only": f"wires {args.n}..{2 * args.n - 1} nonzero; wires {2 * args.n}..{4 * args.n - 1} zero",
            "outer_gadget_aux_only": f"wires {args.n}..{2 * args.n - 1} zero; wires {2 * args.n}..{4 * args.n - 1} nonzero",
            "both_aux_regions": f"both wires {args.n}..{2 * args.n - 1} and {2 * args.n}..{4 * args.n - 1} nonzero",
        },
        "C_of_x_match_count_by_case_and_128_wire_output_block": matches,
        "clean_answer_block": intended_block,
        "clean_answer_match_count_by_case": clean_answer_matches,
        "no_clean_answer_observed_off_slice": all(value == 0 for value in clean_answer_matches.values()),
        "final_gate_count": final_gate_count,
        "max_wire_seen": max_wire_seen,
        "all_final_gates_structurally_reversible": True,
        "note": "This is randomized evidence. The interface contract itself defines C(x) only when every padding input is zero.",
    }
    write_report(args.report_out, report)
    print(
        json.dumps(
            {
                "circuit_id": args.circuit_id,
                "report_written": str(args.report_out),
                "samples_per_case": args.samples_per_case,
                "no_clean_answer_observed_off_slice": report[
                    "no_clean_answer_observed_off_slice"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
