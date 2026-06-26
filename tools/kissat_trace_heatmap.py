#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--max-var", type=int, required=True)
    parser.add_argument("--inputs", type=int, default=384)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--milestones",
        default="0,1000,10000,100000,500000,1000000,2000000,4000000,6000000",
        help="comma-separated conflict-count snapshots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_vars = args.inputs
    gate_vars = max(0, args.max_var - input_vars)
    bins = args.bins
    milestones = sorted({int(x) for x in args.milestones.split(",") if x})

    total = [0] * bins
    for gate in range(gate_vars):
        total[min(bins - 1, gate * bins // gate_vars)] += 1

    reduced = [0] * bins
    fixed = [0] * bins
    eliminated = [0] * bins
    input_events = defaultdict(int)
    snapshots: list[tuple[int, list[int]]] = []
    milestone_idx = 0
    last_conflicts = 0
    events = 0
    gate_events = 0

    def snapshot(conflicts: int) -> None:
        active = [total[i] - reduced[i] for i in range(bins)]
        snapshots.append((conflicts, active))

    while milestone_idx < len(milestones) and milestones[milestone_idx] <= 0:
        snapshot(milestones[milestone_idx])
        milestone_idx += 1

    with args.trace.open(errors="ignore") as file:
        for line in file:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            event = parts[0]
            try:
                conflicts = int(parts[1])
                external = int(parts[4])
            except ValueError:
                continue

            while (
                milestone_idx < len(milestones)
                and conflicts >= milestones[milestone_idx]
            ):
                snapshot(milestones[milestone_idx])
                milestone_idx += 1

            events += 1
            last_conflicts = conflicts
            if external <= input_vars:
                if input_vars == 384 and 1 <= external <= 384:
                    block = "ABC"[(external - 1) // 128]
                else:
                    block = "I"
                input_events[(block, event)] += 1
                continue

            gate = external - input_vars - 1
            if gate < 0 or gate >= gate_vars:
                continue
            bin_id = min(bins - 1, gate * bins // gate_vars)
            reduced[bin_id] += 1
            gate_events += 1
            if event == "F":
                fixed[bin_id] += 1
            elif event == "E":
                eliminated[bin_id] += 1

    while milestone_idx < len(milestones):
        snapshot(milestones[milestone_idx])
        milestone_idx += 1

    final_active = [total[i] - reduced[i] for i in range(bins)]
    print(
        "SUMMARY",
        "events",
        events,
        "gate_events",
        gate_events,
        "last_conflicts",
        last_conflicts,
        "gate_vars",
        gate_vars,
    )
    print("INPUT_EVENTS block event count")
    for key in sorted(input_events):
        print(key[0], key[1], input_events[key])

    print("FINAL_BINS bin start end total active fixed eliminated pct_active")
    for bin_id in range(bins):
        start = bin_id * gate_vars // bins
        end = ((bin_id + 1) * gate_vars // bins) - 1
        pct = final_active[bin_id] / total[bin_id] if total[bin_id] else 0
        print(
            bin_id,
            start,
            end,
            total[bin_id],
            final_active[bin_id],
            fixed[bin_id],
            eliminated[bin_id],
            f"{pct:.6f}",
        )

    print("SNAPSHOT_ACTIVE_PCT conflicts", *range(bins))
    for conflicts, active in snapshots:
        print(
            conflicts,
            *[
                f"{(active[i] / total[i] if total[i] else 0):.4f}"
                for i in range(bins)
            ],
        )

    print("HARDEST_FINAL_BINS bin active pct_active")
    hardest = sorted(
        range(bins),
        key=lambda i: (final_active[i] / total[i] if total[i] else 0),
        reverse=True,
    )
    for bin_id in hardest[:10]:
        pct = final_active[bin_id] / total[bin_id] if total[bin_id] else 0
        print(bin_id, final_active[bin_id], f"{pct:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
