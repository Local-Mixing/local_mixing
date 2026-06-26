#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
VAL = {c: i for i, c in enumerate(CHARS)}


def parse(path: Path) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    w: list[int] = []
    overflow = 0
    for ch in path.read_text():
        if ch == ";":
            if w:
                if len(w) != 3:
                    raise RuntimeError(f"bad gate arity {len(w)} near {len(gates)}")
                gates.append((w[0], w[1], w[2]))
                w.clear()
            overflow = 0
        elif ch == "~":
            overflow += 1
        elif ch in "\n\r\t ":
            continue
        else:
            if ch not in VAL:
                raise RuntimeError(f"bad char {ch!r}")
            w.append(VAL[ch] + 83 * overflow)
            overflow = 0
    if w:
        raise RuntimeError(f"unterminated gate {w}")
    return gates


def block(wire: int) -> int:
    return wire // 128


def summarize_counts(label: str, counts: list[int]) -> None:
    mean = sum(counts) / len(counts)
    var = sum((x - mean) ** 2 for x in counts) / len(counts)
    print(
        label,
        "min",
        min(counts),
        "max",
        max(counts),
        "mean",
        f"{mean:.2f}",
        "stdev",
        f"{math.sqrt(var):.2f}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=Path)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    gates = parse(args.circuit)
    n = len(gates)
    max_wire = max(max(g) for g in gates)
    print("BASIC gates", n, "wires", max_wire + 1)

    unique = len(set(gates))
    print("UNIQUE_GATES unique", unique, "duplicates", n - unique)

    role_counts = [[0, 0, 0] for _ in range(3)]
    target_per_wire = [0] * (max_wire + 1)
    control1_per_wire = [0] * (max_wire + 1)
    control2_per_wire = [0] * (max_wire + 1)
    block_pattern = collections.Counter()
    span_counts = collections.Counter()
    repeated_wire_gate = 0
    for t, c1, c2 in gates:
        role_counts[0][block(t)] += 1
        role_counts[1][block(c1)] += 1
        role_counts[2][block(c2)] += 1
        target_per_wire[t] += 1
        control1_per_wire[c1] += 1
        control2_per_wire[c2] += 1
        block_pattern[(block(t), block(c1), block(c2))] += 1
        span_counts[max(t, c1, c2) - min(t, c1, c2)] += 1
        if len({t, c1, c2}) < 3:
            repeated_wire_gate += 1

    print("ROLE_BLOCK_COUNTS role A B C")
    for name, counts in zip(("target", "control1", "control2"), role_counts):
        print(name, *counts)

    summarize_counts("TARGET_PER_WIRE", target_per_wire)
    summarize_counts("CONTROL1_PER_WIRE", control1_per_wire)
    summarize_counts("CONTROL2_PER_WIRE", control2_per_wire)
    print("REPEATED_WIRE_INSIDE_GATE", repeated_wire_gate)

    print("TOP_BLOCK_PATTERNS target control1 control2 count pct")
    for pat, count in block_pattern.most_common(args.top):
        print(*pat, count, f"{count / n:.6f}")

    immediate_same = 0
    immediate_same_target = 0
    adjacent_share = 0
    adjacent_target_control = 0
    pair_counts = collections.Counter()
    triple_counts = collections.Counter()
    for i in range(1, n):
        prev = gates[i - 1]
        cur = gates[i]
        if cur == prev:
            immediate_same += 1
        if cur[0] == prev[0]:
            immediate_same_target += 1
        if set(cur) & set(prev):
            adjacent_share += 1
        if cur[0] in prev[1:] or prev[0] in cur[1:]:
            adjacent_target_control += 1
        pair_counts[(prev, cur)] += 1
        if i >= 2:
            triple_counts[(gates[i - 2], prev, cur)] += 1

    print(
        "ADJACENCY",
        "same_gate",
        immediate_same,
        "same_target",
        immediate_same_target,
        "share_any_wire",
        adjacent_share,
        f"{adjacent_share / (n - 1):.6f}",
        "target_control_collision",
        adjacent_target_control,
        f"{adjacent_target_control / (n - 1):.6f}",
    )

    repeated_pairs = [(k, v) for k, v in pair_counts.items() if v > 1]
    repeated_triples = [(k, v) for k, v in triple_counts.items() if v > 1]
    print(
        "REPEATED_NGRAMS",
        "pairs",
        len(repeated_pairs),
        "max_pair_count",
        max((v for _, v in repeated_pairs), default=1),
        "triples",
        len(repeated_triples),
        "max_triple_count",
        max((v for _, v in repeated_triples), default=1),
    )

    print("TOP_REPEATED_PAIRS count pair")
    for pair, count in sorted(repeated_pairs, key=lambda kv: kv[1], reverse=True)[
        : args.top
    ]:
        print(count, pair)

    bins = args.bins
    print("WINDOW_BINS bin start end targetA targetB targetC share_adj pct_share distinct_wires")
    for b in range(bins):
        start = b * n // bins
        end = (b + 1) * n // bins
        sub = gates[start:end]
        targets = [0, 0, 0]
        wires = set()
        local_share = 0
        for j, gate in enumerate(sub):
            targets[block(gate[0])] += 1
            wires.update(gate)
            if j and set(gate) & set(sub[j - 1]):
                local_share += 1
        denom = max(1, len(sub) - 1)
        print(
            b,
            start,
            end - 1,
            *targets,
            local_share,
            f"{local_share / denom:.6f}",
            len(wires),
        )

    print("SPAN_QUANTILES q span")
    expanded = []
    for span, count in span_counts.items():
        expanded.append((span, count))
    expanded.sort()
    total = 0
    idx = 0
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        target = q * n
        while idx < len(expanded) and total + expanded[idx][1] < target:
            total += expanded[idx][1]
            idx += 1
        if idx < len(expanded):
            print(q, expanded[idx][0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
