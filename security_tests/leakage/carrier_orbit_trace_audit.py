#!/usr/bin/env python3
"""Audit affine cancellation across every repeated-U0 carrier coordinate.

Each carrier coordinate at each iteration of the public U0 permutation is a
Boolean truth-table column.  The audit asks whether the corresponding decode D
is in their GF(2) span.  This is an isolated carrier-map test: it deliberately
does not model masks, source-dependent updates, folds, rolling, or ports.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path


def parse_table(source: str, name: str) -> list[int]:
    match = re.search(rf"const {name}: \[u8; \d+\] = \[(.*?)\];", source, re.S)
    if not match:
        raise RuntimeError(f"missing {name}")
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


def decode(name: str, word: int) -> int:
    c = lambda lane: (word >> lane) & 1
    if name == "supplied-five":
        return c(0) ^ c(1) * c(2) ^ c(1) * c(3) ^ c(2) * c(3) ^ c(1) * c(4)
    if name == "strong-five":
        return (
            c(0)
            ^ c(2)
            ^ c(2) * c(3)
            ^ c(1) * c(2) * c(3)
            ^ c(1) * c(2) * c(4)
            ^ c(3) * c(4)
        )
    if name in ("compact-six", "strong-six"):
        return (
            c(0)
            ^ c(1) * c(2)
            ^ c(3)
            ^ c(1) * c(3)
            ^ c(4)
            ^ c(1) * c(3) * c(4)
            ^ c(5)
            ^ c(2) * c(5)
            ^ c(2) * c(3) * c(5)
        )
    if name == "seven":
        return (
            c(0)
            ^ c(1)
            ^ c(2)
            ^ c(3) * c(4)
            ^ c(5) * c(6)
            ^ c(3) * c(4) * c(5) * c(6)
        )
    raise ValueError(name)


def orbit_period(permutation: list[int]) -> tuple[int, list[int]]:
    seen: set[int] = set()
    period = 1
    cycles = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        word = start
        length = 0
        while word not in seen:
            seen.add(word)
            length += 1
            word = permutation[word]
        cycles.append(length)
        period = math.lcm(period, length)
    return period, sorted(cycles)


def truth_vector(values) -> int:
    return sum(1 << row for row, value in enumerate(values) if value)


def anf_degree(vector: int, carriers: int) -> int:
    coefficients = [
        (vector >> word) & 1 for word in range(1 << carriers)
    ]
    for lane in range(carriers):
        step = 1 << lane
        for mask in range(1 << carriers):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    return max(
        (
            mask.bit_count()
            for mask, coefficient in enumerate(coefficients)
            if coefficient
        ),
        default=-1,
    )


def insert(vector: int, combination: int, basis: dict[int, tuple[int, int]]) -> bool:
    while vector:
        pivot = vector.bit_length() - 1
        if pivot in basis:
            old_vector, old_combination = basis[pivot]
            vector ^= old_vector
            combination ^= old_combination
        else:
            basis[pivot] = (vector, combination)
            return True
    return False


def solve(columns: list[int], target: int) -> tuple[int, int | None]:
    basis: dict[int, tuple[int, int]] = {}
    for index, column in enumerate(columns):
        insert(column, 1 << index, basis)
    combination = 0
    residual = target
    while residual:
        pivot = residual.bit_length() - 1
        if pivot not in basis:
            return len(basis), None
        vector, backpointer = basis[pivot]
        residual ^= vector
        combination ^= backpointer
    return len(basis), combination


def exact_minimum_support(columns: list[int], target: int):
    """Meet-in-the-middle minimum over one explicitly bounded column window."""
    middle = len(columns) // 2
    left = columns[:middle]
    right = columns[middle:]
    right_best: dict[int, tuple[int, int]] = {}
    for mask in range(1 << len(right)):
        vector = 0
        for index, column in enumerate(right):
            if (mask >> index) & 1:
                vector ^= column
        candidate = (mask.bit_count(), mask)
        if vector not in right_best or candidate < right_best[vector]:
            right_best[vector] = candidate
    optimum = None
    for mask in range(1 << len(left)):
        vector = 0
        for index, column in enumerate(left):
            if (mask >> index) & 1:
                vector ^= column
        complement = target ^ vector
        if complement not in right_best:
            continue
        right_weight, right_mask = right_best[complement]
        candidate = (mask.bit_count() + right_weight, mask, right_mask)
        if optimum is None or candidate < optimum:
            optimum = candidate
    return optimum


def audit_map(name: str, permutation: list[int]) -> dict[str, object]:
    carriers = len(permutation).bit_length() - 1
    period, cycles = orbit_period(permutation)
    state = list(range(1 << carriers))
    columns: list[int] = []
    metadata: list[tuple[int, int]] = []
    for time in range(period):
        for lane in range(carriers):
            columns.append(
                truth_vector((word >> lane) & 1 for word in state)
            )
            metadata.append((time, lane))
        state = [permutation[word] for word in state]
    target = truth_vector(
        decode(name, word) for word in range(1 << carriers)
    )
    rank, witness = solve(columns, target)
    result: dict[str, object] = {
        "mode": name,
        "carriers": carriers,
        "period": period,
        "cycle_lengths": cycles,
        "coordinate_columns": len(columns),
        "trace_rank": rank,
        "function_space_dimension": 1 << carriers,
        "decode_in_span": witness is not None,
        "augmented_rank": rank + int(witness is None),
    }
    if witness is not None:
        selected = [
            index
            for index in range(len(columns))
            if (witness >> index) & 1
        ]
        histogram = collections.Counter(
            anf_degree(columns[index], carriers) for index in selected
        )
        result["gaussian_witness"] = {
            "support_upper_bound": len(selected),
            "coordinates": [metadata[index] for index in selected],
            "feature_anf_degree_histogram": dict(sorted(histogram.items())),
            "minimum_support_certified": False,
        }
    if name == "supplied-five":
        bounded_columns = columns[: 6 * carriers]
        optimum = exact_minimum_support(bounded_columns, target)
        if optimum is None:
            raise AssertionError("supplied-five bounded decode left the span")
        weight, left_mask, right_mask = optimum
        middle = len(bounded_columns) // 2
        selected = [
            index
            for index in range(len(bounded_columns))
            if (index < middle and (left_mask >> index) & 1)
            or (index >= middle and (right_mask >> (index - middle)) & 1)
        ]
        result["times_0_through_5_exact_minimum"] = {
            "support": weight,
            "coordinates": [metadata[index] for index in selected],
        }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "src/preprocessing/gadgets.rs",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    maps = [
        ("supplied-five", parse_table(source, "FIVE_CARRIER_U0")),
        ("strong-five", parse_table(source, "STRONG_FIVE_CARRIER_U0")),
        ("compact-six", parse_table(source, "SIX_CARRIER_U0")),
        ("strong-six", parse_table(source, "STRONG_SIX_CARRIER_U0")),
        ("seven", parse_table(source, "SEVEN_CARRIER_U0")),
    ]
    report = {
        "schema": "carrier-orbit-trace-affine/v1",
        "source": str(args.source),
        "model": "all raw carrier coordinates over the complete repeated-U0 orbit",
        "scope": "isolated update permutation only; not a full-gadget claim",
        "maps": [audit_map(name, permutation) for name, permutation in maps],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
