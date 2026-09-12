#!/usr/bin/env python3
"""Exact small-n whole-trace XOR witnesses with support-reduction heuristics.

The circuit is exhaustively evaluated on its honest slice.  Candidate features
are the affine constant, nonzero initial wires, and physical gate deltas.  A
gate delta is exactly ``wire_before XOR wire_after``, so this spans arbitrary
space-time wire samples without materializing every checkpoint.

For each distinct source prefix bit, each source-gate firing, and each changed
source target state, Gaussian
elimination first obtains one exact witness.  Reversed and randomized global
basis orders then try to reduce its support.  Every reported witness is
replay-checked as a complete truth table; its support is an upper bound, not a
minimum-support certificate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
from pathlib import Path


RawCoordinate = tuple[str, int | None, int | None]


def load_exact_trace_module(path: Path):
    spec = importlib.util.spec_from_file_location("exact_trace_span", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anf_degree(table: int, n: int) -> int:
    coefficients = [(table >> assignment) & 1 for assignment in range(1 << n)]
    for bit in range(n):
        step = 1 << bit
        for mask in range(1 << n):
            if mask & step:
                coefficients[mask] ^= coefficients[mask ^ step]
    return max((mask.bit_count() for mask, value in enumerate(coefficients) if value), default=0)


def source_targets(module, path: Path, n: int):
    source = module.source_truths(path, n)
    targets = []
    seen = set()
    prefix_states = source["prefix_states"]
    for prefix in range(1, len(prefix_states) - 1):
        for wire, value in enumerate(prefix_states[prefix]):
            if value not in seen:
                seen.add(value)
                targets.append(("prefix_bit", prefix, wire, value))
    changed_targets = []
    source_gates = source["gates"]
    for gate, value in enumerate(source["firings"]):
        # Keep every firing even if it duplicates a prefix function: users
        # usually want a per-source-gate count, while prefix bits are deduped
        # only to keep the much larger target catalog tractable.
        targets.append(("gate_firing", gate, None, value))
    for gate, source_gate in enumerate(source_gates):
        target_wire = source_gate[0]
        changed_targets.append(
            (
                "changed_prefix_bit",
                gate + 1,
                target_wire,
                prefix_states[gate + 1][target_wire],
            )
        )
    return source, targets, changed_targets


def trace_features(
    module, path: Path, n: int, wire_limit: int | None, feature_mode: str
):
    with path.open() as handle:
        declared_wires, _ = module.parse_header(handle.readline(), path)
    wires = module.effective_wire_count(path, declared_wires, n)
    state, truth_mask, _ = module.initial_functions(n, wires)
    features = [
        ("constant", None, None, truth_mask, (("constant", None, None),))
    ]
    features.extend(
        (
            "initial_wire",
            None,
            wire,
            value,
            (("initial_wire", None, wire),),
        )
        for wire, value in enumerate(state)
        if value and (wire_limit is None or wire < wire_limit)
    )
    previous: dict[int, RawCoordinate] = {
        wire: ("initial_wire", None, wire) for wire in range(wires)
    }
    raw_values: dict[RawCoordinate, int] = {
        ("constant", None, None): truth_mask,
        **{
            ("initial_wire", None, wire): value
            for wire, value in enumerate(state)
        },
    }
    with path.open() as handle:
        handle.readline()
        for gate_index, line in enumerate(handle):
            if not line.strip():
                continue
            target, comp, controls = module.parse_gate(line)
            delta = module.firing(state, controls, comp, truth_mask)
            state[target] ^= delta
            after: RawCoordinate = ("target_after", gate_index, target)
            raw_values[after] = state[target]
            if wire_limit is None or target < wire_limit:
                if feature_mode == "gate-delta":
                    features.append(
                        (
                            "gate_delta",
                            gate_index,
                            target,
                            delta,
                            (previous[target], after),
                        )
                    )
                else:
                    features.append(
                        (
                            "target_after",
                            gate_index,
                            target,
                            state[target],
                            (after,),
                        )
                    )
            previous[target] = after

    # Identical truth-table columns are interchangeable for span/support.
    unique = {}
    for feature in features:
        signature = feature[3]
        if signature and signature not in unique:
            unique[signature] = feature
    return wires, list(unique.values()), raw_values


def build_basis(features, order=None, max_rank: int | None = None):
    if order is None:
        order = range(len(features))
    basis = {}
    for index in order:
        value = features[index][3]
        combination = 1 << index
        while value:
            pivot = value.bit_length() - 1
            if pivot in basis:
                value ^= basis[pivot][0]
                combination ^= basis[pivot][1]
            else:
                basis[pivot] = (value, combination)
                break
        if max_rank is not None and len(basis) == max_rank:
            break
    return basis


def solve_basis(basis, target: int, feature_count: int):
    value = target
    combination = 0
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return None
        value ^= basis[pivot][0]
        combination ^= basis[pivot][1]
    return [
        index for index in range(feature_count) if (combination >> index) & 1
    ]


def raw_coordinates(features, witness):
    selected: set[RawCoordinate] = set()
    for feature_index in witness:
        for coordinate in features[feature_index][4]:
            if coordinate in selected:
                selected.remove(coordinate)
            else:
                selected.add(coordinate)
    return selected


def support_score(features, witness):
    coordinates = raw_coordinates(features, witness)
    wire_support = sum(coordinate[0] != "constant" for coordinate in coordinates)
    return wire_support, len(witness)


def summarize(values):
    return {
        "min": min(values, default=None),
        "median": statistics.median(values) if values else None,
        "max": max(values, default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--g", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--support-cap", type=int, default=100)
    parser.add_argument("--random-bases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--include-reverse-basis",
        action="store_true",
        help="also search a basis built from the end of the artifact (may select final-port features)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("target-after", "gate-delta"),
        default="target-after",
        help="raw post-write wire samples (default), or the equivalent delta basis",
    )
    parser.add_argument(
        "--wire-limit",
        type=int,
        help="retain only features targeting wires below this physical bound",
    )
    parser.add_argument(
        "--target-kind",
        choices=("all", "prefix-bit", "gate-firing", "changed-prefix-bit"),
        default="all",
        help="restrict the audited source targets (default: all)",
    )
    parser.add_argument(
        "--target-min-degree",
        type=int,
        default=0,
        help="skip source targets whose exact honest-slice ANF degree is smaller",
    )
    parser.add_argument(
        "--target-max-degree",
        type=int,
        help="skip source targets whose exact honest-slice ANF degree is larger",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        help="restrict to one source prefix/gate index after kind filtering",
    )
    parser.add_argument(
        "--report-feature-degrees",
        action="store_true",
        help="compute exact input-ANF degrees of raw coordinates in witnesses",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.n < 1 or args.n > 20:
        raise SystemExit("exact enumeration is intended for 1 <= n <= 20")
    if args.random_bases < 0 or args.support_cap < 0:
        raise SystemExit("support controls must be nonnegative")

    module = load_exact_trace_module(Path(__file__).with_name("exact_trace_span.py"))
    source, targets, changed_targets = source_targets(module, args.source, args.n)
    requested_kind = args.target_kind.replace("-", "_")
    if requested_kind == "changed_prefix_bit":
        targets = changed_targets
    elif args.target_kind != "all":
        targets = [target for target in targets if target[0] == requested_kind]
    if args.target_min_degree:
        targets = [
            target
            for target in targets
            if anf_degree(target[3], args.n) >= args.target_min_degree
        ]
    if args.target_max_degree is not None:
        targets = [
            target
            for target in targets
            if anf_degree(target[3], args.n) <= args.target_max_degree
        ]
    if args.target_index is not None:
        targets = [target for target in targets if target[1] == args.target_index]
    wires, features, raw_values = trace_features(
        module, args.g, args.n, args.wire_limit, args.feature_mode
    )
    rng = random.Random(args.seed)
    feature_count = len(features)
    orders = [list(range(feature_count))]
    if args.include_reverse_basis:
        orders.append(list(reversed(range(feature_count))))
    for _ in range(args.random_bases):
        order = list(range(feature_count))
        rng.shuffle(order)
        orders.append(order)

    dimension = 1 << args.n
    initial_basis = build_basis(features, orders[0], dimension)
    witnesses = [solve_basis(initial_basis, target[3], feature_count) for target in targets]
    for order in orders[1:]:
        alternate_basis = build_basis(features, order, dimension)
        for target_index, target in enumerate(targets):
            candidate = solve_basis(alternate_basis, target[3], feature_count)
            current = witnesses[target_index]
            if candidate is not None and (
                current is None
                or support_score(features, candidate)
                < support_score(features, current)
            ):
                witnesses[target_index] = candidate

    results = []
    coordinate_degree_cache: dict[RawCoordinate, int] = {}

    def coordinate_degree(coordinate: RawCoordinate) -> int:
        if coordinate not in coordinate_degree_cache:
            coordinate_degree_cache[coordinate] = anf_degree(
                raw_values[coordinate], args.n
            )
        return coordinate_degree_cache[coordinate]

    for (kind, index, wire, signature), witness in zip(targets, witnesses):
        initial = solve_basis(initial_basis, signature, feature_count)
        if initial is None:
            results.append(
                {
                    "kind": kind,
                    "index": index,
                    "wire": wire,
                    "anf_degree": anf_degree(signature, args.n),
                    "recovered": False,
                }
            )
            continue
        replay = 0
        for feature_index in witness:
            replay ^= features[feature_index][3]
        if replay != signature:
            raise AssertionError("reported support-reduced witness failed replay")
        coordinates = raw_coordinates(features, witness)
        includes_constant = ("constant", None, None) in coordinates
        wire_support = sum(
            coordinate[0] != "constant" for coordinate in coordinates
        )
        results.append(
            {
                "kind": kind,
                "index": index,
                "wire": wire,
                "anf_degree": anf_degree(signature, args.n),
                "recovered": True,
                "basis_support": len(initial),
                "heuristic_support": len(witness),
                "delta_support": sum(
                    features[feature_index][0] == "gate_delta"
                    for feature_index in witness
                ),
                "wire_support": wire_support,
                "includes_affine_constant": includes_constant,
                "at_most_cap": wire_support <= args.support_cap,
                "features": [
                    {
                        "kind": features[feature_index][0],
                        "gate_index": features[feature_index][1],
                        "wire": features[feature_index][2],
                    }
                    for feature_index in witness
                ],
                "raw_coordinates": [
                    {
                        "kind": raw_kind,
                        "gate_index": gate_index,
                        "wire": raw_wire,
                        **(
                            {"anf_degree": coordinate_degree(coordinate)}
                            if args.report_feature_degrees
                            else {}
                        ),
                    }
                    for coordinate in sorted(
                        coordinates,
                        key=lambda coordinate: (
                            coordinate[0],
                            -1 if coordinate[1] is None else coordinate[1],
                            -1 if coordinate[2] is None else coordinate[2],
                        ),
                    )
                    for raw_kind, gate_index, raw_wire in [coordinate]
                ],
            }
        )

    recovered = [result for result in results if result["recovered"]]
    supports = [result["wire_support"] for result in recovered]
    by_kind = {}
    for kind in ("prefix_bit", "gate_firing", "changed_prefix_bit"):
        selected = [result for result in results if result["kind"] == kind]
        selected_recovered = [result for result in selected if result["recovered"]]
        selected_supports = [
            result["wire_support"] for result in selected_recovered
        ]
        by_kind[kind] = {
            "targets": len(selected),
            "recovered": len(selected_recovered),
            "at_most_cap": sum(
                result["at_most_cap"] for result in selected_recovered
            ),
            "heuristic_support": summarize(selected_supports),
        }
    report = {
        "schema": "exact-trace-xor-support/v1",
        "n": args.n,
        "assignments_exhaustively_enumerated": 1 << args.n,
        "source": str(args.source),
        "source_gates": source["gate_count"],
        "g": str(args.g),
        "wires": wires,
        "wire_limit": args.wire_limit,
        "feature_mode": args.feature_mode,
        "candidate_semantics": (
            "affine constant + nonzero honest-slice initial wire samples + "
            + (
                "post-write target-wire samples"
                if args.feature_mode == "target-after"
                else "physical target deltas"
            )
        ),
        "unique_nonzero_candidate_functions": len(features),
        "targets": len(results),
        "target_kind_filter": args.target_kind,
        "target_min_anf_degree": args.target_min_degree,
        "target_max_anf_degree": args.target_max_degree,
        "target_index_filter": args.target_index,
        "raw_coordinate_anf_degrees_reported": args.report_feature_degrees,
        "targets_recovered": len(recovered),
        "support_cap": args.support_cap,
        "targets_with_witness_at_most_cap": sum(
            result["at_most_cap"] for result in recovered
        ),
        "heuristic_wire_support": summarize(supports),
        "by_kind": by_kind,
        "basis_orders_tried": len(orders),
        "reverse_basis_included": args.include_reverse_basis,
        "random_basis_orders": args.random_bases,
        "witnesses_replay_validated_exhaustively": len(recovered),
        "minimum_support_certified": False,
        "target_results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
