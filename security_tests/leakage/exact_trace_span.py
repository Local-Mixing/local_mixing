#!/usr/bin/env python3
"""Exact whole-trace affine-span audit for small honest-slice circuits.

For n small enough to enumerate all 2**n data inputs, represent each Boolean
function by its complete truth table packed into one Python integer.  For each
MPMCT1 stage H, reduce

    {1, initial wire functions, every physical gate firing delta}

over GF(2), then test every source-C prefix wire and source-gate firing for
membership.  Initial wires plus gate deltas span exactly the same functions as
all wire values at all gate boundaries, without materializing the much larger
space-time state matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Iterable


BASE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
CHAR_TO_WIRE = {char: index for index, char in enumerate(BASE_CHARS)}


class XorBasis:
    def __init__(self, track_provenance: bool = False) -> None:
        self.rows: dict[int, int] = {}
        # Kept separate so the ordinary membership audit has its original
        # memory profile.  When enabled, each integer is a bitset over the
        # accepted independent feature catalog.
        self.combinations: dict[int, int] | None = (
            {} if track_provenance else None
        )

    @property
    def rank(self) -> int:
        return len(self.rows)

    def reduce(self, value: int, combination: int = 0) -> tuple[int, int]:
        while value:
            pivot = value.bit_length() - 1
            row = self.rows.get(pivot)
            if row is None:
                break
            value ^= row
            if self.combinations is not None:
                combination ^= self.combinations[pivot]
        return value, combination

    def offer(self, value: int, generator: int | None = None) -> bool:
        if self.combinations is not None:
            if generator is None:
                raise ValueError("tracked bases require a generator ID")
            combination = 1 << generator
        else:
            combination = 0
        value, combination = self.reduce(value, combination)
        if not value:
            return False
        pivot = value.bit_length() - 1
        self.rows[pivot] = value
        if self.combinations is not None:
            self.combinations[pivot] = combination
        return True

    def contains(self, value: int) -> bool:
        return self.reduce(value)[0] == 0

    def solve(self, value: int) -> int | None:
        """Return a generator bitset whose XOR is value, or None."""
        if self.combinations is None:
            raise ValueError("solve requires track_provenance=True")
        remainder, combination = self.reduce(value)
        return combination if remainder == 0 else None


def set_bits(value: int) -> Iterable[int]:
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_wire(token: str, cursor: int) -> tuple[int, int]:
    overflow = 0
    while cursor < len(token) and token[cursor] == "~":
        overflow += 1
        cursor += 1
    if cursor >= len(token):
        raise ValueError(f"missing base character after '~' in {token!r}")
    try:
        base = CHAR_TO_WIRE[token[cursor]]
    except KeyError as exc:
        raise ValueError(f"invalid base-83 wire character in {token!r}") from exc
    return overflow * 83 + base, cursor + 1


def read_g57(path: Path) -> list[tuple[int, int, int]]:
    gates: list[tuple[int, int, int]] = []
    for token in path.read_text().strip().split(";"):
        if not token:
            continue
        wires: list[int] = []
        cursor = 0
        while cursor < len(token):
            wire, cursor = decode_wire(token, cursor)
            wires.append(wire)
        if len(wires) != 3:
            raise ValueError(f"G57 gate does not contain three wires: {token!r}")
        gates.append((wires[0], wires[1], wires[2]))
    return gates


def initial_functions(n: int, wires: int) -> tuple[list[int], int, int]:
    samples = 1 << n
    truth_mask = (1 << samples) - 1
    state = [0] * wires
    for wire in range(n):
        value = 0
        for assignment in range(samples):
            if (assignment >> wire) & 1:
                value |= 1 << assignment
        state[wire] = value
    return state, truth_mask, samples


def g57_firing(state: list[int], gate: tuple[int, int, int], truth_mask: int) -> int:
    _, x, y = gate
    if x == y:
        return truth_mask
    return state[x] | ((~state[y]) & truth_mask)


def source_truths(path: Path, n: int) -> dict[str, object]:
    gates = read_g57(path)
    state, truth_mask, samples = initial_functions(n, n)
    prefix_states = [tuple(state)]
    firings: list[int] = []
    for gate in gates:
        firing = g57_firing(state, gate, truth_mask)
        firings.append(firing)
        state[gate[0]] ^= firing
        prefix_states.append(tuple(state))
    return {
        "gates": gates,
        "gate_count": len(gates),
        "samples": samples,
        "truth_mask": truth_mask,
        "prefix_states": prefix_states,
        "firings": firings,
        "final": tuple(state),
    }


def parse_header(line: str, path: Path) -> tuple[int, int]:
    fields = line.split()
    if len(fields) != 3 or fields[0] != "mpmct1":
        raise ValueError(f"bad MPMCT1 header in {path}: {line.rstrip()!r}")
    return int(fields[1]), int(fields[2])


def parse_gate(line: str) -> tuple[int, bool, list[tuple[int, bool]]]:
    fields = [int(field) for field in line.split()]
    if len(fields) < 3:
        raise ValueError(f"short MPMCT1 gate: {line.rstrip()!r}")
    target, comp, width = fields[:3]
    if len(fields) != 3 + 2 * width:
        raise ValueError(f"MPMCT1 gate width mismatch: {line.rstrip()!r}")
    controls = [(fields[i], bool(fields[i + 1])) for i in range(3, len(fields), 2)]
    return target, bool(comp), controls


def effective_wire_count(path: Path, declared_wires: int, n: int) -> int:
    """Honor the header unless gate references prove that the circuit is wider.

    A historical crossing-stage writer emitted ``mpmct1 1 ...`` while retaining
    the full layout.  The Rust parser also accepts that header because it does
    not range-check gate wires.  Scanning once lets this audit execute the same
    gate list without silently trusting inconsistent metadata.
    """
    maximum = n - 1
    with path.open() as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue
            target, _, controls = parse_gate(line)
            maximum = max(maximum, target, *(wire for wire, _ in controls))
    return max(declared_wires, maximum + 1, n)


def firing(
    state: list[int], controls: Iterable[tuple[int, bool]], comp: bool, truth_mask: int
) -> int:
    value = truth_mask
    for wire, positive in controls:
        source = state[wire]
        value &= source if positive else (~source) & truth_mask
    return ((~value) & truth_mask) if comp else value


def membership_counts(source: dict[str, object], basis: XorBasis, n: int) -> dict[str, int]:
    prefix_states = source["prefix_states"]
    source_firings = source["firings"]
    assert isinstance(prefix_states, list)
    assert isinstance(source_firings, list)

    all_states = [value for prefix in prefix_states for value in prefix]
    interior_states = [value for prefix in prefix_states[1:-1] for value in prefix]
    return {
        "source_state_bits": len(all_states),
        "source_state_bits_recovered": sum(basis.contains(value) for value in all_states),
        "interior_state_bits": len(interior_states),
        "interior_state_bits_recovered": sum(
            basis.contains(value) for value in interior_states
        ),
        "source_gate_firings": len(source_firings),
        "source_gate_firings_recovered": sum(
            basis.contains(value) for value in source_firings
        ),
        "unique_source_state_functions": len(set(all_states)),
        "unique_source_firing_functions": len(set(source_firings)),
    }


FeatureLabel = dict[str, object]


def offer_labeled(
    basis: XorBasis,
    catalog: list[FeatureLabel],
    value: int,
    label: FeatureLabel,
) -> bool:
    """Offer a candidate and catalog it only if it becomes a basis generator."""
    generator = len(catalog)
    if not basis.offer(value, generator):
        return False
    catalog.append({"id": generator, **label})
    return True


def build_witnesses(
    source: dict[str, object],
    basis: XorBasis,
    input_basis: XorBasis,
    max_total_terms: int,
) -> list[dict[str, object]]:
    source_firings = source["firings"]
    source_gates = source["gates"]
    assert isinstance(source_firings, list)
    assert isinstance(source_gates, list)
    witnesses: list[dict[str, object]] = []
    total_terms = 0
    for source_gate_index, target in enumerate(source_firings):
        selection = basis.solve(target)
        if selection is None:
            continue
        feature_ids = list(set_bits(selection))
        total_terms += len(feature_ids)
        if max_total_terms and total_terms > max_total_terms:
            raise ValueError(
                "constructive witnesses exceed --witness-max-total-terms="
                f"{max_total_terms}; refusing to truncate"
            )
        target_wire, positive_control, negative_control = source_gates[source_gate_index]
        witnesses.append(
            {
                "source_gate_index": source_gate_index,
                "source_gate": {
                    "target": target_wire,
                    "positive_control": positive_control,
                    "negative_control": negative_control,
                },
                "input_affine": input_basis.contains(target),
                "feature_ids": feature_ids,
                "basis_term_count": len(feature_ids),
            }
        )
    return witnesses


def validate_witnesses(
    path: Path,
    n: int,
    wires: int,
    catalog: list[FeatureLabel],
    witnesses: list[dict[str, object]],
    source: dict[str, object],
) -> int:
    """Independently replay labeled features and verify every full truth table."""
    state, truth_mask, _ = initial_functions(n, wires)
    reconstructed = [0] * len(witnesses)
    subscribers: dict[int, list[int]] = {}
    for witness_index, witness in enumerate(witnesses):
        feature_ids = witness["feature_ids"]
        assert isinstance(feature_ids, list)
        for feature_id in feature_ids:
            subscribers.setdefault(int(feature_id), []).append(witness_index)

    delta_features: dict[int, int] = {}
    for feature_id, feature in enumerate(catalog):
        if int(feature["id"]) != feature_id:
            raise AssertionError("feature catalog IDs are not contiguous")
        kind = feature["kind"]
        if kind == "constant":
            value = truth_mask
        elif kind == "initial_wire":
            value = state[int(feature["wire"])]
        elif kind == "gate_delta":
            gate_index = int(feature["gate_index"])
            if gate_index in delta_features:
                raise AssertionError("duplicate gate-delta feature")
            delta_features[gate_index] = feature_id
            continue
        else:
            raise AssertionError(f"unknown feature kind: {kind}")
        for witness_index in subscribers.get(feature_id, []):
            reconstructed[witness_index] ^= value

    processed = 0
    with path.open() as handle:
        parse_header(handle.readline(), path)
        for line in handle:
            if not line.strip():
                continue
            target, comp, controls = parse_gate(line)
            delta = firing(state, controls, comp, truth_mask)
            feature_id = delta_features.get(processed)
            if feature_id is not None:
                feature = catalog[feature_id]
                if int(feature["target_wire"]) != target:
                    raise AssertionError("gate-delta label target does not match replay")
                for witness_index in subscribers.get(feature_id, []):
                    reconstructed[witness_index] ^= delta
            state[target] ^= delta
            processed += 1

    source_firings = source["firings"]
    assert isinstance(source_firings, list)
    validated = 0
    for witness, actual in zip(witnesses, reconstructed):
        source_gate_index = int(witness["source_gate_index"])
        ok = actual == source_firings[source_gate_index]
        witness["validated_by_exhaustive_replay"] = ok
        validated += ok
    if validated != len(witnesses):
        raise AssertionError(
            f"constructive witness validation failed: {validated}/{len(witnesses)}"
        )
    return validated


def stat_triplet(values: list[int]) -> dict[str, int | float | None]:
    return {
        "min": min(values, default=None),
        "median": median(values) if values else None,
        "max": max(values, default=None),
    }


def summarize_witnesses(
    catalog: list[FeatureLabel], witnesses: list[dict[str, object]]
) -> dict[str, object]:
    by_kind: dict[str, int] = {}
    for feature in catalog:
        kind = str(feature["kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1

    term_counts: list[int] = []
    delta_counts: list[int] = []
    delta_spans: list[int] = []
    non_input: list[dict[str, object]] = []
    for witness in witnesses:
        feature_ids = witness["feature_ids"]
        assert isinstance(feature_ids, list)
        delta_gates = [
            int(catalog[int(feature_id)]["gate_index"])
            for feature_id in feature_ids
            if catalog[int(feature_id)]["kind"] == "gate_delta"
        ]
        witness["delta_term_count"] = len(delta_gates)
        witness["first_delta_gate"] = min(delta_gates, default=None)
        witness["last_delta_gate"] = max(delta_gates, default=None)
        witness["delta_gate_span"] = (
            max(delta_gates) - min(delta_gates) if delta_gates else 0
        )
        term_counts.append(int(witness["basis_term_count"]))
        delta_counts.append(len(delta_gates))
        delta_spans.append(int(witness["delta_gate_span"]))
        if not bool(witness["input_affine"]):
            non_input.append(witness)

    sample = min(
        non_input,
        key=lambda witness: (
            int(witness["basis_term_count"]),
            int(witness["source_gate_index"]),
        ),
        default=None,
    )
    sample_expression: dict[str, object] | None = None
    if sample is not None:
        feature_ids = sample["feature_ids"]
        assert isinstance(feature_ids, list)
        selected = [catalog[int(feature_id)] for feature_id in feature_ids]
        sample_expression = {
            "source_gate_index": sample["source_gate_index"],
            "source_gate": sample["source_gate"],
            "basis_term_count": sample["basis_term_count"],
            "constant": any(feature["kind"] == "constant" for feature in selected),
            "initial_wires": [
                feature["wire"]
                for feature in selected
                if feature["kind"] == "initial_wire"
            ],
            "gate_deltas": [
                {
                    "gate_index": feature["gate_index"],
                    "target_wire": feature["target_wire"],
                }
                for feature in selected
                if feature["kind"] == "gate_delta"
            ],
            "trace_coordinate_upper_bound": sum(
                1 if feature["kind"] == "initial_wire" else 2
                for feature in selected
                if feature["kind"] != "constant"
            ),
        }

    return {
        "witness_schema": "exact-whole-trace-affine-witness/v1",
        "witness_basis_features_by_kind": by_kind,
        "source_gate_firing_witnesses_emitted": len(witnesses),
        "source_gate_firing_witnesses_validated": sum(
            bool(witness["validated_by_exhaustive_replay"])
            for witness in witnesses
        ),
        "non_input_affine_source_firings_with_witness": len(non_input),
        "witness_basis_term_counts": stat_triplet(term_counts),
        "witness_delta_term_counts": stat_triplet(delta_counts),
        "witness_delta_gate_spans": stat_triplet(delta_spans),
        "sample_non_input_affine_witness": sample_expression,
    }


def audit_stage(
    label: str,
    path: Path,
    n: int,
    source: dict[str, object],
    witness_max_total_terms: int | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    track_witnesses = witness_max_total_terms is not None
    with path.open() as handle:
        declared_wires, declared_gates = parse_header(handle.readline(), path)
        wires = effective_wire_count(path, declared_wires, n)
        state, truth_mask, samples = initial_functions(n, wires)
        basis = XorBasis(track_provenance=track_witnesses)
        input_basis = XorBasis()
        catalog: list[FeatureLabel] = []
        if track_witnesses:
            offer_labeled(basis, catalog, truth_mask, {"kind": "constant"})
        else:
            basis.offer(truth_mask)
        input_basis.offer(truth_mask)
        for wire, value in enumerate(state):
            if track_witnesses:
                offer_labeled(
                    basis,
                    catalog,
                    value,
                    {"kind": "initial_wire", "wire": wire},
                )
            else:
                basis.offer(value)
            input_basis.offer(value)
        input_rank = basis.rank
        full_rank_at: int | None = 0 if basis.rank == samples else None
        processed = 0
        for line in handle:
            if not line.strip():
                continue
            target, comp, controls = parse_gate(line)
            delta = firing(state, controls, comp, truth_mask)
            if full_rank_at is None:
                if track_witnesses:
                    offer_labeled(
                        basis,
                        catalog,
                        delta,
                        {
                            "kind": "gate_delta",
                            "gate_index": processed,
                            "target_wire": target,
                        },
                    )
                else:
                    basis.offer(delta)
                if basis.rank == samples:
                    full_rank_at = processed + 1
            state[target] ^= delta
            processed += 1
    if processed != declared_gates:
        raise ValueError(
            f"gate-count mismatch in {path}: header={declared_gates}, parsed={processed}"
        )

    trace_rank = basis.rank
    counts = membership_counts(source, basis, n)
    witness_summary: dict[str, object] = {}
    witness_stage: dict[str, object] | None = None
    if track_witnesses:
        assert witness_max_total_terms is not None
        witnesses = build_witnesses(
            source, basis, input_basis, witness_max_total_terms
        )
        validated = validate_witnesses(
            path, n, wires, catalog, witnesses, source
        )
        recovered = int(counts["source_gate_firings_recovered"])
        if len(witnesses) != recovered or validated != recovered:
            raise AssertionError(
                "constructive witness counts do not match span membership: "
                f"emitted={len(witnesses)} validated={validated} recovered={recovered}"
            )
        witness_summary = summarize_witnesses(catalog, witnesses)
        witness_stage = {
            "label": label,
            "path": str(path),
            "sha256": sha256_file(path),
            "declared_wires": declared_wires,
            "effective_wires": wires,
            "gates": processed,
            "feature_catalog_is_independent_basis": True,
            "feature_catalog": catalog,
            "source_gate_firing_witnesses": witnesses,
            "validation": "exhaustive-truth-table-feature-replay",
        }

    port_basis = XorBasis()
    port_basis.offer(truth_mask)
    initial, _, _ = initial_functions(n, wires)
    for value in initial:
        port_basis.offer(value)
    for value in state:
        port_basis.offer(value)
    port_counts = membership_counts(source, port_basis, n)

    source_final = source["final"]
    assert isinstance(source_final, tuple)
    output_offset = n
    output_matches = wires >= output_offset + n and all(
        state[output_offset + wire] == source_final[wire] for wire in range(n)
    )

    summary = {
        "label": label,
        "path": str(path),
        "declared_wires": declared_wires,
        "wires": wires,
        "gates": processed,
        "truth_table_dimension": samples,
        "input_affine_rank": input_rank,
        "trace_affine_rank": trace_rank,
        "trace_span_is_full_function_space": trace_rank == samples,
        "full_rank_first_gate": full_rank_at,
        "honest_slice_output_matches_source_C_on_wires_n_to_2n": output_matches,
        **counts,
        **witness_summary,
        "port_affine_rank": port_basis.rank,
        "port_interior_state_bits_recovered": port_counts[
            "interior_state_bits_recovered"
        ],
        "port_source_gate_firings_recovered": port_counts[
            "source_gate_firings_recovered"
        ],
    }
    return summary, witness_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for each MPMCT1 stage",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--witness-out",
        type=Path,
        help="optional aggregate sidecar containing explicit basis witnesses",
    )
    parser.add_argument(
        "--witness-max-total-terms",
        type=int,
        default=5_000_000,
        help="fail rather than truncate if emitted witnesses exceed this many terms; 0 is unlimited",
    )
    parser.add_argument(
        "--witness-max-dimension",
        type=int,
        default=4096,
        help="safety cap for provenance tracking; 0 is unlimited",
    )
    args = parser.parse_args()
    if args.n < 1 or args.n > 20:
        raise SystemExit("exact enumeration is intended for 1 <= n <= 20")
    if args.witness_max_total_terms < 0 or args.witness_max_dimension < 0:
        raise SystemExit("witness safety caps must be nonnegative")
    dimension = 1 << args.n
    if (
        args.witness_out
        and args.witness_max_dimension
        and dimension > args.witness_max_dimension
    ):
        raise SystemExit(
            f"--witness-out dimension {dimension} exceeds safety cap "
            f"{args.witness_max_dimension}; raise --witness-max-dimension explicitly"
        )

    source = source_truths(args.source, args.n)
    stages: list[dict[str, object]] = []
    witness_stages: list[dict[str, object]] = []
    for spec in args.stage:
        if "=" not in spec:
            raise SystemExit(f"--stage must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        summary, witness_stage = audit_stage(
            label,
            Path(raw_path),
            args.n,
            source,
            args.witness_max_total_terms if args.witness_out else None,
        )
        if args.witness_out:
            summary["witness_sidecar"] = str(args.witness_out)
            summary["witness_stage_label"] = label
        stages.append(summary)
        if witness_stage is not None:
            witness_stages.append(witness_stage)

    report = {
        "schema": "exact-whole-trace-affine-span/v1",
        "n": args.n,
        "assignments_exhaustively_enumerated": 1 << args.n,
        "source": str(args.source),
        "source_gates": source["gate_count"],
        "feature_definition": "constant + honest-slice initial wires + every physical gate firing delta",
        "stages": stages,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    if args.witness_out:
        witness_report = {
            "schema": "exact-whole-trace-affine-witness/v1",
            "n": args.n,
            "assignments_exhaustively_enumerated": dimension,
            "source": str(args.source),
            "source_sha256": sha256_file(args.source),
            "source_gates": source["gate_count"],
            "feature_semantics": {
                "constant": "affine constant 1",
                "initial_wire": "wire value before physical gate 0",
                "gate_delta": "target value immediately before XOR target value immediately after the physical gate",
            },
            "basis_witnesses_are_minimum_support": False,
            "validation": "exhaustive-truth-table-feature-replay",
            "stages": witness_stages,
        }
        witness_encoded = json.dumps(witness_report, indent=2, sort_keys=True)
        args.witness_out.parent.mkdir(parents=True, exist_ok=True)
        args.witness_out.write_text(witness_encoded + "\n")


if __name__ == "__main__":
    main()
