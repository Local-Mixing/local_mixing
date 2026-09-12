#!/usr/bin/env python3
"""Bounded blind control for short trace-parity relations.

Candidate generation deliberately does not read the source circuit.  It uses
only public zero-slice input samples and a structural cell pattern visible in
an MPMCT1 artifact.  Once the candidate family and its fingerprint are frozen,
the source G57 circuit is loaded solely to score whether any generated parity
equals a source-gate firing.  Matches and source-free zero/constant relations
are replayed on independent validation and locked-test samples.

The candidate family is intentionally bounded rather than exhaustive:

* every detected partition-cell A-gate delta (two raw checkpoints);
* every aligned block of eight such deltas inside an equal-width run; and
* arithmetic progressions of two through six eight-cell blocks, with a bounded
  stride.  Raw checkpoint support is capped exactly after symmetric-difference
  expansion of the deltas.

This is a control experiment, not a minimum-support certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
CHAR_TO_WIRE = {char: index for index, char in enumerate(BASE_CHARS)}


@dataclass(frozen=True)
class Gate:
    target: int
    comp: bool
    controls: tuple[tuple[int, bool], ...]


@dataclass(frozen=True)
class Cell:
    gate_index: int
    before_index: int
    target: int
    width: int


@dataclass(frozen=True)
class Atom:
    stream: int
    ordinal: int
    cells: tuple[int, ...]


def parse_gate(line: str) -> Gate:
    fields = [int(field) for field in line.split()]
    target, comp, width = fields[:3]
    if len(fields) != 3 + 2 * width:
        raise ValueError("MPMCT1 gate width mismatch")
    controls = tuple(
        (fields[index], bool(fields[index + 1]))
        for index in range(3, len(fields), 2)
    )
    return Gate(target, bool(comp), controls)


def read_header(path: Path) -> tuple[int, int]:
    fields = path.open().readline().split()
    if len(fields) != 3 or fields[0] != "mpmct1":
        raise ValueError(f"bad MPMCT1 header in {path}")
    return int(fields[1]), int(fields[2])


def is_partition_cell(window: tuple[tuple[int, Gate, int | None], ...]) -> bool:
    if len(window) != 4:
        return False
    (_, a, _), (_, b, _), (_, c, _), (_, d, _) = window
    if a.comp or b.comp or c.comp or d.comp:
        return False
    if not (a.target == b.target == c.target and d.target != a.target):
        return False
    if not (len(b.controls) == 3 and len(c.controls) == 5 and len(d.controls) == 2):
        return False
    bs, cs, ds = set(b.controls), set(c.controls), set(d.controls)
    return ds < bs and bs < cs


def structural_catalog(path: Path, wires: int) -> tuple[list[Cell], list[list[int]], list[Atom], list[int | None]]:
    """Find cells/atoms using gate syntax only; no circuit evaluation."""
    previous: list[int | None] = [None] * wires
    last: list[int | None] = [None] * wires
    window: deque[tuple[int, Gate, int | None]] = deque(maxlen=4)
    cells: list[Cell] = []
    with path.open() as handle:
        handle.readline()
        for gate_index, line in enumerate(handle):
            if not line.strip():
                continue
            gate = parse_gate(line)
            before = previous[gate.target]
            previous[gate.target] = gate_index
            last[gate.target] = gate_index
            window.append((gate_index, gate, before))
            if len(window) == 4 and is_partition_cell(tuple(window)):
                index, first, first_before = window[0]
                if first_before is not None:
                    cells.append(
                        Cell(index, first_before, first.target, len(first.controls))
                    )

    # Enforce the same endpoint policy as sampled_trace_support.
    cells = [cell for cell in cells if last[cell.target] != cell.gate_index]
    streams: list[list[int]] = []
    for cell_id, cell in enumerate(cells):
        if (
            not streams
            or cells[streams[-1][-1]].target != cell.target
            or cells[streams[-1][-1]].gate_index + 4 != cell.gate_index
        ):
            streams.append([])
        streams[-1].append(cell_id)

    atoms: list[Atom] = []
    for stream_id, stream in enumerate(streams):
        cursor = 0
        ordinal = 0
        while cursor < len(stream):
            width = cells[stream[cursor]].width
            end = cursor + 1
            while end < len(stream) and cells[stream[end]].width == width:
                end += 1
            # Alignment is fixed by the visible maximal equal-width run.
            for start in range(cursor, end - 7, 8):
                atoms.append(
                    Atom(stream_id, ordinal, tuple(stream[start : start + 8]))
                )
                ordinal += 1
            cursor = end
    return cells, streams, atoms, last


def random_inputs(n: int, samples: int, seed: int, domain: int) -> tuple[list[int], int]:
    rng = random.Random(seed ^ domain)
    mask = (1 << samples) - 1
    return [rng.getrandbits(samples) & mask for _ in range(n)], mask


def simulate_selected_deltas(
    path: Path,
    wires: int,
    n: int,
    samples: int,
    seed: int,
    domain: int,
    selected: set[int],
) -> tuple[dict[int, int], int]:
    inputs, mask = random_inputs(n, samples, seed, domain)
    state = [0] * wires
    state[:n] = inputs
    captured: dict[int, int] = {}
    with path.open() as handle:
        handle.readline()
        for gate_index, line in enumerate(handle):
            if not line.strip():
                continue
            gate = parse_gate(line)
            firing = mask
            for wire, positive in gate.controls:
                value = state[wire]
                firing &= value if positive else value ^ mask
            if gate.comp:
                firing ^= mask
            if gate_index in selected:
                captured[gate_index] = firing
            state[gate.target] ^= firing
    if len(captured) != len(selected):
        raise AssertionError((len(captured), len(selected)))
    return captured, mask


def decode_wire(token: str, cursor: int) -> tuple[int, int]:
    overflow = 0
    while cursor < len(token) and token[cursor] == "~":
        overflow += 1
        cursor += 1
    base = CHAR_TO_WIRE[token[cursor]]
    return overflow * 83 + base, cursor + 1


def read_g57(path: Path) -> list[tuple[int, int, int]]:
    gates = []
    for token in path.read_text().strip().split(";"):
        if not token:
            continue
        cursor = 0
        wires = []
        while cursor < len(token):
            wire, cursor = decode_wire(token, cursor)
            wires.append(wire)
        if len(wires) != 3:
            raise ValueError(f"bad G57 token {token!r}")
        gates.append(tuple(wires))
    return gates


def source_firings(path: Path, n: int, samples: int, seed: int, domain: int) -> tuple[list[int], int]:
    # This function is intentionally called only after blind-family freeze.
    inputs, mask = random_inputs(n, samples, seed, domain)
    state = inputs[:]
    out = []
    for target, left, right in read_g57(path):
        firing = mask if left == right else state[left] | (state[right] ^ mask)
        out.append(firing)
        state[target] ^= firing
    return out, mask


def coordinate_support(cell_ids: tuple[int, ...], cells: list[Cell]) -> int:
    coordinates: set[tuple[int, int]] = set()
    for cell_id in cell_ids:
        cell = cells[cell_id]
        for coordinate in (
            (cell.before_index, cell.target),
            (cell.gate_index, cell.target),
        ):
            if coordinate in coordinates:
                coordinates.remove(coordinate)
            else:
                coordinates.add(coordinate)
    return len(coordinates)


def mix64(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def candidate_iter(
    cells: list[Cell],
    atoms: list[Atom],
    signatures: dict[int, int],
    support_cap: int,
    max_step: int,
):
    # Every structurally detected cell delta.
    for cell_id, cell in enumerate(cells):
        yield ("cell", (cell_id,), signatures[cell.gate_index])

    by_stream: dict[int, list[Atom]] = {}
    for atom in atoms:
        by_stream.setdefault(atom.stream, []).append(atom)
    for stream_atoms in by_stream.values():
        stream_atoms.sort(key=lambda atom: atom.ordinal)
        atom_signatures = []
        for atom in stream_atoms:
            signature = 0
            for cell_id in atom.cells:
                signature ^= signatures[cells[cell_id].gate_index]
            atom_signatures.append(signature)
        for atom, signature in zip(stream_atoms, atom_signatures):
            selected = atom.cells
            if 2 * len(selected) <= support_cap:
                yield ("atom-progression-1", selected, signature)
        for start in range(len(stream_atoms)):
            for step in range(1, max_step + 1):
                selected: list[int] = []
                signature = 0
                for length in range(1, 7):
                    index = start + (length - 1) * step
                    if index >= len(stream_atoms):
                        break
                    atom = stream_atoms[index]
                    selected.extend(atom.cells)
                    signature ^= atom_signatures[index]
                    if length == 1:
                        continue
                    selected_tuple = tuple(selected)
                    # These A-gate delta endpoint pairs are disjoint in the
                    # detected cell grammar.  Use the cheap upper bound during
                    # blind enumeration and calculate exact symmetric-
                    # difference support only for zero/source matches.
                    if 2 * len(selected_tuple) <= support_cap:
                        yield (
                            f"atom-progression-{length}",
                            selected_tuple,
                            signature,
                        )


@lru_cache(maxsize=None)
def kind_code(kind: str) -> int:
    return int.from_bytes(hashlib.blake2s(kind.encode(), digest_size=8).digest(), "little")


def candidate_key(kind: str, selected: tuple[int, ...], signature: int) -> int:
    value = len(selected) ^ (signature & ((1 << 64) - 1))
    value ^= (signature >> max(0, signature.bit_length() - 64)) & ((1 << 64) - 1)
    value ^= kind_code(kind)
    if selected:
        value ^= selected[0] << 17
        value ^= selected[-1] << 33
    return mix64(value)


def replay_candidate(
    selected: tuple[int, ...],
    cells: list[Cell],
    signatures: dict[int, int],
) -> int:
    value = 0
    for cell_id in selected:
        value ^= signatures[cells[cell_id].gate_index]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--fit-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=8192)
    parser.add_argument("--support-cap", type=int, default=100)
    parser.add_argument("--max-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=910731)
    parser.add_argument("--max-zero-candidates", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    wires, declared_gates = read_header(args.g)

    # Phase 1a: syntax-only family construction.
    cells, streams, atoms, last = structural_catalog(args.g, wires)
    selected_gate_indices = {cell.gate_index for cell in cells}
    fit_signatures, fit_mask = simulate_selected_deltas(
        args.g,
        wires,
        args.n,
        args.fit_samples,
        args.seed,
        0x424C494E440001,
        selected_gate_indices,
    )

    # Phase 1b: enumerate/fingerprint every candidate and collect source-free
    # zero/constant hypotheses.  The source file has not been opened yet.
    family_count = 0
    family_fingerprint = 0
    zero_candidates: list[dict[str, object]] = []
    fit_zero_or_constant_total = 0
    minimum_blind_zero_support: int | None = None
    for kind, selected, signature in candidate_iter(
        cells, atoms, fit_signatures, args.support_cap, args.max_step
    ):
        family_count += 1
        family_fingerprint ^= candidate_key(kind, selected, signature)
        if signature not in (0, fit_mask):
            continue
        fit_zero_or_constant_total += 1
        support = coordinate_support(selected, cells)
        if minimum_blind_zero_support is None or support < minimum_blind_zero_support:
            minimum_blind_zero_support = support
        if len(zero_candidates) < args.max_zero_candidates:
            zero_candidates.append(
                {
                    "kind": kind,
                    "selected": selected,
                    "fit_constant": signature == fit_mask,
                    "raw_support": support,
                }
            )
    phase1_seconds = time.monotonic() - started

    # Phase 2: only now load source truth for after-the-fact scoring.  Re-run
    # the already frozen deterministic candidate iterator without adaptation.
    targets, _ = source_firings(
        args.source,
        args.n,
        args.fit_samples,
        args.seed,
        0x424C494E440001,
    )
    target_map: dict[int, list[tuple[int, bool]]] = {}
    for target_gate, signature in enumerate(targets):
        target_map.setdefault(signature, []).append((target_gate, False))
        target_map.setdefault(signature ^ fit_mask, []).append((target_gate, True))
    matches: dict[int, dict[str, object]] = {}
    scored_count = 0
    scored_fingerprint = 0
    for kind, selected, signature in candidate_iter(
        cells, atoms, fit_signatures, args.support_cap, args.max_step
    ):
        scored_count += 1
        scored_fingerprint ^= candidate_key(kind, selected, signature)
        for target_gate, constant in target_map.get(signature, ()):
            support = coordinate_support(selected, cells)
            prior = matches.get(target_gate)
            if prior is None or (support, len(selected)) < (
                prior["raw_support"],
                len(prior["selected"]),
            ):
                matches[target_gate] = {
                    "target_gate": target_gate,
                    "kind": kind,
                    "selected": selected,
                    "constant": constant,
                    "raw_support": support,
                }
    if scored_count != family_count or scored_fingerprint != family_fingerprint:
        raise AssertionError("candidate family changed after source was loaded")

    # Independent replay of matches and a bounded set of blind zero/constant
    # hypotheses.  The locked test is evaluated after validation selection.
    hypotheses = list(matches.values()) + zero_candidates
    replay_indices = {
        cells[cell_id].gate_index
        for hypothesis in hypotheses
        for cell_id in hypothesis["selected"]
    }
    validation_signatures, validation_mask = simulate_selected_deltas(
        args.g,
        wires,
        args.n,
        args.validation_samples,
        args.seed,
        0x424C494E440002,
        replay_indices,
    )
    validation_targets, _ = source_firings(
        args.source,
        args.n,
        args.validation_samples,
        args.seed,
        0x424C494E440002,
    )

    validation_matches: list[dict[str, object]] = []
    for hypothesis in matches.values():
        observed = replay_candidate(
            hypothesis["selected"], cells, validation_signatures
        )
        wanted = validation_targets[hypothesis["target_gate"]]
        if hypothesis["constant"]:
            wanted ^= validation_mask
        if observed == wanted:
            validation_matches.append(hypothesis)
    validation_zero: list[dict[str, object]] = []
    for hypothesis in zero_candidates:
        observed = replay_candidate(
            hypothesis["selected"], cells, validation_signatures
        )
        wanted = validation_mask if hypothesis["fit_constant"] else 0
        if observed == wanted:
            validation_zero.append(hypothesis)

    locked_hypotheses = validation_matches + validation_zero
    locked_indices = {
        cells[cell_id].gate_index
        for hypothesis in locked_hypotheses
        for cell_id in hypothesis["selected"]
    }
    test_signatures, test_mask = simulate_selected_deltas(
        args.g,
        wires,
        args.n,
        args.test_samples,
        args.seed,
        0x424C494E440003,
        locked_indices,
    )
    test_targets, _ = source_firings(
        args.source,
        args.n,
        args.test_samples,
        args.seed,
        0x424C494E440003,
    )
    locked_matches = []
    for hypothesis in validation_matches:
        observed = replay_candidate(hypothesis["selected"], cells, test_signatures)
        wanted = test_targets[hypothesis["target_gate"]]
        if hypothesis["constant"]:
            wanted ^= test_mask
        if observed == wanted:
            locked_matches.append(hypothesis)
    locked_zero = []
    for hypothesis in validation_zero:
        observed = replay_candidate(hypothesis["selected"], cells, test_signatures)
        wanted = test_mask if hypothesis["fit_constant"] else 0
        if observed == wanted:
            locked_zero.append(hypothesis)

    def public_hypothesis(hypothesis: dict[str, object]) -> dict[str, object]:
        selected = hypothesis["selected"]
        coordinates = []
        for cell_id in selected:
            cell = cells[cell_id]
            coordinates.extend(
                [
                    f"{cell.before_index}:{cell.target}",
                    f"{cell.gate_index}:{cell.target}",
                ]
            )
        return {
            key: value
            for key, value in hypothesis.items()
            if key != "selected"
        } | {
            "delta_terms": len(selected),
            "coordinates_gate:wire": coordinates,
        }

    report = {
        "schema": "blind-trace-control/v1",
        "g": str(args.g),
        "source_scoring_only": str(args.source),
        "declared_wires": wires,
        "declared_gates": declared_gates,
        "n": args.n,
        "endpoint_policy": "initial coordinates and each wire's final post-write coordinate excluded",
        "candidate_generation_source_blind": True,
        "source_opened_after_family_freeze": True,
        "candidate_family": "partition-cell A deltas; aligned 8-cell equal-width atoms; length 1..6 bounded-stride atom progressions",
        "family_selection_caveat": "post-hoc structural control motivated by the visible partition-cell grammar; candidate signatures themselves are generated and frozen without source truth",
        "detected_cells": len(cells),
        "detected_streams": len(streams),
        "aligned_atoms": len(atoms),
        "support_cap_raw_checkpoints": args.support_cap,
        "max_atom_stride": args.max_step,
        "fit_samples": args.fit_samples,
        "validation_samples": args.validation_samples,
        "locked_test_samples": args.test_samples,
        "seed": args.seed,
        "blind_candidate_count": family_count,
        "blind_family_fingerprint_xor64": f"{family_fingerprint:016x}",
        "blind_phase_seconds": phase1_seconds,
        "fit_zero_or_constant_candidates": fit_zero_or_constant_total,
        "stored_zero_or_constant_candidates": len(zero_candidates),
        "stored_zero_or_constant_candidates_truncated": fit_zero_or_constant_total > len(zero_candidates),
        "minimum_fit_zero_or_constant_support": minimum_blind_zero_support,
        "validation_zero_or_constant_survivors": len(validation_zero),
        "locked_zero_or_constant_survivors": len(locked_zero),
        "source_gates_scored_after_freeze": len(targets),
        "fit_source_gate_matches": len(matches),
        "validation_source_gate_matches": len(validation_matches),
        "locked_source_gate_matches": len(locked_matches),
        "locked_matches": [public_hypothesis(item) for item in locked_matches],
        "locked_zero_or_constant_examples": [
            public_hypothesis(item) for item in locked_zero[:16]
        ],
        "minimum_support_certified": False,
        "interpretation": "bounded blind structural family; non-findings do not cover arbitrary parities",
        "elapsed_seconds": time.monotonic() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
