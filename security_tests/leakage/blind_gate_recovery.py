#!/usr/bin/env python3
"""Blind level-1 gate recovery against a gadgetized/mixed MPMCT1 artifact.

This is the "walk down C using exact linear attacks" sketch, run blind: the
candidate family, the feature catalog and every measurement below are computed
from the mixed artifact alone.  The source G57 circuit is never opened here --
only ``score_blind_recovery.py`` opens it, after this ranking is written and
hashed.

Model (identical to security_tests/leakage/exact_trace_span.py):

* honest slice -- logical inputs occupy wires 0..n, every other wire starts 0;
* features     -- the constant, the initial wire functions, and every physical
                  gate delta (the "flip bit": exactly the conjunction the gate
                  XORs into its target).  Initial wires plus deltas span the
                  same functions as all wire values at all gate boundaries.

Candidate family (source-free).  A G57 gate fires ``x_a OR NOT x_b``, whose
only nonlinear content is the monomial ``x_a x_b``; the constant and ``x_b``
are weight-1 features, so a level-1 gate is recoverable exactly when its
unordered input monomial is.  We rank all ``C(n,2)`` degree-2 input monomials,
and optionally all ``C(n,3)`` degree-3 monomials as the level-2 analogue.

Two blind statistics per candidate:

* ``entry_gate`` -- the smallest prefix of the artifact whose feature span
  already contains the monomial.  Exact, and the discriminating one: it asks
  *when* the mixed circuit first makes that product linearly available.
* ``min_support`` -- exhaustive over weight 1 and 2 across every distinct
  feature value, so a reported 1 or 2 is a true minimum; larger values come
  from the chronological entry basis and are upper bounds only.

Randomized information-set decoding is deliberately NOT used: once the trace
saturates the 2**n function space, a random basis returns weight ~rank/2 for
every candidate alike and separates nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------- artifact io

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


def effective_wire_count(path: Path, declared: int, n: int) -> int:
    """Honour the header unless gate references prove the circuit is wider."""
    maximum = n - 1
    with path.open() as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue
            target, _, controls = parse_gate(line)
            maximum = max(maximum, target, *(wire for wire, _ in controls))
    return max(declared, maximum + 1, n)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_state(n: int, wires: int) -> tuple[list[int], int]:
    samples = 1 << n
    truth_mask = (1 << samples) - 1
    state = [0] * wires
    for wire in range(n):
        value = 0
        for assignment in range(samples):
            if (assignment >> wire) & 1:
                value |= 1 << assignment
        state[wire] = value
    return state, truth_mask


def gate_delta(state, controls, comp: bool, truth_mask: int) -> int:
    value = truth_mask
    for wire, positive in controls:
        source = state[wire]
        value &= source if positive else (~source) & truth_mask
    return ((~value) & truth_mask) if comp else value


def iter_gates(path: Path):
    with path.open() as handle:
        handle.readline()
        for line in handle:
            if line.strip():
                yield parse_gate(line)


def set_bits(value: int):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


# ------------------------------------------------- pass 1: distinct features

def distinct_features(path: Path, n: int, wires: int):
    """Every distinct nonzero feature value, with the first label that made it.

    Duplicates collapse: a repeated value can never lower a minimum support.
    """
    state, truth_mask = initial_state(n, wires)
    index: dict[int, int] = {}
    labels: list[dict] = []

    def add(value: int, label: dict) -> None:
        if value and value not in index:
            index[value] = len(labels)
            labels.append(label)

    add(truth_mask, {"kind": "constant"})
    for wire in range(n):
        add(state[wire], {"kind": "initial_wire", "wire": wire})

    gates = 0
    for target, comp, controls in iter_gates(path):
        delta = gate_delta(state, controls, comp, truth_mask)
        add(delta, {"kind": "gate_delta", "gate_index": gates, "target_wire": target})
        state[target] ^= delta
        gates += 1
    return index, labels, truth_mask, gates


# ------------------------------------------------------ pass 2: entry times

def entry_times(path: Path, n: int, wires: int, candidates, truth_mask: int,
                report_every: int, log):
    """Earliest prefix whose feature span contains each candidate.

    One incremental chronological basis with provenance over accepted
    generators.  Each unresolved candidate keeps its partially reduced residual
    and is bucketed by that residual's leading bit; when a new pivot lands on
    that bit the reduction resumes.  Total reduction work per candidate is
    therefore bounded by the rank, not by the gate count.
    """
    state, _ = initial_state(n, wires)
    full_rank = 1 << n

    rows: dict[int, int] = {}          # pivot -> reduced value
    combos: dict[int, int] = {}        # pivot -> bitset over accepted generators
    generators: list[dict] = []        # accepted generator -> label
    pending: dict[int, list[int]] = {}  # leading bit -> candidate ids
    residual: list[int] = []
    combo_of: list[int] = []
    resolved: list[dict | None] = []

    def advance(cid: int) -> None:
        """Reduce candidate cid as far as the current basis allows, re-bucket."""
        value, combo = residual[cid], combo_of[cid]
        while value:
            pivot = value.bit_length() - 1
            row = rows.get(pivot)
            if row is None:
                residual[cid], combo_of[cid] = value, combo
                pending.setdefault(pivot, []).append(cid)
                return
            value ^= row
            combo ^= combos[pivot]
        residual[cid], combo_of[cid] = 0, combo
        resolved[cid] = {"combo": combo}

    def offer(value: int, label: dict) -> bool:
        combo = 1 << len(generators)
        while value:
            pivot = value.bit_length() - 1
            row = rows.get(pivot)
            if row is None:
                rows[pivot] = value
                combos[pivot] = combo
                generators.append(label)
                for cid in pending.pop(pivot, ()):
                    advance(cid)
                return True
            value ^= row
            combo ^= combos[pivot]
        return False

    for cid, (_, target) in enumerate(candidates):
        residual.append(target)
        combo_of.append(0)
        resolved.append(None)

    entry: list[int | None] = [None] * len(candidates)
    entry_support: list[int | None] = [None] * len(candidates)
    entry_generators: list[list[int] | None] = [None] * len(candidates)

    def harvest(gate_index: int | None) -> None:
        for cid, entry_row in enumerate(resolved):
            if entry_row is not None and entry[cid] is None:
                entry[cid] = -1 if gate_index is None else gate_index
                bits = list(set_bits(entry_row["combo"]))
                entry_support[cid] = len(bits)
                entry_generators[cid] = bits

    offer(truth_mask, {"kind": "constant"})
    for wire in range(n):
        offer(state[wire], {"kind": "initial_wire", "wire": wire})
    for cid in range(len(candidates)):
        advance(cid)
    harvest(None)

    full_rank_gate = None
    gates = 0
    started = time.time()
    for target, comp, controls in iter_gates(path):
        delta = gate_delta(state, controls, comp, truth_mask)
        offer(delta, {"kind": "gate_delta", "gate_index": gates,
                      "target_wire": target})
        state[target] ^= delta
        harvest(gates)
        gates += 1
        if len(rows) >= full_rank:
            full_rank_gate = gates - 1
            log(f"[blind]   trace span saturated at gate {full_rank_gate} "
                f"(rank {len(rows)} = 2^{n}); every candidate is now resolved")
            break
        if report_every and gates % report_every == 0:
            done = sum(1 for value in entry if value is not None)
            log(f"[blind]   gate {gates} rank={len(rows)} resolved={done}/"
                f"{len(candidates)} ({time.time() - started:.0f}s)")

    unresolved = [cid for cid, value in enumerate(entry) if value is None]
    return {
        "entry": entry,
        "entry_support": entry_support,
        "entry_generators": entry_generators,
        "generators": generators,
        "rank": len(rows),
        "full_rank_gate": full_rank_gate,
        "gates_scanned": gates,
        "unresolved": unresolved,
    }


# ---------------------------------------------------------------- candidates

def monomials(n: int, degree: int, state):
    out = []
    for combo in combinations(range(n), degree):
        value = state[combo[0]]
        for wire in combo[1:]:
            value &= state[wire]
        out.append((combo, value))
    return out


def exhaustive_small_support(target: int, index: dict[int, int]):
    """Exact minimum support if it is 1 or 2, else None."""
    hit = index.get(target)
    if hit is not None:
        return [hit]
    for value, position in index.items():
        other = index.get(target ^ value)
        if other is not None:
            return sorted({position, other})
    return None


# ------------------------------------------------------------------- replay

def replay_check(path: Path, n: int, wires: int, truth_mask: int, requests):
    """Independently recompute each requested XOR straight from the artifact.

    ``requests`` maps a key to a list of (kind, wire_or_gate_index) selectors.
    """
    accumulator = {key: 0 for key in requests}
    by_gate: dict[int, list] = {}
    state, _ = initial_state(n, wires)
    for key, selectors in requests.items():
        for kind, value in selectors:
            if kind == "constant":
                accumulator[key] ^= truth_mask
            elif kind == "initial_wire":
                accumulator[key] ^= state[value]
            else:
                by_gate.setdefault(value, []).append(key)

    gates = 0
    for target, comp, controls in iter_gates(path):
        delta = gate_delta(state, controls, comp, truth_mask)
        for key in by_gate.get(gates, ()):
            accumulator[key] ^= delta
        state[target] ^= delta
        gates += 1
    return accumulator


def selectors_from_labels(labels):
    out = []
    for label in labels:
        if label["kind"] == "constant":
            out.append(("constant", 0))
        elif label["kind"] == "initial_wire":
            out.append(("initial_wire", label["wire"]))
        else:
            out.append(("gate_delta", label["gate_index"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", required=True, help="mixed/gadgetized MPMCT1 artifact")
    parser.add_argument("--n", type=int, required=True, help="logical source width")
    parser.add_argument("--out", required=True, help="blind ranking JSON")
    parser.add_argument("--degree3", action="store_true",
                        help="also rank all degree-3 input monomials")
    parser.add_argument("--report-every", type=int, default=2000)
    parser.add_argument("--skip-min-support", action="store_true",
                        help="skip the exhaustive weight<=2 sweep")
    args = parser.parse_args()

    path = Path(args.g)
    started = time.time()

    def log(message: str) -> None:
        print(message, flush=True)

    with path.open() as handle:
        declared, _ = parse_header(handle.readline(), path)
    wires = effective_wire_count(path, declared, args.n)
    log(f"[blind] artifact={path} wires={wires} n={args.n}")

    state, truth_mask = initial_state(args.n, wires)
    candidates = [(("deg2",) + combo, value)
                  for combo, value in monomials(args.n, 2, state)]
    if args.degree3:
        candidates += [(("deg3",) + combo, value)
                       for combo, value in monomials(args.n, 3, state)]
    candidates += [(("deg1", wire), state[wire]) for wire in range(args.n)]
    log(f"[blind] candidates={len(candidates)}")

    timings = {}
    timings["entry_start"] = round(time.time() - started, 1)
    walk = entry_times(path, args.n, wires, candidates, truth_mask,
                       args.report_every, log)
    log(f"[blind] entry pass done: rank={walk['rank']} "
        f"full_rank_gate={walk['full_rank_gate']} "
        f"gates_scanned={walk['gates_scanned']} ({time.time() - started:.0f}s)")

    index: dict[int, int] = {}
    labels: list[dict] = []
    gate_count = walk["gates_scanned"]
    min_support = [None] * len(candidates)
    min_support_labels = [None] * len(candidates)
    if not args.skip_min_support:
        index, labels, truth_mask, gate_count = distinct_features(path, args.n, wires)
        log(f"[blind] gates={gate_count} distinct feature values={len(index)} "
            f"({time.time() - started:.0f}s)")
        for cid, (_, target) in enumerate(candidates):
            witness = exhaustive_small_support(target, index)
            if witness is not None:
                min_support[cid] = len(witness)
                min_support_labels[cid] = [labels[p] for p in witness]
        found = sum(1 for value in min_support if value is not None)
        log(f"[blind] exhaustive weight<=2 minima found for {found}/"
            f"{len(candidates)} candidates ({time.time() - started:.0f}s)")

    requests = {}
    for cid, (key, _) in enumerate(candidates):
        generators = walk["entry_generators"][cid]
        if generators is not None:
            picked = [walk["generators"][g] for g in generators]
            requests[("entry", cid)] = selectors_from_labels(picked)
        if min_support_labels[cid] is not None:
            requests[("min", cid)] = selectors_from_labels(min_support_labels[cid])
    replayed = replay_check(path, args.n, wires, truth_mask, requests)
    for (tag, cid), actual in replayed.items():
        if actual != candidates[cid][1]:
            raise AssertionError(f"{tag} witness replay failed for candidate {cid}")
    log(f"[blind] replay-validated {len(replayed)} witnesses "
        f"({time.time() - started:.0f}s)")

    rows = []
    for cid, (key, _) in enumerate(candidates):
        generators = walk["entry_generators"][cid] or []
        gate_terms = [walk["generators"][g]["gate_index"] for g in generators
                      if walk["generators"][g]["kind"] == "gate_delta"]
        rows.append({
            "family": key[0],
            "wires": list(key[1:]),
            "entry_gate": walk["entry"][cid],
            "entry_support": walk["entry_support"][cid],
            "entry_delta_terms": len(gate_terms),
            "entry_last_delta_gate": max(gate_terms, default=None),
            "min_support": min_support[cid],
            "min_support_is_exact": min_support[cid] is not None,
        })
    rows.sort(key=lambda row: (row["family"],
                               row["entry_gate"] if row["entry_gate"] is not None
                               else 1 << 30,
                               row["wires"]))

    report = {
        "schema": "blind-gate-recovery/v2",
        "source_circuit_read": False,
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "n": args.n,
        "wires": wires,
        "gates": gate_count,
        "gates_scanned_for_entry": walk["gates_scanned"],
        "trace_span_rank": walk["rank"],
        "trace_span_full_rank_gate": walk["full_rank_gate"],
        "distinct_feature_values": len(index) if index else None,
        "unresolved_candidates": len(walk["unresolved"]),
        "caveat": ("entry_gate is exact; min_support is an exhaustive minimum "
                   "only when 1 or 2; entry_support is a chronological-basis "
                   "upper bound, not a minimum-support certificate"),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    log(f"[blind] wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
