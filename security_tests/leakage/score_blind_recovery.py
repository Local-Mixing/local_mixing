#!/usr/bin/env python3
"""Score a frozen blind ranking against the source circuit it never read.

``blind_gate_recovery.py`` ranks every degree-2 input monomial by two blind
statistics -- the earliest artifact prefix whose trace span contains it
(``entry_gate``) and the shortest witness found (``min_support``) -- using only
the mixed artifact.  This stage opens the source G57 circuit for the first time
and asks whether either ranking separates the monomials C actually uses at its
input boundary from the ones it does not.

A level-1 gate of C is one whose firing is still a function of raw inputs --
both of its controls are wires C has not yet written.  Its firing is
``x_a OR NOT x_b = 1 XOR x_b XOR x_a x_b``, so the unordered pair {a, b} is the
positive label; every other pair is a null of exactly the same shape, measured
by exactly the same procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path


BASE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
CHAR_TO_WIRE = {char: index for index, char in enumerate(BASE_CHARS)}


def decode_wire(token: str, cursor: int) -> tuple[int, int]:
    overflow = 0
    while cursor < len(token) and token[cursor] == "~":
        overflow += 1
        cursor += 1
    if cursor >= len(token):
        raise ValueError(f"missing base character after '~' in {token!r}")
    return overflow * 83 + CHAR_TO_WIRE[token[cursor]], cursor + 1


def read_g57(path: Path) -> list[tuple[int, int, int]]:
    gates = []
    for token in path.read_text().strip().split(";"):
        if not token:
            continue
        wires, cursor = [], 0
        while cursor < len(token):
            wire, cursor = decode_wire(token, cursor)
            wires.append(wire)
        if len(wires) != 3:
            raise ValueError(f"G57 gate does not contain three wires: {token!r}")
        gates.append(tuple(wires))
    return gates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_state(n: int) -> tuple[list[int], int]:
    samples = 1 << n
    truth_mask = (1 << samples) - 1
    state = []
    for wire in range(n):
        value = 0
        for assignment in range(samples):
            if (assignment >> wire) & 1:
                value |= 1 << assignment
        state.append(value)
    return state, truth_mask


def auc(positive, negative):
    """P(random positive ranks strictly better than random null), ties 0.5.

    Smaller statistic = better rank, so 'better' means a smaller value.
    0.5 is no signal; 1.0 is a perfect separator.
    """
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for q in negative:
            if p < q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return round(wins / (len(positive) * len(negative)), 4)


def summary(values):
    finite = sorted(v for v in values if v != float("inf"))
    return {
        "count": len(values),
        "unresolved": sum(1 for v in values if v == float("inf")),
        "min": finite[0] if finite else None,
        "median": finite[len(finite) // 2] if finite else None,
        "max": finite[-1] if finite else None,
    }


def evaluate(rows, statistic, positive_pairs):
    infinity = float("inf")
    scored = []
    for row in rows:
        value = row.get(statistic)
        scored.append({
            "pair": row["wires"],
            "value": value,
            "score": infinity if value is None else value,
            "is_level1_pair": tuple(row["wires"]) in positive_pairs,
        })
    positives = [row["score"] for row in scored if row["is_level1_pair"]]
    negatives = [row["score"] for row in scored if not row["is_level1_pair"]]
    ordered = sorted(scored, key=lambda row: (row["score"], row["pair"]))
    k = len(positives)
    top_k = ordered[:k]
    best = ordered[0]["score"] if ordered else None
    tied = [row for row in ordered if row["score"] == best]
    return {
        "statistic": statistic,
        "positives": summary(positives),
        "nulls": summary(negatives),
        "auc_positive_vs_null": auc(positives, negatives),
        "precision_at_k": (sum(row["is_level1_pair"] for row in top_k) / k)
                          if k else None,
        "natural_cut": {
            "value": best if best != infinity else None,
            "selected": len(tied),
            "of_which_level1": sum(row["is_level1_pair"] for row in tied),
            "precision": (sum(row["is_level1_pair"] for row in tied) / len(tied))
                         if tied else None,
        },
        "ranking": [{"pair": row["pair"], "value": row["value"],
                     "is_level1_pair": row["is_level1_pair"]} for row in ordered],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", required=True, help="blind ranking JSON")
    parser.add_argument("--c", required=True, help="source G57 circuit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    blind = json.loads(Path(args.blind).read_text())
    if blind.get("source_circuit_read"):
        raise SystemExit("refusing to score: blind stage reports it read the source")
    n = int(blind["n"])

    source_path = Path(args.c)
    gates = read_g57(source_path)
    state, truth_mask = initial_state(n)

    level1 = []
    written: set[int] = set()
    for gate_index, (target, x, y) in enumerate(gates):
        firing = truth_mask if x == y else state[x] | ((~state[y]) & truth_mask)
        if x != y and x not in written and y not in written:
            level1.append({"gate_index": gate_index, "target": target,
                           "positive_control": x, "negative_control": y,
                           "pair": sorted((x, y))})
        state[target] ^= firing
        written.add(target)

    positive_pairs = {tuple(entry["pair"]) for entry in level1}
    rows = [row for row in blind["rows"] if row["family"] == "deg2"]
    expected = n * (n - 1) // 2
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} degree-2 rows, got {len(rows)}")

    anchors = [row for row in blind["rows"] if row["family"] == "deg1"]
    anchor_supports = sorted({row["min_support"] for row in anchors})

    report = {
        "schema": "blind-gate-recovery-score/v2",
        "label": args.label,
        "blind_report": args.blind,
        "blind_artifact": blind["artifact"],
        "blind_artifact_sha256": blind["artifact_sha256"],
        "blind_artifact_gates": blind["gates"],
        "trace_span_full_rank_gate": blind.get("trace_span_full_rank_gate"),
        "source_circuit": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_gates": len(gates),
        "n": n,
        "level1_gates": level1,
        "level1_pair_count": len(positive_pairs),
        "candidate_pairs": len(rows),
        "degree1_anchor_min_supports": anchor_supports,
        "by_statistic": {
            statistic: evaluate(rows, statistic, positive_pairs)
            for statistic in ("entry_gate", "min_support", "entry_support")
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"source={source_path.name} gates={len(gates)} n={n}")
    print(f"level-1 gates of C: {len(level1)}  distinct pairs: {len(positive_pairs)}"
          f"  of {len(rows)} candidates")
    for statistic, block in report["by_statistic"].items():
        print(f"\n[{statistic}]")
        print(f"  positives {block['positives']}")
        print(f"  nulls     {block['nulls']}")
        print(f"  AUC={block['auc_positive_vs_null']}  "
              f"precision@k={block['precision_at_k']}")
        print(f"  natural cut {block['natural_cut']}")


if __name__ == "__main__":
    main()
