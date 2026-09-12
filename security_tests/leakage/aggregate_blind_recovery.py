#!/usr/bin/env python3
"""Pool blind-recovery arms across seeds and measure them against a real null.

Two things a single arm cannot tell you:

* whether an AUC above 0.5 is signal or small-sample noise -- 7 positives
  against 48 nulls is a wide interval, so seeds are pooled and a permutation
  test is run over the pooled labels;
* whether the *procedure* is biased -- so every blind ranking is also scored
  against the source circuits of the OTHER seeds.  Those pairings share the
  construction, the preset and the statistic, and differ only in that the
  labels belong to a different C.  Their AUC is what "no signal" looks like
  for this measurement; if the matched arms do not beat it, there is no
  recovery to report.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path

if __package__:
    from . import score_blind_recovery as scorer
else:
    import score_blind_recovery as scorer


def level1_pairs(source: Path, n: int):
    gates = scorer.read_g57(source)
    state, truth_mask = scorer.initial_state(n)
    pairs = set()
    count = 0
    written: set[int] = set()
    for target, x, y in gates:
        firing = truth_mask if x == y else state[x] | ((~state[y]) & truth_mask)
        if x != y and x not in written and y not in written:
            pairs.add(tuple(sorted((x, y))))
            count += 1
        state[target] ^= firing
        written.add(target)
    return pairs, count


def labelled(rows, statistic, positive_pairs):
    infinity = float("inf")
    out = []
    for row in rows:
        value = row.get(statistic)
        out.append((infinity if value is None else value,
                    tuple(row["wires"]) in positive_pairs))
    return out


def auc_of(labelled_rows):
    positives = [value for value, flag in labelled_rows if flag]
    negatives = [value for value, flag in labelled_rows if not flag]
    return scorer.auc(positives, negatives)


def pooled_auc(all_rows):
    """AUC pooled within arm, then averaged -- each arm weighted equally."""
    values = [auc_of(rows) for rows in all_rows if rows]
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else None, values


def permutation_p(all_rows, observed, trials, seed):
    """Label-reshuffle test, one- and two-sided.

    The two-sided form matters here: an AUC reliably *below* 0.5 is just as
    exploitable as one above it, because the attacker only has to invert the
    ranking.  Only |AUC - 0.5| near zero means no recoverable signal.
    """
    rng = random.Random(seed)
    above = 0
    extreme = 0
    observed_gap = abs(observed - 0.5)
    for _ in range(trials):
        shuffled = []
        for rows in all_rows:
            flags = [flag for _, flag in rows]
            rng.shuffle(flags)
            shuffled.append([(value, flag)
                             for (value, _), flag in zip(rows, flags)])
        mean, _ = pooled_auc(shuffled)
        if mean is None:
            continue
        if mean >= observed:
            above += 1
        if abs(mean - 0.5) >= observed_gap:
            extreme += 1
    return ((above + 1) / (trials + 1), (extreme + 1) / (trials + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True,
                        help="label=blind.json:source.g57 (repeatable)")
    parser.add_argument("--statistic", default="entry_gate")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--out")
    args = parser.parse_args()

    arms = []
    for spec in args.arm:
        label, rest = spec.split("=", 1)
        blind_path, source_path = rest.rsplit(":", 1)
        blind = json.loads(Path(blind_path).read_text())
        if blind.get("source_circuit_read"):
            raise SystemExit(f"{blind_path} reports it read the source")
        rows = [row for row in blind["rows"] if row["family"] == "deg2"]
        pairs, count = level1_pairs(Path(source_path), int(blind["n"]))
        arms.append({"label": label, "blind": blind_path, "source": source_path,
                     "n": int(blind["n"]), "rows": rows, "pairs": pairs,
                     "level1_gates": count, "gates": blind["gates"],
                     "saturates_at": blind.get("trace_span_full_rank_gate")})

    matched = [labelled(arm["rows"], args.statistic, arm["pairs"]) for arm in arms]
    matched_mean, matched_each = pooled_auc(matched)

    mismatched = []
    mismatched_detail = []
    for left, right in combinations(range(len(arms)), 2):
        for a, b in ((left, right), (right, left)):
            rows = labelled(arms[a]["rows"], args.statistic, arms[b]["pairs"])
            mismatched.append(rows)
            mismatched_detail.append({
                "blind_from": arms[a]["label"], "labels_from": arms[b]["label"],
                "auc": auc_of(rows)})
    mismatched_mean, mismatched_each = pooled_auc(mismatched)

    p_one, p_two = permutation_p(matched, matched_mean, args.permutations,
                                 args.seed) if matched_mean is not None \
        else (None, None)

    report = {
        "schema": "blind-recovery-aggregate/v1",
        "statistic": args.statistic,
        "arms": [{"label": arm["label"], "blind": arm["blind"],
                  "source": arm["source"], "gates": arm["gates"],
                  "saturates_at": arm["saturates_at"],
                  "level1_gates": arm["level1_gates"],
                  "level1_pairs": len(arm["pairs"]),
                  "candidates": len(arm["rows"])} for arm in arms],
        "matched": {"mean_auc": round(matched_mean, 4) if matched_mean else None,
                    "per_arm": matched_each},
        "mismatched_null": {
            "mean_auc": round(mismatched_mean, 4) if mismatched_mean else None,
            "pairings": len(mismatched_detail),
            "per_pairing": mismatched_detail},
        "permutation_test": {"trials": args.permutations,
                             "p_one_sided": p_one, "p_two_sided": p_two},
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"statistic         {args.statistic}")
    print(f"arms              {len(arms)}")
    print(f"matched mean AUC  {report['matched']['mean_auc']}   "
          f"per-arm {[round(v, 3) for v in matched_each]}")
    print(f"mismatched null   {report['mismatched_null']['mean_auc']}   "
          f"({len(mismatched_detail)} wrong-C pairings)")
    print(f"permutation p     one-sided {p_one}  two-sided {p_two}  "
          f"({args.permutations} trials)")


if __name__ == "__main__":
    main()
