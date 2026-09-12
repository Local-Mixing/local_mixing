#!/usr/bin/env python3
"""Pairwise structural-similarity profile of a curated-key candidate sample.

Input: lines of `gates<TAB>hexblob` (curated_key_structure --sample output).
For each gate-count bucket, computes pairwise similarity under several metrics
and prints percentile tables, plus each candidate's nearest-neighbor
similarity. Used to compare the raw store against filtered variants: a good
diversity filter drives the nearest-neighbor percentiles down at equal count.

Metrics (all in [0,1], 1 = identical):
  multiset  Jaccard over the bag of normalized gates
  bigram    Jaccard over the bag of adjacent normalized gate pairs
  lcs       longest-common-subsequence ratio over gate token sequences
"""

import sys
from collections import Counter
from itertools import combinations


def norm_gates(blob: bytes):
    gates = []
    for i in range(0, len(blob), 3):
        t, a, b = blob[i], blob[i + 1], blob[i + 2]
        gates.append((t, min(a, b), max(a, b)))
    return gates


def jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 1.0


def lcs_ratio(a, b) -> float:
    n, m = len(a), len(b)
    if not n or not m:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m] / max(n, m)


def percentiles(values, ps=(5, 25, 50, 75, 95, 100)):
    if not values:
        return {}
    values = sorted(values)
    return {p: values[min(len(values) - 1, int(len(values) * p / 100))] for p in ps}


def main():
    path = sys.argv[1]
    buckets = {}
    with open(path) as fh:
        for line in fh:
            gates_s, hexblob = line.split()
            buckets.setdefault(int(gates_s), []).append(bytes.fromhex(hexblob))

    for gates, blobs in sorted(buckets.items()):
        if len(blobs) < 2:
            continue
        seqs = [norm_gates(b) for b in blobs]
        msets = [Counter(s) for s in seqs]
        bigrams = [Counter(zip(s, s[1:])) for s in seqs]
        n = len(seqs)
        nn = {"multiset": [0.0] * n, "bigram": [0.0] * n, "lcs": [0.0] * n}
        pair = {"multiset": [], "bigram": [], "lcs": []}
        for i, j in combinations(range(n), 2):
            sims = {
                "multiset": jaccard(msets[i], msets[j]),
                "bigram": jaccard(bigrams[i], bigrams[j]),
                "lcs": lcs_ratio(seqs[i], seqs[j]),
            }
            for k, v in sims.items():
                pair[k].append(v)
                nn[k][i] = max(nn[k][i], v)
                nn[k][j] = max(nn[k][j], v)

        print(f"\n=== gates={gates}  samples={n} ===")
        print(f"{'metric':>9} {'kind':>9}   p5   p25   p50   p75   p95  p100")
        for k in ("multiset", "bigram", "lcs"):
            for kind, vals in (("pair", pair[k]), ("nearest", nn[k])):
                pct = percentiles(vals)
                row = " ".join(f"{pct[p]:5.2f}" for p in (5, 25, 50, 75, 95, 100))
                print(f"{k:>9} {kind:>9} {row}")


if __name__ == "__main__":
    main()
