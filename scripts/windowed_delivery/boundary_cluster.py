#!/usr/bin/env python3
"""Equality-cluster attack on a WINDOWED deliverable's boundary values.

The adversary of a windowed artifact sees only the wire states at window
boundaries (plus inputs/outputs). Its linkage graph clusters boundary
values by function-equality (complement-normalized fingerprints over
shared random inputs), exactly like the born-value cluster attack on
gate-level circuits (RESULTS.md, 2026-10-03: TDP proxy clusters up to
135 = band plumbing; froth ceiling ~19 after s6 packing; purity
threshold >=6 at s3 / >=20 at s6).

A "run" = a maximal stretch of consecutive boundaries where a wire holds
the same value (trivially linked; counted once). Cluster size of a value
= number of runs holding it. Clusters are split proxy (value matches a
reference born value) vs froth.

usage: boundary_cluster.py A.mpmct1 packed.mpmct1 packed.bounds [label]
"""
import sys, random, hashlib
from collections import defaultdict

LANES = 16
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
NBYTES = NBITS // 8
SEED = 20261004

def load(path):
    with open(path) as f:
        hdr = f.readline().split()
        n_hdr = int(hdr[1])
        gates = []
        maxw = 0
        for line in f:
            q = line.split()
            if not q:
                continue
            t, comp, k = int(q[0]), int(q[1]), int(q[2])
            cps = tuple(int(q[3 + 2*i]) for i in range(k))
            pol = tuple(int(q[4 + 2*i]) for i in range(k))
            m = max((t,) + cps)
            if m > maxw:
                maxw = m
            gates.append((t, comp, cps, pol))
    return max(n_hdr, maxw + 1), gates

def main():
    a_path, b_path, bounds_path = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else "bcluster"
    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    m = len(B)
    bounds = sorted(set(int(x) for x in open(bounds_path) if x.strip()) | {m})
    rng = random.Random(SEED)
    inputs = [rng.getrandbits(NBITS) for _ in range(n)]

    def digest(v):
        fp = v ^ MASK if v & 1 else v
        if fp == 0:
            return None                 # constants: degenerate, skip
        return hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()

    # reference fingerprints
    a_index = set()
    st = list(inputs[:na]) + [0] * (n - na)
    for t, comp, cps, pol in A:
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        st[t] ^= acc
        d = digest(st[t])
        if d:
            a_index.add(d)

    # stream packed circuit; digest only wires dirty since last boundary
    st = list(inputs[:nb]) + [0] * (n - nb)
    last = [digest(v) for v in st]      # input boundary seeds the runs
    counts = defaultdict(int)
    for d in last:
        if d:
            counts[d] += 1
    lo = 0
    dirty = set()
    for cut in bounds:
        for t, comp, cps, pol in B[lo:cut]:
            acc = MASK
            for c, p in zip(cps, pol):
                acc &= st[c] if p else st[c] ^ MASK
            if comp:
                acc ^= MASK
            st[t] ^= acc
            dirty.add(t)
        for w in dirty:
            d = digest(st[w])
            if d != last[w]:
                last[w] = d
                if d:
                    counts[d] += 1
        dirty.clear()
        lo = cut

    prox = [c for d, c in counts.items() if d in a_index]
    froth = [c for d, c in counts.items() if d not in a_index]
    prox.sort(); froth.sort()
    tot_runs = sum(counts.values())
    print(f"{label}: {len(bounds)} boundaries; {tot_runs} runs, "
          f"{len(counts)} distinct values ({len(prox)} proxy / {len(froth)} froth)")
    for name, arr in (("proxy", prox), ("froth", froth)):
        if arr:
            k = len(arr)
            print(f"  {name}: med={arr[k//2]} p99={arr[99*k//100]} "
                  f"max={arr[-1]}; clusters>=2: {sum(1 for c in arr if c >= 2)}")
    for s in (2, 3, 5, 6, 10, 20):
        p = sum(1 for c in prox if c >= s)
        f = sum(1 for c in froth if c >= s)
        if p + f:
            print(f"  clusters >= {s}: {p+f} total, proxy precision "
                  f"{100*p/(p+f):.1f}% (proxy {p} / froth {f})")

if __name__ == "__main__":
    main()
