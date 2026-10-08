#!/usr/bin/env python3
"""Sanity check: is the sandwich's wire split still visible?

Stage-1 output is phase-structured (first half of gates touch only the
top wires, second half only the bottom); values do not hop wires in
stage 2, so the split persists there. This tool measures how much of
it survives any later stage:

  1. positional profile: circuit cut into bins; per bin, the fraction
     of wire-touches landing on the top half (0..n/2-1);
  2. per-wire features (touch count, target fraction, mean/std of
     touch positions, first/last touch) summarized per class;
  3. separability AUC per class pair and feature (rank statistic:
     0.5 = indistinguishable, 1.0 = perfectly split), computed on the
     FULL circuit and on the MIDDLE 80% of positions only (RC's ideal:
     outside the tails, all wires should look the same).

usage: wire_balance.py circuit.mpmct1 [label] [bins]
"""
import sys
from collections import defaultdict

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
            m = max((t,) + cps)
            if m > maxw:
                maxw = m
            gates.append((t, cps))
    return max(n_hdr, maxw + 1), gates

def auc(xs, ys):
    """P(x > y) + 0.5 P(x == y) via ranks."""
    allv = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    rsum = i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for kk in range(i, j):
            if allv[kk][1] == 0:
                rsum += rank
        i = j
    nx, ny = len(xs), len(ys)
    u = rsum - nx * (nx + 1) / 2
    return u / (nx * ny)

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    nb = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    n, gates = load(path)
    m = len(gates)
    half = n // 2

    classes = {
        "top": range(0, half), "bottom": range(half, n),
        "data": range(0, 64), "mid": range(64, min(128, n)),
        "band": range(128, n),
    }
    classes = {k: v for k, v in classes.items() if len(v) > 0}
    cls_of = {}
    for w in range(n):
        cls_of[w] = ("top" if w < half else "bottom",
                     "data" if w < 64 else ("mid" if w < 128 else "band"))

    # positional profile
    binc = [[0, 0] for _ in range(nb)]           # [top, bottom] touches
    bincd = [[0, 0] for _ in range(nb)]          # [data 0-63, mid 64-127] touches
    cnt = defaultdict(int)                       # (w) -> touches (full)
    tgt = defaultdict(int)
    pos_sum = defaultdict(float)
    pos_sq = defaultdict(float)
    first = {}
    last = {}
    cnt_mid = defaultdict(int)                   # middle 80% only
    tgt_mid = defaultdict(int)
    pos_sum_mid = defaultdict(float)
    lo80, hi80 = m // 10, m - m // 10
    for i, (t, cps) in enumerate(gates):
        b = i * nb // m
        p = i / m
        inmid = lo80 <= i < hi80
        for w in (t,) + cps:
            binc[b][0 if w < half else 1] += 1
            if w < 128:
                bincd[b][0 if w < 64 else 1] += 1
            cnt[w] += 1
            pos_sum[w] += p
            pos_sq[w] += p * p
            if w not in first:
                first[w] = p
            last[w] = p
            if inmid:
                cnt_mid[w] += 1
                pos_sum_mid[w] += p
        tgt[t] += 1
        if inmid:
            tgt_mid[t] += 1

    print(f"{label}: {m} gates, {n} wires")
    prof = [c[0] / max(c[0] + c[1], 1) for c in binc]
    print("top-half touch fraction per position bin:")
    print("  " + " ".join(f"{x:.2f}" for x in prof))
    tails = [c[0] + c[1] for c in binc]
    print(f"  (bin mass min/med/max: {min(tails)}/{sorted(tails)[nb//2]}/{max(tails)})")
    profcd = [c[0] / max(c[0] + c[1], 1) for c in bincd]
    print("data-wire (0-63) fraction of sandwich-wire (0-127) touches per bin:")
    print("  " + " ".join(f"{x:.2f}" for x in profcd))

    # per-class feature summaries + AUCs
    def feats(w, midonly):
        if midonly:
            c = cnt_mid.get(w, 0)
            mp = pos_sum_mid[w] / c if c else 0.5
            tf = tgt_mid.get(w, 0) / c if c else 0.0
            return c, mp, tf
        c = cnt.get(w, 0)
        mp = pos_sum[w] / c if c else 0.5
        tf = tgt.get(w, 0) / c if c else 0.0
        return c, mp, tf

    fn = ["touches", "mean-pos", "target-frac"]
    for midonly, tag in ((False, "FULL"), (True, "MIDDLE-80%")):
        print(f"  [{tag}] class stats (med touches, mean-pos p10/med/p90, med pos-std, med target-frac):")
        fv = {w: feats(w, midonly) for w in range(n)}
        for cname, ws in classes.items():
            ws = list(ws)
            mt = sorted(fv[w][0] for w in ws)[len(ws) // 2]
            mps = sorted(fv[w][1] for w in ws)
            tf = sorted(fv[w][2] for w in ws)[len(ws) // 2]
            stds = []
            for w in ws:
                c = cnt.get(w, 0)
                if c:
                    mu = pos_sum[w] / c
                    stds.append(max(pos_sq[w] / c - mu * mu, 0.0) ** 0.5)
                else:
                    stds.append(0.0)
            stds.sort()
            print(f"    {cname:>6}: {mt:>8.0f}  "
                  f"{mps[len(ws)//10]:.3f}/{mps[len(ws)//2]:.3f}/{mps[9*len(ws)//10]:.3f}  "
                  f"{stds[len(ws)//2]:.3f}  {tf:.3f}")
        for a, b in (("top", "bottom"), ("data", "band"),
                     ("data", "mid"), ("mid", "band")):
            if a not in classes or b not in classes:
                continue
            line = []
            for fi in range(3):
                xa = [fv[w][fi] for w in classes[a]]
                xb = [fv[w][fi] for w in classes[b]]
                line.append(f"{fn[fi]}={auc(xa, xb):.3f}")
            print(f"    AUC {a} vs {b}: " + "  ".join(line))

if __name__ == "__main__":
    main()
