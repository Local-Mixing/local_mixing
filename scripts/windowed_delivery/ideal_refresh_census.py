#!/usr/bin/env python3
"""Ideal-refresh (carrier migration) ceiling census.

If every band/mask carrier were overwritten (re-masked / migrated)
immediately after its LAST read -- which stage-2 generation or a
band-refresh stage can arrange, and which a post-hoc masking pair
(t ^= c1c2 after the last read, undone before the next real write)
implements function-preservingly -- then a value's minimal bare
residency is the DAG HULL of {birth} u {its readers}:
    hull = fwd-taint(birth) /\ bwd-taint(all readers), + endpoints.
This census measures that hull per carried value (max over its spans)
and reports the hideable ceiling per window cap W. Compare with
interval_census.py (next-writer intervals): the gap is what the stage-2
band-subsystem redesign / value-migration buys. Unread spans get hull
size 1. Same sampling frame as interval_census.py (same seed).

usage: ideal_refresh_census.py A.mpmct1 B.mpmct1 [n_values] [label]
"""
import sys, random, hashlib
from collections import defaultdict

LANES = 16
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
NBYTES = NBITS // 8
SEED = 20261004
SPAN_CAP = 100_000
WS = [10, 20, 30, 40, 64, 80, 160, 320]

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

def wire_class(w):
    return "data" if w < 64 else ("mid" if w < 128 else "band")

def eval_circuit(gates, st, on_born):
    for i, (t, comp, cps, pol) in enumerate(gates):
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        v = st[t] ^ acc
        st[t] = v
        on_born(i, t, v ^ MASK if v & 1 else v)

def hull_size(gates, b, readers, n):
    if not readers:
        return 1
    r_last = max(readers)
    rset = set(readers)
    tb, _, cb, _ = gates[b]
    TW = bytearray(n); TR = bytearray(n)
    TW[tb] = 1
    for c in cb: TR[c] = 1
    fwd = bytearray(r_last - b)
    for idx in range(b + 1, r_last + 1):
        t, _, cps, _ = gates[idx]
        tainted = TR[t] != 0
        if not tainted:
            for c in cps:
                if TW[c]:
                    tainted = True
                    break
        if tainted:
            fwd[idx - b - 1] = 1
            TW[t] = 1
            for c in cps: TR[c] = 1
    TWb = bytearray(n); TRb = bytearray(n)
    cnt = 0
    for idx in range(r_last, b, -1):
        if not fwd[idx - b - 1]:
            continue
        t, _, cps, _ = gates[idx]
        tainted = idx in rset
        if not tainted:
            tainted = TRb[t] != 0
        if not tainted:
            for c in cps:
                if TWb[c]:
                    tainted = True
                    break
        if tainted:
            cnt += 1
            TRb[t] = 1
            for c in cps: TWb[c] = 1
    return cnt + 1          # + b itself

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n_values = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    label = sys.argv[4] if len(sys.argv) > 4 else b_path

    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    rng = random.Random(SEED)
    inputs = [rng.getrandbits(NBITS) for _ in range(n)]

    index, a_wire = {}, {}
    def on_a(i, t, fp):
        if fp == 0:
            return
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        if d not in index:
            index[d] = i
            a_wire[i] = t
    st = list(inputs[:na]) + [0] * (n - na)
    eval_circuit(A, st, on_a)

    m = len(B)
    pending, spans = {}, []
    def on_b(i, t, fp):
        prev = pending.pop(t, None)
        if prev is not None:
            spans.append((prev[0], i, prev[1]))
        if fp == 0:
            return
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        a_id = index.get(d)
        if a_id is not None:
            pending[t] = (i, a_id)
    st = list(inputs[:nb]) + [0] * (n - nb)
    eval_circuit(B, st, on_b)
    forced = set(a for _, a in pending.values())

    rd = defaultdict(list)
    for i, (t, _, cps, _) in enumerate(B):
        for c in cps:
            rd[c].append(i)
    import bisect
    def readers_of(b, e, t):
        rs = rd.get(t, [])
        lo = bisect.bisect_right(rs, b)
        hi = bisect.bisect_left(rs, e)
        return rs[lo:hi]

    by_val = defaultdict(list)
    for b, e, a in spans:
        by_val[a].append((b, e))
    carried = set(by_val) | forced
    cands, censored = [], 0
    for a, ss in by_val.items():
        if a in forced:
            continue
        if max(e - b for b, e in ss) > SPAN_CAP:
            censored += 1
        else:
            cands.append(a)
    print(f"{label}: {m} gates; {len(spans)} spans, {len(carried)} carried, "
          f"{len(forced)} forced, {censored} censored; candidates {len(cands)}")

    sample = rng.sample(cands, min(n_values, len(cands)))
    res = []
    nread_tot = 0
    for a in sample:
        hv = 0
        for b, e in by_val[a]:
            t = B[b][0]
            rr = readers_of(b, e, t)
            nread_tot += len(rr)
            hv = max(hv, hull_size(B, b, rr, n))
        res.append((a, hv))
    hs = sorted(h for _, h in res)
    def pct(p):
        return hs[min(len(hs) - 1, int(p * len(hs)))]
    print(f"sampled {len(res)} values ({nread_tot} reads scanned); "
          f"ideal-refresh hull: med={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={hs[-1]}")
    denom = len(carried)
    scale = len(cands) / max(len(res), 1)
    for W in WS:
        hid = sum(1 for _, h in res if h <= W)
        print(f"W={W:>3}: hideable {100*hid/len(res):.1f}% of sampled "
              f"=> {100*hid*scale/denom:.1f}% of all carried values")
    for cls in ("data", "mid", "band"):
        sub = sorted(h for a, h in res if wire_class(a_wire[a]) == cls)
        if sub:
            line = " ".join(f"W{W}:{100*sum(1 for x in sub if x<=W)/len(sub):.0f}%"
                            for W in WS)
            print(f"  {cls} (n={len(sub)}): {line}")

if __name__ == "__main__":
    main()
