#!/usr/bin/env python3
"""Reorder-aware hideability census: DAG-interval sizes of proxy carriers.

waist_census.py measured carrier spans in the DELIVERED gate order; those
are upper bounds. Under commuting reorder, a born value v (birth gate b,
wire t, next t-writer o) has a bare lifetime no smaller than the DAG
interval I(b,o) = {g : b <= g <= o} in the conflict partial order
(g1 -> g2 iff target(g1) in controls(g2) [RAW] or target(g2) in
controls(g1) [WAR]; same-target pairs commute). An ANF window hides v iff
it contains all of I(b,o) for some admissible first-next-writer o (v1:
o = the delivered-order next writer). So |I(b,o)| <= W is the
reorder-aware hideable ceiling at window cap W.

A distinct A value is hidden only if ALL its carrier spans are hidden, so
we sample distinct carried values and take max over their spans.

usage: interval_census.py A.mpmct1 B.mpmct1 [n_values] [label]
"""
import sys, random, hashlib
from collections import defaultdict

LANES = 16
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
NBYTES = NBITS // 8
SEED = 20261004
SPAN_CAP = 100_000          # spans longer than this: censored (not hideable at any sane W)
WS = [10, 20, 30, 40, 80, 160, 320]

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

def interval_size(gates, b, e, n):
    """|{g : b <= g <= o}| - 2 interior, where o = gates[e]; scan b..e."""
    # forward taint from b
    TW = bytearray(n)   # wire has a tainted writer so far
    TR = bytearray(n)   # wire has a tainted reader so far
    tb, _, cb, _ = gates[b]
    TW[tb] = 1
    for c in cb:
        TR[c] = 1
    fwd = bytearray(e - b - 1)
    for idx in range(b + 1, e):
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
            for c in cps:
                TR[c] = 1
    # backward taint from o
    TWb = bytearray(n)  # wire is a control of a tainted later gate (RAW back)
    TRb = bytearray(n)  # wire is the target of a tainted later gate (WAR back)
    to, _, co, _ = gates[e]
    TRb[to] = 1
    for c in co:
        TWb[c] = 1
    cnt = 0
    for idx in range(e - 1, b, -1):
        if not fwd[idx - b - 1]:
            continue        # only forward-tainted gates can be in the interval
        t, _, cps, _ = gates[idx]
        tainted = TWb[t] != 0
        if not tainted:
            for c in cps:
                if TRb[c]:
                    tainted = True
                    break
        if tainted:
            cnt += 1
            TRb[t] = 1
            for c in cps:
                TWb[c] = 1
    return cnt + 2          # + b and o themselves

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    n_values = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    label = sys.argv[4] if len(sys.argv) > 4 else b_path

    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    rng = random.Random(SEED)
    inputs = [rng.getrandbits(NBITS) for _ in range(n)]

    index = {}
    a_wire = {}
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
    pending = {}
    spans = []
    def on_b(i, t, fp):
        prev = pending.pop(t, None)
        if prev is not None:
            spans.append((prev[0], i, prev[1]))   # (birth gate, next-writer gate, a_id)
        if fp == 0:
            return
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        a_id = index.get(d)
        if a_id is not None:
            pending[t] = (i, a_id)
    st = list(inputs[:nb]) + [0] * (n - nb)
    eval_circuit(B, st, on_b)
    forced = set(a for _, a in pending.values())   # alive at output: never hideable

    by_val = defaultdict(list)
    for b, e, a in spans:
        by_val[a].append((b, e))
    carried = set(by_val) | forced
    print(f"{label}: {m} gates; {len(spans)} interior proxy spans, "
          f"{len(carried)} distinct carried values, {len(forced)} output-forced")

    # candidate values: not output-forced, all spans <= SPAN_CAP
    cands, censored = [], 0
    for a, ss in by_val.items():
        if a in forced:
            continue
        if max(e - b for b, e in ss) > SPAN_CAP:
            censored += 1
        else:
            cands.append(a)
    print(f"candidates (no forced span, all spans <= {SPAN_CAP}): {len(cands)}; "
          f"long-span censored values: {censored} "
          f"(treated as not hideable)")

    sample = rng.sample(cands, min(n_values, len(cands)))
    results = []        # (a_id, max interval over spans, max span, n spans)
    work = 0
    for a in sample:
        ss = by_val[a]
        iv = 0
        sp = 0
        for b, e in ss:
            work += 2 * (e - b)
            iv = max(iv, interval_size(B, b, e, n))
            sp = max(sp, e - b + 1)
        results.append((a, iv, sp, len(ss)))
    print(f"sampled {len(results)} values, {work/1e6:.0f}M gate-visits")

    ivs = sorted(r[1] for r in results)
    sps = sorted(r[2] for r in results)
    def pct(arr, p):
        return arr[min(len(arr) - 1, int(p * len(arr)))]
    print(f"max-span per value: med={pct(sps,.5)} p90={pct(sps,.9)}; "
          f"max-INTERVAL per value: med={pct(ivs,.5)} p90={pct(ivs,.9)} "
          f"(compression med x{pct(sps,.5)/max(pct(ivs,.5),1):.1f})")

    # population-level hideable ceiling: sample fraction scaled to candidates,
    # censored + forced count as exposed
    denom = len(carried)
    base_hidden_scale = len(cands) / max(len(results), 1)
    for W in WS:
        hid = sum(1 for _, iv, _, _ in results if iv <= W)
        est = hid * base_hidden_scale
        print(f"W={W:>3}: hideable (reorder-aware) {100*hid/max(len(results),1):.1f}% of sampled "
              f"=> {100*est/denom:.1f}% of all carried values "
              f"(vs delivered-order ceiling from waist census)")
    for cls in ("data", "mid", "band"):
        sub = [iv for a, iv, _, _ in results if wire_class(a_wire[a]) == cls]
        if sub:
            sub.sort()
            line = " ".join(f"W{W}:{100*sum(1 for x in sub if x<=W)/len(sub):.0f}%" for W in WS)
            print(f"  {cls} (n={len(sub)}): {line}")

if __name__ == "__main__":
    main()
