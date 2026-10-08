#!/usr/bin/env python3
"""Alive-carrier / waist census for ANF-window boundary placement.

For the windowed-ANF delivery design: a delivered artifact made of ANF
windows exposes only the wire states at window BOUNDARIES. A reference
(A, pre-mixed) born value is exposed at a boundary cut iff some wire of
the candidate circuit (B) carries that exact value (up to complement)
across the cut -- i.e. the cut falls inside a proxy span (birth ..
next-write on that wire). This census measures, per candidate:

  1. alive(k) = number of proxy spans covering each cut k (exact, full
     resolution, no sampling of cuts);
  2. the span-length distribution of proxy carriers (windowable mass);
  3. optimal boundary placement by DP under a window cap W (every window
     <= W gates), minimizing total boundary exposure; compared against
     uniform boundaries every W gates (the waist gain);
  4. distinct-A-value exposure: how many distinct reference values are
     carried across >= 1 chosen boundary, vs carried anywhere, vs the
     hideable ceiling (all spans shorter than W).

Function-equality is fingerprinted on LANES*64 shared random inputs,
complement-normalized (first sample bit), exactly like proxy_census.py.

usage: waist_census.py A.mpmct1 B.mpmct1 [W1,W2,...] [label]
"""
import sys, random, hashlib
from collections import defaultdict
from array import array

LANES = 16                      # 16*64 = 1024 random samples
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
            cps = [(int(q[3 + 2*i]), int(q[4 + 2*i])) for i in range(k)]
            m = max([t] + [c for c, _ in cps])
            if m > maxw:
                maxw = m
            gates.append((t, comp, cps))
    return max(n_hdr, maxw + 1), gates

def wire_class(w):
    return "data" if w < 64 else ("mid" if w < 128 else "band")

def eval_circuit(n, gates, st, on_born):
    """st: list of bigint lanes per wire (mutated). on_born(i, t, fp) for
    each gate's complement-normalized born fingerprint (0 = constant)."""
    for i, (t, comp, cps) in enumerate(gates):
        acc = MASK
        for c, pol in cps:
            acc &= st[c] if pol else st[c] ^ MASK
        if comp:
            acc ^= MASK
        v = st[t] ^ acc
        st[t] = v
        on_born(i, t, v ^ MASK if v & 1 else v)

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    caps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [10, 20, 40, 80]
    label = sys.argv[4] if len(sys.argv) > 4 else b_path

    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    rng = random.Random(SEED)
    inputs = [rng.getrandbits(NBITS) for _ in range(n)]

    # --- pass 1: A fingerprints ---------------------------------------
    index = {}                  # digest -> a_id (first A gate with that value)
    a_wire = {}                 # a_id -> A target wire
    a_nonconst = a_dup = 0
    def on_a(i, t, fp):
        nonlocal a_nonconst, a_dup
        if fp == 0:
            return
        a_nonconst += 1
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        if d in index:
            a_dup += 1
        else:
            index[d] = i
            a_wire[i] = t
    st_a = list(inputs[:na]) + [0] * (n - na)
    eval_circuit(na, A, st_a, on_a)
    print(f"A {a_path}: {len(A)} gates, {na} wires; "
          f"{a_nonconst} nonconst born values, {len(index)} distinct ({a_dup} repeats)")

    # --- pass 2: stream B, build proxy spans ---------------------------
    m = len(B)
    pending = {}                # b wire -> (birth_gate, a_id) for live proxy value
    spans = []                  # (first_cut, last_cut, a_id)  cuts in 1..m
    b_nonconst = 0
    def on_b(i, t, fp):
        nonlocal b_nonconst
        prev = pending.pop(t, None)
        if prev is not None:
            spans.append((prev[0] + 1, i, prev[1]))
        if fp == 0:
            return
        b_nonconst += 1
        d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
        a_id = index.get(d)
        if a_id is not None:
            pending[t] = (i, a_id)
    st_b = list(inputs[:nb]) + [0] * (n - nb)
    eval_circuit(nb, B, st_b, on_b)
    final_proxies = [(bg + 1, m, a_id) for bg, a_id in pending.values()]
    spans.extend(final_proxies)         # alive at the output boundary (forced)

    # sanity: function preservation on shared wires
    agree = sum(1 for w in range(min(na, nb)) if st_a[w] == st_b[w])
    print(f"B {label}: {m} gates, {nb} wires; {b_nonconst} nonconst born; "
          f"function check: {agree}/{min(na,nb)} wires agree")

    nprox = len(spans)
    distinct_carried = len({a for _, _, a in spans})
    cls_spans = defaultdict(int)
    cls_vals = defaultdict(set)
    for _, _, a in spans:
        c = wire_class(a_wire[a])
        cls_spans[c] += 1
        cls_vals[c].add(a)
    print(f"proxy spans: {nprox} ({100*nprox/max(b_nonconst,1):.1f}% of born), "
          f"distinct A values carried: {distinct_carried} "
          f"({100*distinct_carried/max(len(index),1):.1f}% of A distinct)")
    print("  by A wire class: " + "  ".join(
        f"{c}: {cls_spans[c]} spans / {len(cls_vals[c])} vals"
        for c in ("data", "mid", "band")))

    # span lengths (number of cuts covered)
    lens = sorted(e - s + 1 for s, e, _ in spans)
    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))] if lens else 0
    print(f"span length (cuts): med={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1] if lens else 0}")
    for W in caps:
        frac = sum(1 for L in lens if L <= W - 1) / max(len(lens), 1)
        print(f"  spans windowable at W={W}: {100*frac:.1f}%")

    # per-value max span length -> hideable ceiling
    vmax = defaultdict(int)
    interior_vals = set()
    for s, e, a in spans:
        L = e - s + 1
        if L > vmax[a]:
            vmax[a] = L
        if e < m:
            interior_vals.add(a)
    forced_vals = {a for _, e, a in spans if e == m}   # on an output wire: always exposed

    # --- alive(k) over interior cuts 1..m-1 ----------------------------
    diff = array("i", [0]) * 0
    diff = array("i", bytes(4 * (m + 2)))
    for s, e, _ in spans:
        diff[s] += 1
        if e + 1 <= m + 1:
            diff[e + 1] -= 1
    alive = array("i", bytes(4 * (m + 1)))
    run = 0
    for k in range(1, m + 1):
        run += diff[k]
        alive[k] = run
    interior = alive[1:m]
    tot = sum(interior)
    mean_alive = tot / max(m - 1, 1)
    srt = sorted(interior)
    print(f"alive(cut): mean={mean_alive:.2f} min={srt[0]} p10={srt[len(srt)//10]} "
          f"med={srt[len(srt)//2]} p90={srt[9*len(srt)//10]} max={srt[-1]}; "
          f"output boundary (forced): {len(final_proxies)} spans / "
          f"{len(forced_vals)} distinct values")

    # spans sorted for boundary->value queries
    import bisect
    spans_by_start = sorted(spans)
    starts = [s for s, _, _ in spans_by_start]

    def exposed_values(bounds):
        out = set()
        for s, e, a in spans_by_start:
            i = bisect.bisect_left(bounds, s)
            if i < len(bounds) and bounds[i] <= e:
                out.add(a)
        return out

    # --- DP: optimal boundaries under window cap W ----------------------
    from collections import deque
    for W in caps:
        dp = array("q", bytes(8 * (m + 1)))
        par = array("i", bytes(4 * (m + 1)))
        dq = deque([0])         # indices j with increasing dp
        for i in range(1, m + 1):
            while dq and dq[0] < i - W:
                dq.popleft()
            j = dq[0]
            cost = alive[i] if i < m else 0
            dp[i] = dp[j] + cost
            par[i] = j
            while dq and dp[dq[-1]] >= dp[i]:
                dq.pop()
            dq.append(i)
        bounds = []
        i = m
        while i > 0:
            if i < m:
                bounds.append(i)
            i = par[i]
        bounds.reverse()
        nb_ = len(bounds)
        opt_cost = dp[m]
        # uniform control
        uni = list(range(W, m, W))
        uni_cost = float(sum(alive[k] for k in uni))
        exp_vals = exposed_values(bounds)
        uni_vals = exposed_values(uni)
        hideable = sum(1 for a in interior_vals
                       if vmax[a] <= W - 1 and a not in forced_vals)
        denom = max(distinct_carried, 1)
        print(f"W={W:>3}: boundaries={nb_} "
              f"| mean alive@boundary opt={opt_cost/max(nb_,1):.2f} uniform={uni_cost/max(len(uni),1):.2f} grand-mean={mean_alive:.2f} "
              f"| distinct A values exposed opt={len(exp_vals)} ({100*len(exp_vals)/denom:.1f}%) "
              f"uniform={len(uni_vals)} ({100*len(uni_vals)/denom:.1f}%) "
              f"| hideable ceiling (all spans<W, not output): {100*hideable/denom:.1f}%")
        for c in ("data", "mid", "band"):
            ev = sum(1 for a in exp_vals if wire_class(a_wire[a]) == c)
            cv = len(cls_vals[c])
            if cv:
                print(f"        {c}: exposed {ev}/{cv} ({100*ev/cv:.1f}%)")

if __name__ == "__main__":
    main()
