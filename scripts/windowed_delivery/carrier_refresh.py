#!/usr/bin/env python3
"""Carrier-refresh / value-migration prototype (function-preserving,
post-hoc): break long bare residencies of UNWINNABLE carrier values.

For each span (birth b on wire t, next writer e) of a value the window
packer cannot hide (interval > W_cap or span > its cap), the value is
kept MASKED between its uses: right after b and after each reader, a
CCX pair c1,c2 -> t is inserted (t ^= c1&c2), and it is undone just
before the next reader / before e. While masked, the wire holds
v ^ c1c2 -- a different function (froth to the equality-cluster and
affine-carrier adversaries). The pair (c1,c2) must hold constant values
over its masked stretch; when no pair survives long enough, the mask is
ROTATED (mask new pair, then unmask old -- never bare in between).
Bare residency shrinks to <=1-cut envelopes around b, each read, and e,
all of which are windowable. Mask gates on other spans' wires count as
writes for stability (global insert ledger).

This is the post-hoc form of the stage-2 "values keep moving" redesign;
in stage 2 proper the same discipline is wire-relocation + mask hand-off
at generation time. NOTE the prototype's masks are same-cut-derivable
(degree-2 in boundary values) -- good against the affine/exact-equality
metrics in use, but production masks should be chained band material.

usage: carrier_refresh.py A.mpmct1 B.mpmct1 out.mpmct1 [W_cap] [skip] [label]
"""
import sys, random, hashlib, bisect
from collections import defaultdict

LANES = 16
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
NBYTES = NBITS // 8
SEED = 20261004
SPAN_HARD = 100_000          # spans longer than this are left alone
ROT_CAP = 64
INF = 1 << 60

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

def eval_fp(gates, st, sink=None):
    for i, (t, comp, cps, pol) in enumerate(gates):
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        st[t] ^= acc
        if sink is not None:
            sink(i, t, st[t])

def interval_le(gates, b, e, n, cap):
    """True iff |I(b,e)| <= cap (same taint as window_pack3.interval_set)."""
    TW = bytearray(n); TR = bytearray(n)
    tb, _, cb, _ = gates[b]
    TW[tb] = 1
    for c in cb: TR[c] = 1
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
            for c in cps: TR[c] = 1
    TWb = bytearray(n); TRb = bytearray(n)
    to, _, co, _ = gates[e]
    TRb[to] = 1
    for c in co: TWb[c] = 1
    cnt = 2
    for idx in range(e - 1, b, -1):
        if not fwd[idx - b - 1]:
            continue
        t, _, cps, _ = gates[idx]
        tainted = TWb[t] != 0
        if not tainted:
            for c in cps:
                if TRb[c]:
                    tainted = True
                    break
        if tainted:
            cnt += 1
            if cnt > cap:
                return False
            TRb[t] = 1
            for c in cps: TWb[c] = 1
    return True

def main():
    a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    W_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    SKIP = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    label = sys.argv[6] if len(sys.argv) > 6 else "refresh"
    import os as _os
    seed = int(_os.environ.get("CR_SEED", "0")) or random.SystemRandom().randrange(1 << 30)
    rng = random.Random(seed)

    na, A = load(a_path)
    nb, B = load(b_path)
    n = max(na, nb)
    srng = random.Random(SEED)
    inputs = [srng.getrandbits(NBITS) for _ in range(n)]

    # reference fingerprints
    index = {}
    def norm(v):
        return v ^ MASK if v & 1 else v
    st = list(inputs[:na]) + [0] * (n - na)
    def on_a(i, t, v):
        fp = norm(v)
        if fp:
            d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
            index.setdefault(d, i)
    eval_fp(A, st, on_a)
    final_a = list(st)

    # spans of B
    m = len(B)
    pending, spans = {}, []
    st = list(inputs[:nb]) + [0] * (n - nb)
    def on_b(i, t, v):
        prev = pending.pop(t, None)
        if prev is not None:
            spans.append((prev[0], i, prev[1]))
        fp = norm(v)
        if fp:
            d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
            a_id = index.get(d)
            if a_id is not None:
                pending[t] = (i, a_id)
    eval_fp(B, st, on_b)
    forced = set(a for _, a in pending.values())

    wr_pos = defaultdict(list)
    rd_pos = defaultdict(list)
    for i, (t, _, cps, _) in enumerate(B):
        wr_pos[t].append(i)
        for c in cps:
            rd_pos[c].append(i)

    # winnability at W_cap (same classification as window_pack3)
    by_val = defaultdict(list)
    for b, e, a in spans:
        by_val[a].append((b, e))
    target_vals = []
    for a, ss in by_val.items():
        if a in forced:
            continue
        if max(e - b for b, e in ss) > SPAN_HARD:
            continue
        winnable = all(
            e - b + 1 <= 3000 and (
                interval_le(B, b, e, n, W_cap) or (
                    (lambda ws, k: k + 1 < len(ws) and
                     ws[k + 1] - b + 1 <= 3000 and
                     interval_le(B, b, ws[k + 1], n, W_cap))
                    (wr_pos[B[e][0]], bisect.bisect_left(wr_pos[B[e][0]], e))
                )
            )
            for b, e in ss)
        if not winnable:
            target_vals.append(a)
    import os
    _lim = int(os.environ.get("CR_LIMIT", "0"))
    if _lim:
        target_vals = target_vals[:_lim]
    print(f"{label}: seed={seed}; {len(by_val)} carried values; "
          f"refresh targets (unwinnable @W_cap={W_cap}): {len(target_vals)}")

    # --- plan insertions --------------------------------------------------
    ins_w = defaultdict(list)       # wire -> sorted anchors of inserted writes
    res = defaultdict(list)         # wire -> sorted (lo, hi) anchor ranges where
                                    # the wire is a mask-pair control (no inserted
                                    # writes allowed inside)
    inserts = []                    # (anchor, seq, gate)
    seqc = 0

    def next_write(w, p):
        """(anchor, inserted?) of first write to w strictly after anchor p."""
        ws = wr_pos[w]
        i = bisect.bisect_right(ws, p)
        nw = ws[i] if i < len(ws) else INF
        iw = ins_w[w]
        j = bisect.bisect_right(iw, p)
        niw = iw[j] if j < len(iw) else INF
        return (niw, True) if niw <= nw else (nw, False)

    def res_add(w, lo, hi):
        rs = res[w]
        i = bisect.bisect_right(rs, (lo, INF)) - 1
        if i < 0 or rs[i][1] < lo - 1:
            i += 1
        j = i
        nlo, nhi = lo, hi
        while j < len(rs) and rs[j][0] <= hi + 1:
            nlo = min(nlo, rs[j][0]); nhi = max(nhi, rs[j][1]); j += 1
        rs[i:j] = [(nlo, nhi)]

    def blocked(w, q):
        rs = res[w]
        i = bisect.bisect_right(rs, (q, INF)) - 1
        return i >= 0 and rs[i][1] >= q    # disjoint, so stabbing is exact

    def next_block(w, p):
        rs = res[w]
        i = bisect.bisect_right(rs, (p, INF))
        return rs[i][0] if i < len(rs) else INF

    def pair_horizon(w, p):
        h, ins = next_write(w, p)
        return h - 1 if ins else h      # cannot straddle another insert's anchor

    def pick_pair(t, p, goal):
        best = []
        ws = list(range(nb))
        rng.shuffle(ws)
        for w in ws:
            if w == t:
                continue
            h = pair_horizon(w, p)
            if h > p:
                best.append((h, w))
                if len(best) >= 24:
                    break
        if len(best) < 2:
            return None
        best.sort(reverse=True)
        (h1, c1), (h2, c2) = best[0], best[1]
        h = min(h1, h2, goal)
        if h <= p:
            return None
        return c1, c2, h

    def emit(anchor, t, c1, c2):
        nonlocal seqc
        inserts.append((anchor, seqc, (t, 0, (c1, c2), (1, 1))))
        seqc += 1
        bisect.insort(ins_w[t], anchor)

    nseg = nrot = ngiveup = 0
    nspan_ref = 0
    import os as _os3
    max_spans = int(_os3.environ.get("CR_MAXSPANS", "24"))
    skipped_mega = 0
    tv = []
    for a in target_vals:
        nlong = sum(1 for b, e in by_val[a]
                    if W_cap < e - b + 1 and e - b <= SPAN_HARD)
        if nlong > max_spans:
            skipped_mega += 1
        else:
            tv.append(a)
    print(f"mega values skipped (> {max_spans} long spans): {skipped_mega}; "
          f"refreshing {len(tv)} values")
    for a in tv:
        for b, e in by_val[a]:
            if e - b > SPAN_HARD:
                continue
            if e - b + 1 <= W_cap:
                continue        # interval <= span <= W_cap: already windowable
            t = B[b][0]
            rs = rd_pos.get(t, [])
            lo = bisect.bisect_right(rs, b)
            hi = bisect.bisect_left(rs, e)
            events = [b] + rs[lo:hi] + [e]
            nspan_ref += 1
            for u, v in zip(events, events[1:]):
                if v - u - 1 <= SKIP:
                    continue
                pos = u + 1
                if blocked(t, pos):
                    ngiveup += 1
                    continue
                nb_ = next_block(t, pos)
                # toggles allowed only at anchors STRICTLY before a reservation
                # start (equal-anchor inserts planned later get larger seq and
                # would execute inside the reserved stretch)
                v_eff = min(v, nb_ - 1 if nb_ <= v else v)
                if v_eff - pos <= SKIP:
                    ngiveup += 1
                    continue
                cur = None
                rots = 0
                while pos < v_eff:
                    pk = pick_pair(t, pos - 1, v_eff)
                    if pk is None or pk[2] - pos <= 1:
                        ngiveup += 1
                        break
                    c1, c2, h = pk
                    emit(pos, t, c1, c2)              # mask new
                    if cur is not None:
                        emit(pos, t, cur[0], cur[1])  # unmask old (never bare)
                        nrot += 1
                    res_add(c1, pos, h)
                    res_add(c2, pos, h)
                    cur = (c1, c2)
                    nseg += 1
                    pos = h if h < v_eff else v_eff
                    rots += 1
                    if rots > ROT_CAP:
                        break
                if cur is not None:
                    emit(min(pos, v_eff), t, cur[0], cur[1])   # final unmask
    print(f"planned: {nspan_ref} spans refreshed, {nseg} mask segments, "
          f"{nrot} rotations, {ngiveup} give-ups, {len(inserts)} inserted gates "
          f"({100*len(inserts)/m:.1f}% inflation)")

    # --- rebuild ----------------------------------------------------------
    inserts.sort(key=lambda r: (r[0], r[1]))
    import os as _os2
    if _os2.environ.get("CR_DUMP"):
        with open(_os2.environ["CR_DUMP"], "w") as df:
            for anc, sq, (gt, _, gc, _) in inserts:
                df.write(f"{anc} {sq} {gt} {gc[0]} {gc[1]}\n")
    out = []
    ii = 0
    for gi in range(m + 1):
        while ii < len(inserts) and inserts[ii][0] == gi:
            out.append(inserts[ii][2])
            ii += 1
        if gi < m:
            out.append(B[gi])
    assert ii == len(inserts)

    with open(out_path, "w") as f:
        f.write(f"mpmct1 {nb} {len(out)}\n")
        for t, comp, cps, pol in out:
            parts = [str(t), str(comp), str(len(cps))]
            for c, p in zip(cps, pol):
                parts.append(str(c)); parts.append(str(p))
            f.write(" ".join(parts) + "\n")

    st = list(inputs[:nb]) + [0] * (n - nb)
    eval_fp(out, st)
    agree = sum(1 for w in range(min(na, nb)) if st[w] == final_a[w])
    print(f"wrote {out_path}: {len(out)} gates; function check {agree}/{min(na,nb)} "
          f"({'OK' if agree == min(na,nb) else 'FAIL'})")

if __name__ == "__main__":
    main()
