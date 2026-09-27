"""Instrumented copy of chain.compile_mono (backend='mmd') that also returns a
per-gate REGION tag, so the cut-exposure audit can attribute every exposed cut
to head / co-blocked-unit / raw-multi-block-unit / scheduled-refresh / tail.

The body mirrors chain.compile_mono line-for-line (same helper calls, same RNG
call order) so the emitted word D is BIT-IDENTICAL to chain.compile_mono with
the same rng -- verified by assertion in __main__.  Region semantics:

  head      the leading mask word (input is public)
  coblk     a co-blocked gate's mono-SYNTHESIZED unit  (born-bare-free claim)
  refresh   a scheduled-refresh mono-SYNTHESIZED unit  (born-bare-free claim)
  raw       a non-co-blocked gate's RAW multi-block unit (bare interior -- the
            honest residual)
  tail      the trailing unmask word (output is public)

acc is assumed 0 (mono-mmd config): no accretion words, so a co-blocked era's
gates are ALL synthesized.  (The instrumented builder asserts acc==0.)
"""
import random

import chain
from chain import (_piece, inv_word, _route_partition, mono_synth_mmd)


def compile_mono_mmd_tagged(C, n, rng, w=6, lam=6, refresh=6, n_anc=None):
    acc = 0
    assert n % w == 0 and lam >= w and w >= 3
    k = len(C)
    NANC = max(0, w - 3) if n_anc is None else n_anc
    ancilla_pool = list(range(n, n + NANC))
    n_total = n + NANC
    wires = list(range(n))
    rng.shuffle(wires)
    P = [wires[i:i + w] for i in range(0, n, w)]
    B = len(P)
    masks = [_piece(b, lam, rng) for b in P]
    last_seen = [0] * B
    D = []
    region = []
    support = []           # per-gate: frozenset of wires the gate's UNIT touches
    coblocked = raw_units = mono_units = refreshed = mono_fail = 0

    def emit(gates, tag):
        supp = frozenset(w for gg in gates for w in gg)
        D.extend(gates)
        region.extend([tag] * len(gates))
        support.extend([supp] * len(gates))

    for m in [list(mm) for mm in masks]:   # head: one mask word per block
        emit(m, "head")

    def synth(word):
        nonlocal mono_units, mono_fail
        s, ok = mono_synth_mmd(word, ancilla_pool)
        mono_units += ok
        mono_fail += not ok
        return s

    for j in range(k):
        g = C[j]
        idx = {wi: bi for bi, b in enumerate(P) for wi in b}
        involved = {idx[wi] for wi in set(g)}
        era = {}          # key -> (gates, tag)
        if len(involved) == 1:
            coblocked += 1
            b = next(iter(involved))
            piece = _piece(P[b], lam, rng)
            raw = inv_word(masks[b]) + [g] + piece
            masks[b] = piece
            era[b] = (list(synth(raw)), "coblk")
            last_seen[b] = j
        else:
            inv = sorted(involved)
            W = [u for bi in inv for u in P[bi]]
            newb = _route_partition(W, w, C, j, rng)
            u = [gg for bi in inv for gg in inv_word(masks[bi])] + [g]
            P2 = [list(b) for b in P]
            for bi, nb in zip(inv, newb):
                P2[bi] = nb
            for bi in inv:
                masks[bi] = _piece(P2[bi], lam, rng)
                u += masks[bi]
                last_seen[bi] = j
            era[("multi", tuple(inv))] = (u, "raw")
            raw_units += 1
            P = P2
        if refresh:
            for bi in range(B):
                if bi not in involved and j - last_seen[bi] >= refresh:
                    piece = _piece(P[bi], lam, rng)
                    ref = inv_word(masks[bi]) + piece
                    masks[bi] = piece
                    era[bi] = (list(synth(ref)), "refresh")
                    refreshed += 1
                    last_seen[bi] = j
        # acc == 0: no accretion branch
        items = list(era.values())
        rng.shuffle(items)
        for (gates, tag) in items:
            emit(gates, tag)
    tail = [inv_word(m) for m in masks]
    rng.shuffle(tail)
    for u in tail:
        emit(u, "tail")
    assert len(region) == len(D) == len(support)
    return {"D": D, "region": region, "support": support, "n_total": n_total,
            "coblocked_frac": round(coblocked / max(1, k), 3),
            "mono_units": mono_units, "mono_fail": mono_fail,
            "raw_units": raw_units, "refreshed": refreshed}


if __name__ == "__main__":
    # verify the tagged builder reproduces chain.compile_mono EXACTLY
    for seed in (1, 2, 3):
        for kk in (24, 48):
            C = chain.local_circuit(64, kk, random.Random(seed))
            r1 = chain.compile_mono(C, 66, random.Random(seed * 1000 + 7),
                                    w=6, lam=6, acc=0, refresh=6, backend="mmd")
            r2 = compile_mono_mmd_tagged(C, 66, random.Random(seed * 1000 + 7),
                                         w=6, lam=6, refresh=6)
            ok = (r1["D"] == r2["D"])
            print(f"seed={seed} k={kk}: len {len(r1['D'])} identical={ok} "
                  f"coblk={r2['coblocked_frac']} raw={r2['raw_units']} "
                  f"refr={r2['refreshed']} fail={r2['mono_fail']}", flush=True)
            assert ok, "tagged builder diverged from chain.compile_mono"
    print("MONO_TAGGED: reproduces chain.compile_mono exactly")
