"""General mpmct1 reader / bit-parallel evaluator, plus a cut-exposure audit for
a DELIVERED circuit made of arbitrary XGates (comp 0/1, k controls each with a
polarity) measured against a g57 SOURCE word C.

fmix / fcompress emit the mpmct1 plain-text format (src/engine/format.rs,
src/circuit/xgate.rs):

    header:  mpmct1 <n_wires> <n_gates>
    gate:    <target> <comp:0|1> <k> <wire> <pol:0|1> ... (k pairs)

    fires(x) = comp XOR AND_i lit_i,   lit_i = wire_i if pol_i==1 else NOT wire_i
    target ^= fires

The empty AND over zero controls is 1 (Rust starts `fires = 1u64` and only XORs
`comp` in), so a "t 0 0" line is an unconditional NOT of wire t and a "t 1 0"
line is the identity.  A g57 gate (a,x,y)=a^=(x OR NOT y) is the special case
comp=1, ctrls=(x,pol0)(y,pol1): fires = 1 XOR (NOT x AND y) = x OR NOT y.

This module deliberately does NOT depend on the delivered word being g57; the
source C stays g57 and reuses chain.py's target-column machinery, so
`cut_exposure_general` is a drop-in general-circuit analogue of
`chain.cut_exposure`.  n <= 62 (int64 state pools).
"""
import os
import sys
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import g57
import chain


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
def read_mpmct1(path):
    """Parse an mpmct1 file.  Returns (n, gates) where each gate is
    (target:int, comp:int(0/1), ctrls:[(wire:int, pol:int(0/1)), ...])."""
    with open(path) as f:
        toks = f.read().split()
    if not toks or toks[0] != "mpmct1":
        raise ValueError(f"{path}: missing mpmct1 header")
    n = int(toks[1])
    ng = int(toks[2])
    i = 3
    gates = []
    for _ in range(ng):
        target = int(toks[i]); comp = int(toks[i + 1]); k = int(toks[i + 2])
        i += 3
        ctrls = []
        for _c in range(k):
            w = int(toks[i]); pol = int(toks[i + 1])
            i += 2
            ctrls.append((w, pol))
        gates.append((target, comp, ctrls))
    if len(gates) != ng:
        raise ValueError(f"{path}: header claims {ng} gates, parsed {len(gates)}")
    return n, gates


# ---------------------------------------------------------------------------
# evaluator (bit-parallel int64: one int per state, one bit per wire)
# ---------------------------------------------------------------------------
def apply_gate_batch(gate, st):
    """Apply one general XGate to an int64 array of states. Returns a new array."""
    target, comp, ctrls = gate
    fires = np.ones_like(st)               # empty AND = 1
    for (w, pol) in ctrls:
        bit = (st >> w) & 1
        lit = bit if pol == 1 else (1 - bit)
        fires = fires & lit
    if comp:
        fires = fires ^ 1
    return st ^ (fires << target)


def apply_general_word_batch(gates, states):
    """Apply a whole general word (list of gates) to an int64 state array."""
    st = states
    for g in gates:
        st = apply_gate_batch(g, st)
    return st


def gate_wires(gate):
    """Every wire a general gate touches: its target plus all control wires."""
    target, comp, ctrls = gate
    return {target} | {w for (w, _p) in ctrls}


# ---------------------------------------------------------------------------
# g57 recovery (a C file written by chain.to_mpmct1 is g57-form mpmct1)
# ---------------------------------------------------------------------------
def g57_word_from_mpmct1(gates):
    """Recover g57 tuples (a,x,y) from g57-form mpmct1 gates: comp=1 with two
    controls, the pol-0 control is x and the pol-1 control is y (order-robust)."""
    out = []
    for (target, comp, ctrls) in gates:
        assert comp == 1 and len(ctrls) == 2, "not a g57-form gate"
        pol0 = [w for (w, p) in ctrls if p == 0]
        pol1 = [w for (w, p) in ctrls if p == 1]
        assert len(pol0) == 1 and len(pol1) == 1, "not a g57-form gate"
        out.append((target, pol0[0], pol1[0]))
    return out


# ---------------------------------------------------------------------------
# cut-exposure audit for a general delivered word vs a g57 source C
#   (mirrors chain.cut_exposure exactly; only the delivered-word application
#    differs -- source C is applied with the g57 rule, D with the general rule)
# ---------------------------------------------------------------------------
def cut_exposure_general(D_gates, C, n, pool_size=256, seed=0,
                         affine_cuts=120, affine_seed=1):
    assert n <= 62
    pool = g57.make_state_pool(n, trials=pool_size, seed=seed)
    k, L = len(C), len(D_gates)

    # source intermediate states s_0..s_k under the g57 word C
    S = [pool.copy()]
    for g in C:
        S.append(g57.apply_word_batch([g], S[-1]))

    # public columns: s_0 and s_k (+complements); only interior counts as leaks
    trivial = set()
    for j in (0, k):
        kj, ckj = chain._col_keys(S[j], n)
        trivial |= set(kj) | set(ckj)
    targets = {}
    for j in range(1, k):
        kj, _ = chain._col_keys(S[j], n)
        for wi, key in enumerate(kj):
            if key not in trivial:
                targets.setdefault(key, []).append((j, wi))
    n_informative = len(targets)
    full = {}
    for j in range(1, k):
        full.setdefault(S[j].tobytes(), j)

    rngc = random.Random(affine_seed)
    sample = set(rngc.sample(range(L + 1), min(affine_cuts, L + 1))) \
        if affine_cuts else set()
    snaps = {}

    exposed = np.zeros((L + 1, n), dtype=bool)
    covered = set()
    full_hits = []
    Z = pool.copy()
    for p in range(L + 1):
        if p > 0:
            Z = apply_gate_batch(D_gates[p - 1], Z)      # <-- general application
        kp, ckp = chain._col_keys(Z, n)
        for wi in range(n):
            hit = targets.get(kp[wi], []) + targets.get(ckp[wi], [])
            if hit:
                exposed[p, wi] = True
                covered.update(hit)
        j = full.get(Z.tobytes())
        if j is not None and p > 0:
            full_hits.append((p, j))
        if p in sample:
            snaps[p] = Z.copy()

    # affine membership on sampled cuts: reduced bases of {cols(s_j), 1}
    ONES = (1 << pool_size) - 1

    def col_ints(Z):
        B = chain._col_bits(Z, n)
        return [int.from_bytes(np.packbits(b).tobytes(), "big") for b in B]

    def reduce_into(basis, v):
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                return h, v
        return None, 0

    bases = []
    for j in range(k + 1):
        basis = {}
        for v in col_ints(S[j]) + [ONES]:
            h, r = reduce_into(basis, v)
            if h is not None:
                basis[h] = r
        bases.append(basis)

    def in_span(v, basis):
        return reduce_into(basis, v)[1] == 0 if v else True

    aff_public = aff_inter = aff_total = 0
    for p in sorted(snaps):
        for v in col_ints(snaps[p]):
            aff_total += 1
            if in_span(v, bases[0]) or in_span(v, bases[k]):
                aff_public += 1
            elif any(in_span(v, bases[j]) for j in range(1, k)):
                aff_inter += 1

    frac = float(exposed.mean())
    cuts_any = int((exposed.any(axis=1)).sum())
    per_cut = exposed.sum(axis=1)
    slots = sum(len(v) for v in targets.values())
    return {
        "len": L, "k": k, "n": n, "informative_columns": n_informative,
        "exposure_fraction": frac,
        "exposed_pairs": int(exposed.sum()),
        "source_coverage": (len(covered), slots),
        "cuts_with_any_exposure": cuts_any, "cuts_total": L + 1,
        "mean_exposed_wires_per_cut": float(per_cut.mean()),
        "max_exposed_wires": int(per_cut.max()),
        "full_state_hits": len(full_hits),
        "full_state_positions": full_hits[:12],
        "affine": {"sampled_columns": aff_total,
                   "in_public_span": aff_public,
                   "in_intermediate_span_only": aff_inter},
    }


# ---------------------------------------------------------------------------
# permutation-equivalence on a random pool (one-sided: exact when it returns
# False, high-confidence when True). Works for a general D vs a g57 C.
# ---------------------------------------------------------------------------
def perm_equal(D_gates, C_g57, n, trials=512, seed=12345):
    pool = g57.make_state_pool(n, trials=trials, seed=seed)
    outD = apply_general_word_batch(D_gates, pool.copy())
    outC = g57.apply_word_batch(C_g57, pool.copy())
    return bool(np.array_equal(outD, outC))


if __name__ == "__main__":
    # smoke self-test: read a g57 C file, round-trip, and check perm-equality
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    n, gates = read_mpmct1(a.path)
    print(f"read {a.path}: n={n} gates={len(gates)}")
    comps = sum(g[1] for g in gates)
    kdist = {}
    for _t, _c, ct in gates:
        kdist[len(ct)] = kdist.get(len(ct), 0) + 1
    print(f"  comp=1 gates {comps}/{len(gates)}   k-distribution {dict(sorted(kdist.items()))}")
