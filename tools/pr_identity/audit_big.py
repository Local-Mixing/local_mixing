"""Big-integer cut-exposure audit -- the int64-free analogue of
`chain.cut_exposure` / `mpmct1.cut_exposure_general`, correct for ANY n.

chain.cut_exposure and mpmct1.cut_exposure_general pack the delivered-circuit
state pool into int64s (one int per state, one bit per wire), so they cap at
n <= 62.  The MMD backend of compile_mono lifts a width-w unit onto w-3 DIRTY
ancilla wires placed at n..n_total-1, so an n=64 delivered circuit lives on up
to 67 wires -- past the int64 ceiling.

This module keeps the SAME audit semantics but uses a TRANSPOSED, big-integer
column representation: instead of `pool_size` states each holding n wire-bits,
we keep, per wire, one `pool_size`-bit Python int whose bit j is that wire's
value in pool state j.  A gate is then a couple of big-int ops on those columns
(independent of n), and a wire's "column key" at a cut is simply its column int
-- so the walk is O(len * pool_size / 64) machine-word ops and any n works.

Accepts BOTH gate encodings in the delivered word (arity-detected per gate):
  * a g57 tuple (a, x, y)                       fires  x OR NOT y   (a ^= fires)
  * a general XGate (target, comp, [(w,pol)..]) fires  comp XOR AND_i lit_i
The source word C is always g57 (a, x, y) tuples.

Source targets are C's INTERIOR intermediates on wires 0..n-1 (public s_0, s_k
and their complements excluded); ancilla wires n..n_total-1 are never source
targets.  Every delivered wire 0..n_total-1 is scanned for a match, so an
ancilla that transiently materialises a source value is still caught.

Equality of results with chain.cut_exposure is EXACT (verified in __main__): the
transposed encoding is a fixed permutation of the pool-coordinate axis, which
preserves column equality, GF(2)-span membership, and the constant vector.
"""
import random

import numpy as np

import g57


# ---------------------------------------------------------------------------
# state pool as Python big-ints
# ---------------------------------------------------------------------------
def _make_pool_ints(n, pool_size, seed):
    """A fixed pool of `pool_size` random n-bit states, as Python ints.

    For n <= 62 this reuses g57.make_state_pool so the pool is BIT-IDENTICAL to
    chain.cut_exposure's pool (exact cross-check).  For n > 62 (where numpy
    int64 overflows on the high wire bits) we draw n-bit ints directly."""
    if n <= 62:
        st = g57.make_state_pool(n, trials=pool_size, seed=seed)
        return [int(v) for v in st.tolist()]
    rng = random.Random((seed << 1) ^ 0x9E3779B97F4A7C15)
    return [rng.getrandbits(n) for _ in range(pool_size)]


def _cols_from_states(states, W):
    """Transpose `states` (list of ints, one per pool slot) into W wire-columns:
    cols[w] is the pool_size-bit int whose bit j = (states[j] >> w) & 1."""
    cols = [0] * W
    for w in range(W):
        col = 0
        for j, s in enumerate(states):
            if (s >> w) & 1:
                col |= (1 << j)
        cols[w] = col
    return cols


# ---------------------------------------------------------------------------
# gate application on the transposed columns (in place)
# ---------------------------------------------------------------------------
def _is_general(g):
    """A g57 gate is (a, x, y) all ints; a general XGate is (target, comp, ctrls)
    with ctrls a (possibly empty) list/tuple of (wire, pol) pairs."""
    return isinstance(g[2], (list, tuple))


def _apply_gate_cols(g, cols, MASK):
    if _is_general(g):
        target, comp, ctrls = g
        fires = MASK                      # empty AND over the pool = all-ones
        for (w, pol) in ctrls:
            fires &= cols[w] if pol == 1 else (cols[w] ^ MASK)
        if comp:
            fires ^= MASK
        cols[target] ^= fires
    else:                                 # g57: a ^= (x OR NOT y)
        a, x, y = g
        cols[a] ^= cols[x] | (cols[y] ^ MASK)


# ---------------------------------------------------------------------------
# GF(2) span helpers (identical algebra to chain.cut_exposure)
# ---------------------------------------------------------------------------
def _reduce_into(basis, v):
    while v:
        h = v.bit_length() - 1
        if h in basis:
            v ^= basis[h]
        else:
            return h, v
    return None, 0


def _in_span(v, basis):
    return _reduce_into(basis, v)[1] == 0 if v else True


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def cut_exposure_big(D_gates, C_g57_word, n, n_total=None, pool_size=256,
                     cut_subsample=None, affine_cuts=120, seed=0, affine_seed=1):
    """Measure materialisation of C's interior intermediates at cuts of D.

    D_gates : delivered word, g57 tuples and/or general XGates (mixed OK).
    C_g57_word : source word, g57 tuples.
    n       : source width (interior source targets live on wires 0..n-1).
    n_total : delivered width (>= n); defaults to n.  Ancilla wires n..n_total-1
              are excluded from source targets but ARE scanned for exposure.
    cut_subsample : if set to K, evaluate exact/complement (and full-state)
              exposure on K evenly-spaced cuts only (for very long circuits);
              source_coverage is then honestly labelled "over sampled cuts".
              None => every cut (exact match to chain.cut_exposure).
    affine_cuts : GF(2)-affine membership sampled on this many random cuts
              (independent of cut_subsample, same RNG contract as chain).

    Returns the same dashboard dict as chain.cut_exposure (plus n_total,
    coverage_scope, cut_subsample, checked_cuts)."""
    W = n if n_total is None else n_total
    assert W >= n
    MASK = (1 << pool_size) - 1
    pool = _make_pool_ints(n, pool_size, seed)
    k, L = len(C_g57_word), len(D_gates)

    # --- source intermediates s_0..s_k as column-sets on wires 0..n-1 ---
    cur = _cols_from_states(pool, n)
    S = [list(cur)]
    for g in C_g57_word:
        _apply_gate_cols(g, cur, MASK)         # C is g57
        S.append(list(cur))

    # public columns: s_0 and s_k (+complements); only interior counts as leaks
    trivial = set()
    for j in (0, k):
        for c in S[j]:
            trivial.add(c)
            trivial.add(c ^ MASK)
    targets = {}                               # source col int -> [(j, wi), ...]
    for j in range(1, k):
        for wi, c in enumerate(S[j]):
            if c not in trivial:
                targets.setdefault(c, []).append((j, wi))
    n_informative = len(targets)
    full = {}                                  # full interior state -> j
    for j in range(1, k):
        full.setdefault(tuple(S[j]), j)

    # affine bases {cols(s_j), 1} for every j
    bases = []
    for j in range(k + 1):
        basis = {}
        for v in S[j] + [MASK]:
            h, r = _reduce_into(basis, v)
            if h is not None:
                basis[h] = r
        bases.append(basis)

    # which cuts to evaluate exact/complement + full-state on
    if cut_subsample is None:
        check = None                           # all cuts
        scope = "all_cuts"
    else:
        K = min(cut_subsample, L + 1)
        check = set(int(round(t)) for t in np.linspace(0, L, K)) if K > 1 else {0}
        scope = "sampled_cuts"

    # affine cut sample (independent; same contract as chain.cut_exposure)
    rngc = random.Random(affine_seed)
    sample = set(rngc.sample(range(L + 1), min(affine_cuts, L + 1))) \
        if affine_cuts else set()
    snaps = {}

    # --- delivered walk ---
    cols = _cols_from_states(pool, W)          # s_0 on 0..n-1, ancilla = 0
    exposed_pairs = 0
    covered = set()
    cuts_with_any = 0
    per_cut_max = 0
    num_checked = 0
    full_hits = []
    for p in range(L + 1):
        if p > 0:
            _apply_gate_cols(D_gates[p - 1], cols, MASK)
        if check is None or p in check:
            num_checked += 1
            cnt = 0
            for wi in range(W):
                c = cols[wi]
                h1 = targets.get(c)
                h2 = targets.get(c ^ MASK)
                if h1 or h2:
                    cnt += 1
                    if h1:
                        covered.update(h1)
                    if h2:
                        covered.update(h2)
            exposed_pairs += cnt
            if cnt:
                cuts_with_any += 1
                if cnt > per_cut_max:
                    per_cut_max = cnt
            j = full.get(tuple(cols[w] for w in range(n)))
            if j is not None and p > 0:
                full_hits.append((p, j))
        if p in sample:
            snaps[p] = [cols[w] for w in range(n)]

    # affine membership on sampled cuts
    aff_public = aff_inter = aff_total = 0
    for p in sorted(snaps):
        for v in snaps[p]:
            aff_total += 1
            if _in_span(v, bases[0]) or _in_span(v, bases[k]):
                aff_public += 1
            elif any(_in_span(v, bases[j]) for j in range(1, k)):
                aff_inter += 1

    slots = sum(len(v) for v in targets.values())
    denom = max(1, num_checked * W)
    return {
        "len": L, "k": k, "n": n, "n_total": W,
        "informative_columns": n_informative,
        "exposure_fraction": exposed_pairs / denom,
        "exposed_pairs": exposed_pairs,
        "source_coverage": (len(covered), slots),
        "coverage_scope": scope,
        "cut_subsample": cut_subsample,
        "checked_cuts": num_checked,
        "cuts_with_any_exposure": cuts_with_any, "cuts_total": L + 1,
        "mean_exposed_wires_per_cut": exposed_pairs / max(1, num_checked),
        "max_exposed_wires": per_cut_max,
        "full_state_hits": len(full_hits),
        "full_state_positions": full_hits[:12],
        "affine": {"sampled_columns": aff_total,
                   "in_public_span": aff_public,
                   "in_intermediate_span_only": aff_inter},
    }


def cut_exposure_by_region(D_gates, region, C_g57_word, n, n_total,
                           pool_size=256, seed=0, supports=None):
    """Same exposure walk as cut_exposure_big, but every exposed (cut,wire) at
    cut p (p>=1) is attributed to region[p-1] -- the region of the gate that
    produced that delivered state.  `region` is a per-gate tag list.

    If `supports` (per-gate frozenset of the wires the gate's UNIT touches) is
    given, each exposure is further split ACTIVE (exposed wire is in the current
    unit's support -- a wire the unit is actually computing on) vs PARKED (an
    idle wire that a previous era left at a value matching a source column, held
    constant while this long unit churns).  The born-bare claim is about ACTIVE
    exposures inside coblk/refresh regions."""
    assert len(region) == len(D_gates)
    W = n_total
    MASK = (1 << pool_size) - 1
    pool = _make_pool_ints(n, pool_size, seed)
    k, L = len(C_g57_word), len(D_gates)

    cur = _cols_from_states(pool, n)
    S = [list(cur)]
    for g in C_g57_word:
        _apply_gate_cols(g, cur, MASK)
        S.append(list(cur))
    trivial = set()
    for j in (0, k):
        for c in S[j]:
            trivial.add(c)
            trivial.add(c ^ MASK)
    targets = {}
    for j in range(1, k):
        for wi, c in enumerate(S[j]):
            if c not in trivial:
                targets.setdefault(c, []).append((j, wi))
    slots = sum(len(v) for v in targets.values())

    regions = sorted(set(region)) + ["<all>"]
    exp = {r: 0 for r in regions}
    active = {r: 0 for r in regions}       # exposed pairs on an active unit wire
    parked = {r: 0 for r in regions}       # exposed pairs on an idle/parked wire
    cov = {r: set() for r in regions}
    cov_active = {r: set() for r in regions}
    cuts = {r: 0 for r in regions}

    cols = _cols_from_states(pool, W)
    for p in range(L + 1):
        if p > 0:
            _apply_gate_cols(D_gates[p - 1], cols, MASK)
        r = region[p - 1] if p > 0 else "head"
        supp = supports[p - 1] if (supports is not None and p > 0) else None
        cnt = 0
        for wi in range(W):
            c = cols[wi]
            h1 = targets.get(c)
            h2 = targets.get(c ^ MASK)
            if h1 or h2:
                cnt += 1
                slotshit = (h1 or []) + (h2 or [])
                cov[r].update(slotshit)
                cov["<all>"].update(slotshit)
                if supp is not None and wi in supp:
                    active[r] += 1
                    active["<all>"] += 1
                    cov_active[r].update(slotshit)
                    cov_active["<all>"].update(slotshit)
                elif supp is not None:
                    parked[r] += 1
                    parked["<all>"] += 1
        if cnt:
            exp[r] += cnt
            exp["<all>"] += cnt
            cuts[r] += 1
            cuts["<all>"] += 1
    return {"slots": slots,
            "by_region": {r: {"exposed_pairs": exp[r],
                              "active_pairs": active[r],
                              "parked_pairs": parked[r],
                              "cuts_with_exposure": cuts[r],
                              "covered_slots": len(cov[r]),
                              "active_covered_slots": len(cov_active[r])}
                          for r in regions},
            "region_gate_counts": {r: region.count(r)
                                   for r in sorted(set(region))}}


def print_exposure_big(rep, label):
    a = rep["affine"]
    at = max(1, a["sampled_columns"])
    print(f"--- cut exposure (big): {label}  (len {rep['len']}, k={rep['k']}, "
          f"n={rep['n']}, n_total={rep['n_total']}, "
          f"informative cols {rep['informative_columns']}) ---")
    cov, slots = rep["source_coverage"]
    print(f"  exact/comp exposure    : {rep['exposure_fraction']*100:6.2f}% of (cut,wire)   "
          f"cuts touched {rep['cuts_with_any_exposure']}/{rep['checked_cuts']}   "
          f"mean/max wires per cut {rep['mean_exposed_wires_per_cut']:.2f}/{rep['max_exposed_wires']}")
    print(f"  absolute / coverage    : {rep['exposed_pairs']} exposed (cut,wire) pairs   "
          f"source values covered {cov}/{slots}"
          f" = {cov/max(1,slots)*100:.0f}%  ({rep['coverage_scope']})")
    print(f"  full-state bare cuts   : {rep['full_state_hits']}"
          + (f"   at {rep['full_state_positions']}" if rep["full_state_hits"] else ""))
    print(f"  affine (sampled)       : public-span {a['in_public_span']/at*100:.1f}%   "
          f"intermediate-span-only {a['in_intermediate_span_only']/at*100:.1f}%")


# ---------------------------------------------------------------------------
# self-verification: EXACT match vs chain.cut_exposure on 3 small circuits
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random as _random
    import chain
    from atoms import Resynth

    _KEYS = ("exposure_fraction", "exposed_pairs", "source_coverage",
             "full_state_hits", "cuts_with_any_exposure", "cuts_total",
             "mean_exposed_wires_per_cut", "max_exposed_wires",
             "informative_columns")

    def _cmp(name, D, C, n):
        ref = chain.cut_exposure(D, C, n)
        got = cut_exposure_big(D, C, n)
        ok = all(ref[key] == got[key] for key in _KEYS) \
            and ref["affine"] == got["affine"]
        print(f"[{name}] len {len(D)}  n={n}  match={'OK' if ok else 'FAIL'}")
        for key in _KEYS:
            flag = "" if ref[key] == got[key] else "   <-- MISMATCH"
            print(f"    {key:32s} ref={ref[key]!s:24s} big={got[key]!s:24s}{flag}")
        print(f"    {'affine':32s} ref={ref['affine']} big={got['affine']}"
              + ("" if ref["affine"] == got["affine"] else "   <-- MISMATCH"))
        return ok

    all_ok = True

    # 1. brick2 D (wire-local source, lookahead routing)
    rng = _random.Random(7)
    n = 24
    C = chain.local_circuit(n, 20, rng)
    res = chain.compile_brick2(C, n, rng, w=6, lam=6, acc=3, lookahead=True)
    all_ok &= _cmp("brick2", res["D"], C, n)

    # 2. mono-ball D  (cheap w=4 radius-4 ball; the cross-check only needs *a*
    # mono-ball delivered word, not the production config)
    rng = _random.Random(11)
    n = 20
    C = chain.local_circuit(n, 18, rng)
    res = chain.compile_mono(C, n, rng, w=4, lam=6, acc=0, refresh=6,
                             backend="ball", rs=Resynth(max_wires=4, radius=4))
    all_ok &= _cmp("mono-ball", res["D"], C, n)

    # 3. raw random word (not equal to C -- audit doesn't require equality)
    rng = _random.Random(13)
    n = 32
    C = chain.random_circuit(n, 24, rng)
    D = chain.rand_word(list(range(n)), 400, rng)
    all_ok &= _cmp("raw-random", D, C, n)

    print()
    print("AUDIT_BIG SELF-VERIFY:", "ALL MATCH" if all_ok else "MISMATCH ABOVE")
