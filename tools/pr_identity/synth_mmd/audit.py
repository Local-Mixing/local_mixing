"""Local bare-state + comeback-degree audit for a synthesized w-wire unit.

Unit is built as inv(M).g.piece (mask M applied on entry).  Bare source value
v ranges over inputs; runtime input to the unit = M(v).  We measure:
  * bare-state-hit: does any INTERIOR cut of the delivered word reproduce the
    full w-wire bare source state v (or the post-g value v'=g(v))?  (the "51%")
  * per-wire exposure count (chain.cut_exposure analog, restricted to v/v').
  * comeback degree: min GF(2) polynomial degree to recover the bare g-wire
    source bits from the unit's I/O (and the algebraic degree of inv(M)).
"""
import numpy as np
import g57

def make_unit(w, lam, rng):
    block = list(range(w))
    def piece():
        tg = list(block)
        while len(tg) < lam: tg.append(rng.choice(block))
        tg = tg[:lam]; rng.shuffle(tg)
        return [(a,) + tuple(rng.sample([u for u in block if u != a], 2)) for a in tg]
    mask = piece()
    a, x, y = rng.sample(block, 3); g = (a, x, y)
    pc = piece()
    unit = list(reversed(mask)) + [g] + pc
    return {"unit": unit, "mask": mask, "g": g, "piece": pc, "w": w}

# ---- bare-state exposure over a pool ----------------------------------------
def bare_audit(word, w, n_anc, mask, g, pool=512, seed=0):
    rng = np.random.default_rng(seed)
    canvas = w + n_anc
    low = (1 << w) - 1
    v = rng.integers(0, 1 << w, size=pool, dtype=np.int64)
    anc = rng.integers(0, 1 << n_anc, size=pool, dtype=np.int64) if n_anc else np.zeros(pool, np.int64)
    Mperm = g57.word_table(mask, w)
    inp_src = Mperm[v]                     # runtime input on source wires = M(v)
    state = (anc << w) | inp_src
    vp = g57.apply_word_batch([g], v)      # post-g bare value
    # bare columns (per source wire), as bit arrays over the pool
    vbits = [((v >> s) & 1) for s in range(w)]
    vpbits = [((vp >> s) & 1) for s in range(w)]
    def full_eq(src_state, target):        # exact full-state match over pool
        return np.array_equal(src_state, target)
    L = len(word)
    full_hit = False; perwire = 0
    hit_cuts = []
    # walk cuts 1..L-1 (interior); cut 0 = masked input, cut L = masked output
    cur = state.copy()
    for t in range(1, L):
        cur = g57.apply_word_batch([word[t - 1]], cur)
        src = cur & low
        if full_eq(src, v) or full_eq(src, vp):
            full_hit = True; hit_cuts.append(t)
        # per-wire exact/complement exposure of v or vp
        for s in range(w):
            col = (src >> s) & 1
            if (np.array_equal(col, vbits[s]) or np.array_equal(col, 1 - vbits[s]) or
                np.array_equal(col, vpbits[s]) or np.array_equal(col, 1 - vpbits[s])):
                perwire += 1
    return {"full_hit": full_hit, "n_full_hit_cuts": len(hit_cuts),
            "perwire_exposures": perwire, "cuts": L - 1}

# ---- algebraic degree (ANF Mobius) ------------------------------------------
def alg_degree(truth):
    """Degree of boolean function given as truth table (len 2^m)."""
    n = len(truth); m = n.bit_length() - 1
    a = truth.copy().astype(np.uint8)
    # Mobius transform in place
    for i in range(m):
        step = 1 << i
        for j in range(0, n, step << 1):
            a[j + step:j + 2 * step] ^= a[j:j + step]
    deg = 0
    for idx in range(n):
        if a[idx]:
            deg = max(deg, bin(idx).count("1"))
    return deg

def invmask_degrees(mask, g, w):
    """Algebraic degree of the map input -> v (=inv(M)(input)) for g's 3 wires
    and for all wires (input-only recovery of the bare value)."""
    invM = g57.word_table(list(reversed(mask)), w)   # inv(M): input -> v
    st = np.arange(1 << w)
    degs = {}
    for s in range(w):
        col = (invM >> s) & 1                        # v[s] as fn of input
        degs[s] = alg_degree(col.astype(np.uint8))
    gw = sorted(set(g))
    return {"g_wire_degrees": {s: degs[s] for s in gw},
            "max_g_wire_deg": max(degs[s] for s in gw),
            "all_max_deg": max(degs.values())}

# ---- comeback: min GF(2) degree to recover bare g-bits from I/O -------------
def _monomials_upto(deg, nv):
    import itertools
    idxs = list(range(nv))
    monos = [()]
    for d in range(1, deg + 1):
        monos += list(itertools.combinations(idxs, d))
    return monos

def _gf2_in_span(cols, target):
    """cols: list of bit-vectors (np uint8 arrays), target: bit-vector.
    True iff target in GF(2) span of cols.  Gaussian elimination."""
    basis = []   # list of (pivot, vector) as python ints for speed
    def to_int(v):
        return int.from_bytes(np.packbits(v).tobytes(), "big")
    rows = [to_int(c) for c in cols]
    tgt = to_int(target)
    piv = {}
    for r in rows:
        x = r
        while x:
            h = x.bit_length() - 1
            if h in piv: x ^= piv[h]
            else: piv[h] = x; break
    x = tgt
    while x:
        h = x.bit_length() - 1
        if h in piv: x ^= piv[h]
        else: return False
    return True

def comeback_io_degree(unit, w, maxdeg=None):
    """Min degree d s.t. each bare g-wire source bit v[wire] is a GF(2)
    polynomial of degree<=d in the 2w I/O bits, over all 2^w inputs.
    Returns the max such min-degree over g's 3 wires (the wall)."""
    if maxdeg is None: maxdeg = w
    g = unit["g"]; mask = unit["mask"]
    P = g57.word_table(unit["unit"], w)
    Mperm = g57.word_table(mask, w)
    invM = g57.word_table(list(reversed(mask)), w)
    st = np.arange(1 << w, dtype=np.int64)
    inp = Mperm[st]                       # input = M(v); here iterate v=st
    v = st
    out = P[inp]                          # output of unit
    # I/O bit variables: w input bits then w output bits
    iobits = []
    for s in range(w): iobits.append(((inp >> s) & 1).astype(np.uint8))
    for s in range(w): iobits.append(((out >> s) & 1).astype(np.uint8))
    nv = 2 * w
    gw = sorted(set(g))
    walls = {}
    for wire in gw:
        target = ((v >> wire) & 1).astype(np.uint8)
        found = None
        for d in range(0, maxdeg + 1):
            monos = _monomials_upto(d, nv)
            cols = []
            for mono in monos:
                if not mono:
                    cols.append(np.ones(1 << w, np.uint8))
                else:
                    acc = iobits[mono[0]].copy()
                    for k in mono[1:]: acc = acc & iobits[k]
                    cols.append(acc)
            if _gf2_in_span(cols, target):
                found = d; break
        walls[wire] = found if found is not None else f">{maxdeg}"
    numeric = [x for x in walls.values() if isinstance(x, int)]
    wall = max(numeric) if len(numeric) == len(walls) else f">{maxdeg}"
    return {"per_g_wire": walls, "wall": wall}
