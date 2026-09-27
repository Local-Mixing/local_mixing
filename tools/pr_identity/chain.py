"""Leg-chain compiler: CCMP-style gate-by-gate block replacement with
*cancelled boundary legs*, plus the cut-exposure audit that measures whether
the source circuit's intermediate states materialize at any cut of the
delivered circuit.

Construction (the [CCMP TCC'24] shape, upgraded):

  Input C = g_1 g_2 ... g_k on n wires.  Draw secret "leg" words A_2..A_k
  (A_1 = A_{k+1} = empty).  Block i is *not* equivalent to g_i alone --
  it realizes  inv(A_i) . g_i . A_{i+1},  so blocks telescope:

      D  =  K_1 K_2 ... K_k  ==  C     (all legs cancel in the product)

  while the state at every block boundary is A_{i+1}(s_i) -- a masked image
  of the source intermediate s_i, never s_i itself.  With uniformly random
  leg permutations this is exactly Kilian-style randomization of the gate
  chain: the boundary transitions inv(A_i).g_i.A_{i+1} reveal nothing beyond
  the composed function.  Legs here are short secret g57 words, so the claim
  degrades from information-theoretic to "pseudo"; the audit below is the
  instrument that measures the degradation.

  The danger is *interior* cuts: a raw realization  inv(A_i) g_i A_{i+1}
  materializes the bare s_{i-1} at the cut after its inv(A_i) part.  Arms:

    naive    blocks equivalent to g_i alone (legs = empty) -- the base
             construction and the measured-broken control (dense-cut result).
    randleg  full-width random legs, raw concatenation, then stagger
             (random adjacent commuting swaps) + window launder.  Statistical
             blending only -- bare interior cuts exist pre-stagger.
    brick    group-local legs: each era mask is a product of independent
             pieces on the blocks of a wire partition; the partition drifts
             minimally each era to keep every g_i inside one block.  Pieces
             on distinct blocks commute, so mask-off/g/mask-on units
             interleave gate-by-gate, and at every cut every wire is covered
             by its old or its new mask -- no bare cut, by construction.
             Residue: inside a transition unit its own <=w (or joined <=3w)
             wires pass through bare briefly; window laundering then blends
             it.  (Killing that residue exactly = monolithic unit synthesis,
             the MITM-engine upgrade of ENCODED_CHAIN_DESIGN.md section 3.2.)

Cut-exposure audit: for every cut p of D and every wire, test whether the
delivered prefix state's wire column equals (or complements) any *informative*
source intermediate column (informative = not equal to an input column: x is
public).  Optionally, on sampled cuts, test GF(2)-affine membership of each
cut column in span{columns of s_j, 1} -- the xtrace/segment_deduce bar.
One-sided leakage instrument: hits are real leaks; absence is evidence, not
proof.  n <= 62 (int64 state pools).
"""
import itertools
import random
from collections import defaultdict

import numpy as np

import g57
from atoms import Resynth


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _commute(g, h):
    a1, x1, y1 = g
    a2, x2, y2 = h
    return a1 not in (a2, x2, y2) and a2 not in (a1, x1, y1)


def inv_word(word):
    """g57 gates are involutions (target never a control), so the inverse of a
    word is the reversed gate sequence."""
    return list(reversed(word))


def rand_gate(wires, rng):
    a, x, y = rng.sample(wires, 3)
    return (a, x, y)


def rand_word(wires, length, rng):
    return [rand_gate(wires, rng) for _ in range(length)]


def spread_word(wires, length, rng):
    """Random word whose targets cycle through shuffled copies of `wires`, so
    every wire is written ~length/|wires| times (coverage by construction --
    a mask that skips a wire leaves that wire bare for its whole era)."""
    tgts = []
    while len(tgts) < length:
        sh = list(wires)
        rng.shuffle(sh)
        tgts += sh
    word = []
    for a in tgts[:length]:
        x, y = rng.sample([u for u in wires if u != a], 2)
        word.append((a, x, y))
    return word


def random_circuit(n, k, rng):
    return rand_word(list(range(n)), k, rng)


def local_circuit(n, k, rng, span=3):
    """Wire-local source: each gate's 3 wires come from a random 2*span+1
    window (structured input, sliced-sandwich flavor)."""
    C = []
    for _ in range(k):
        c = rng.randrange(n)
        cand = [(c + d) % n for d in range(-span, span + 1)]
        a, x, y = rng.sample(cand, 3)
        C.append((a, x, y))
    return C


def words_equivalent(W1, W2, n, pool=None, trials=512, seed=99):
    """W1 == W2 as permutations: pool check (exact when False) + big-int trials."""
    if pool is None:
        pool = g57.make_state_pool(n, trials=256, seed=7)
    if not np.array_equal(g57.apply_word_batch(W1, pool),
                          g57.apply_word_batch(W2, pool)):
        return False
    rng = random.Random(seed)
    for _ in range(trials):
        s = rng.getrandbits(n)
        c1 = c2 = s
        for (a, x, y) in W1:
            if ((c1 >> x) & 1) | (1 - ((c1 >> y) & 1)):
                c1 ^= (1 << a)
        for (a, x, y) in W2:
            if ((c2 >> x) & 1) | (1 - ((c2 >> y) & 1)):
                c2 ^= (1 << a)
        if c1 != c2:
            return False
    return True


def launder_word(D, steps, rs, rng, max_win=4):
    """Window resynthesis + commuting reorder on an arbitrary word.  Preserves
    the global function (each rewrite preserves its window's function)."""
    D = list(D)
    rewrites = 0
    for _ in range(steps):
        L = len(D)
        if L < 3:
            break
        i = rng.randrange(L - 2)
        wlen = rng.randint(2, min(max_win, L - i))
        j = i + wlen
        supp = {w for gg in D[i:j] for w in gg}
        if len(supp) > rs.max_wires:
            if _commute(D[i], D[i + 1]):
                D[i], D[i + 1] = D[i + 1], D[i]
            continue
        alts = rs.alternatives(D[i:j], exclude_self=True, exact_len=wlen)
        if not alts:
            if _commute(D[i], D[i + 1]):
                D[i], D[i + 1] = D[i + 1], D[i]
            continue
        D[i:j] = rng.choice(alts)
        rewrites += 1
    return D, rewrites


# ---------------------------------------------------------------------------
# arm 1: naive base-CCMP -- blocks equivalent to a single gate, no legs
# ---------------------------------------------------------------------------
def g_block(g, n, rng, rs, tries=40, umax=3):
    """A random small circuit equivalent to gate `g` on its wires + one fresh
    wire: split as u ++ v with u random and v looked up in the BFS ball
    (equivalently: a random identity containing g, with g removed)."""
    a, x, y = g
    extra = rng.choice([w for w in range(n) if w not in (a, x, y)])
    wires = [a, x, y, extra]
    m = 4
    rs._build(m)
    back = {i: w for i, w in enumerate(wires)}
    fwd = {w: i for i, w in enumerate(wires)}
    Pg = g57.gate_table((fwd[a], fwd[x], fwd[y]), m)
    for _ in range(tries):
        u = [tuple(rng.sample(range(m), 3)) for _ in range(rng.randint(1, umax))]
        Pu = g57.word_table(u, m)
        Pv = Pg[np.argsort(Pu)]          # u then v == g  <=>  Pv[Pu] == Pg
        alts = rs._by_perm[m].get(Pv.tobytes())
        if alts:
            v = list(rng.choice(alts))
            w = u + [tuple(gg) for gg in v]
            if len(w) > 1:
                return g57.relabel(w, back)
    return [g]


def compile_naive(C, n, rng, rs=None, launder_steps=0):
    rs = rs or Resynth(max_wires=4, radius=4)
    D, fallback = [], 0
    for g in C:
        b = g_block(g, n, rng, rs)
        if b == [g]:
            fallback += 1
        D += b
    rewrites = 0
    if launder_steps:
        D, rewrites = launder_word(D, launder_steps, rs, rng)
    return {"D": D, "arm": "naive", "fallback_blocks": fallback,
            "launder_rewrites": rewrites}


# ---------------------------------------------------------------------------
# arm 2: full-width random legs + stagger + launder
# ---------------------------------------------------------------------------
def compile_randleg(C, n, rng, ell=40, stagger=30, launder_steps=0, rs=None,
                    cover=True):
    k = len(C)
    wires = list(range(n))
    mk = spread_word if cover else rand_word
    legs = {i: mk(wires, ell, rng) for i in range(2, k + 1)}
    legs[1], legs[k + 1] = [], []
    D = []
    for i, g in enumerate(C, start=1):
        D += inv_word(legs[i]) + [g] + legs[i + 1]
    swaps = 0
    L = len(D)
    for _ in range(stagger * L if L >= 2 else 0):
        p = rng.randrange(L - 1)
        if _commute(D[p], D[p + 1]):
            D[p], D[p + 1] = D[p + 1], D[p]
            swaps += 1
    rewrites = 0
    if launder_steps:
        rs = rs or Resynth(max_wires=4, radius=4)
        D, rewrites = launder_word(D, launder_steps, rs, rng)
    return {"D": D, "arm": "randleg", "ell": ell, "stagger_swaps": swaps,
            "launder_rewrites": rewrites}


# ---------------------------------------------------------------------------
# arm 3: brick -- group-local legs on a minimally drifting partition
# ---------------------------------------------------------------------------
def _evolve(P, g, rng):
    """Minimal re-blocking so wires(g) share one block: pull g's wires into the
    block of g's target, evicting random non-g wires to the donors."""
    P = [list(b) for b in P]
    idx = {wi: bi for bi, b in enumerate(P) for wi in b}
    gw = list(dict.fromkeys(g))          # (a, x, y), deduped, order kept
    home = idx[gw[0]]
    for wi in gw[1:]:
        bi = idx[wi]
        if bi == home:
            continue
        ev = rng.choice([u for u in P[home] if u not in gw])
        P[home].remove(ev)
        P[bi].append(ev)
        P[bi].remove(wi)
        P[home].append(wi)
        idx[ev], idx[wi] = bi, home
    return P


def _piece(block, mask_len, rng):
    """Mask piece: word on `block` only, every block wire written >= once."""
    tgts = list(block)
    while len(tgts) < mask_len:
        tgts.append(rng.choice(block))
    tgts = tgts[:mask_len]
    rng.shuffle(tgts)
    word = []
    for a in tgts:
        x, y = rng.sample([u for u in block if u != a], 2)
        word.append((a, x, y))
    return word


def resynth_unit(unit, rs, rng, tries=60, min_u=4, max_u=6):
    """Monolithic re-realization of a transition unit on <= max_wires support:
    replace the raw off|bare|on word by u ++ v where u is a fresh random word
    and v is a BFS-ball lookup completing the unit's permutation.  The interior
    cuts of the result are random-walk images, so the unit's systematic bare
    stretch (which window laundering provably cannot cover) disappears.
    This is the cheap stand-in for ENCODED_CHAIN_DESIGN.md section 3.2's
    monolithic transition synthesis; units wider than rs.max_wires (the
    partition-drift joins) are returned unchanged and measured honestly."""
    cw, order = g57.canonical_support(unit)
    m = len(order)
    if m > rs.max_wires or m < 3:
        return unit, False
    rs._build(m)
    Pt = g57.word_table(cw, m)
    back = {i: w for i, w in enumerate(order)}
    for _ in range(tries):
        u = [tuple(rng.sample(range(m), 3)) for _ in range(rng.randint(min_u, max_u))]
        Pu = g57.word_table(u, m)
        Pv = Pt[np.argsort(Pu)]
        alts = rs._by_perm[m].get(Pv.tobytes())
        if alts:
            v = list(rng.choice(alts))
            return g57.relabel(u + [tuple(gg) for gg in v], back), True
    return unit, False


def _components(blocksA, blocksB):
    """Wire -> component id for the join of two partitions."""
    parent = {}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for b in itertools.chain(blocksA, blocksB):
        for wi in b:
            parent.setdefault(wi, wi)
        for wi in b[1:]:
            ra, rb = find(b[0]), find(wi)
            if ra != rb:
                parent[rb] = ra
    return {wi: find(wi) for wi in parent}


def _seam(Pj, pieces_j, g, Pj1, pieces_j1, rng, gate_interleave=True,
          unit_rs=None):
    """off(M_j) . g . on(M_{j+1}) realized as commuting per-component units.
    Requires wires(g) inside one block of Pj (routing).

    gate_interleave=True spreads unit gates across the seam (better syntax,
    but a block's bare interval stretches across the whole seam);
    False emits each unit contiguously in random order (bare interval stays
    inside the unit's own <=w-wire support, where window laundering can
    genuinely re-encode it -- at the price of a contiguity fingerprint that
    laundering must then erase)."""
    comp = _components(Pj, Pj1)
    assert len({comp[wi] for wi in g}) == 1, \
        "gate not routed into one component -- unsound interleave"
    unit = defaultdict(list)
    for b in Pj:
        unit[comp[b[0]]] += inv_word(pieces_j.get(frozenset(b), []))
    unit[comp[g[0]]].append(g)
    for b in Pj1:
        unit[comp[b[0]]] += pieces_j1.get(frozenset(b), [])
    words = [u for u in unit.values() if u]
    resynthed = 0
    if unit_rs is not None:
        nw = []
        for u in words:
            u2, done = resynth_unit(u, unit_rs, rng)
            resynthed += done
            nw.append(u2)
        words = nw
    if gate_interleave:
        slots = [i for i, u in enumerate(words) for _ in u]
        rng.shuffle(slots)
        ptr = [0] * len(words)
        out = []
        for i in slots:
            out.append(words[i][ptr[i]])
            ptr[i] += 1
    else:
        rng.shuffle(words)
        out = [gg for u in words for gg in u]
    sizes = [len({w for gg in u for w in gg}) for u in words]
    return out, max(sizes) if sizes else 0, resynthed, len(words)


def compile_brick(C, n, rng, w=4, mask_len=6, launder_steps=0, rs=None,
                  gate_interleave=True, unit_resynth=False):
    assert n % w == 0, "n must be divisible by the block width"
    assert w >= 3, "blocks need >= 3 wires (g57 gates read 2, write 1)"
    assert mask_len >= w, "piece length must cover its block"
    k = len(C)
    wires = list(range(n))
    rng.shuffle(wires)
    base = [wires[i:i + w] for i in range(0, n, w)]
    parts = [_evolve(base, C[0], rng)]          # P_1 routes g_1
    for j in range(1, k):
        parts.append(_evolve(parts[-1], C[j], rng))   # P_{j+1} routes g_{j+1}
    pieces = [None] * (k + 2)
    pieces[1], pieces[k + 1] = {}, {}
    for i in range(2, k + 1):
        pieces[i] = {frozenset(b): _piece(b, mask_len, rng) for b in parts[i - 1]}
    rs = rs or Resynth(max_wires=4, radius=4)
    D, max_unit, res_done, res_total = [], 0, 0, 0
    for j in range(1, k + 1):
        Pj = parts[j - 1]
        Pj1 = parts[j] if j < k else parts[k - 1]     # P_{k+1}: no pieces
        seam, mx, rd, rt = _seam(Pj, pieces[j], C[j - 1], Pj1, pieces[j + 1],
                                 rng, gate_interleave=gate_interleave,
                                 unit_rs=rs if unit_resynth else None)
        D += seam
        max_unit = max(max_unit, mx)
        res_done += rd
        res_total += rt
    rewrites = 0
    if launder_steps:
        D, rewrites = launder_word(D, launder_steps, rs, rng)
    return {"D": D, "arm": "brick", "w": w, "mask_len": mask_len,
            "max_unit_support": max_unit, "launder_rewrites": rewrites,
            "units_resynthed": f"{res_done}/{res_total}" if unit_resynth else "off"}


# ---------------------------------------------------------------------------
# arm 4: brick2 -- persistent accreting masks (never-materialized legs)
# ---------------------------------------------------------------------------
def _block_word(block, length, rng):
    """Short random word on `block` only (no coverage requirement)."""
    word = []
    for _ in range(length):
        a = rng.choice(block)
        x, y = rng.sample([u for u in block if u != a], 2)
        word.append((a, x, y))
    return word


def _route_partition(W, w, C, j, rng, horizon=None):
    """Greedy forward-scan repartition of the freed wire set W into |W|/w
    blocks of width w (lookahead routing, register-allocation flavor): scan
    C[j+1:], co-place the wires of the nearest future gates having >= 2 wires
    in W (full triples and straddling pairs; nearest gates win), fill the rest
    randomly.  Measured (2026-08-28 red-team fleet): -23% exposure on random C,
    -36% on wire-local C, and a smaller delivered circuit."""
    W = list(W)
    nb = len(W) // w
    Wset = set(W)
    blocks = [set() for _ in range(nb)]
    free = [w] * nb
    assigned = {}
    end = len(C) if horizon is None else min(len(C), j + 1 + horizon)
    for g2 in C[j + 1:end]:
        win = [u for u in dict.fromkeys(g2) if u in Wset]
        if len(win) < 2:
            continue
        placed = {assigned[u] for u in win if u in assigned}
        if len(placed) > 1:
            continue                       # already split by an earlier gate
        need = [u for u in win if u not in assigned]
        if not need:
            continue
        if placed:
            bi = next(iter(placed))
            if free[bi] < len(need):
                continue
        else:
            cands = [b for b in range(nb) if free[b] >= len(need)]
            if not cands:
                continue
            bi = rng.choice(cands)
        for u in need:
            assigned[u] = bi
            blocks[bi].add(u)
            free[bi] -= 1
        if not any(free):
            break
    rest = [u for u in W if u not in assigned]
    rng.shuffle(rest)
    open_b = [b for b in range(nb) for _ in range(free[b])]
    rng.shuffle(open_b)
    for u, bi in zip(rest, open_b):
        blocks[bi].add(u)
    return [sorted(b) for b in blocks]


def compile_brick2(C, n, rng, w=4, mask_len=6, acc=3, launder_steps=0, rs=None,
                   lookahead=False, horizon=None):
    """Persistent accreting masks.  Each block of the partition carries an
    accumulated mask word phi_b.  At seam j only the blocks involved with g_j
    (its routed home plus any drift donors) unmask -- pay inv(phi), apply g,
    take a fresh covering mask; every other block just *accretes* `acc` fresh
    mask gates.  A mask is therefore never materialized as one word and never
    cancels adjacently: cancellation happens only across a block's whole
    g-visit gap, and the bare interval exists only inside g-units on their
    <=3w wires.  Head masks everything (input is public); tail unmasks
    everything (output is public).

    lookahead=True replaces the random-eviction drift (_evolve) with
    _route_partition: when a seam is forced to unmask >= 2 blocks anyway,
    their freed wire union is re-partitioned to co-block future gates'
    wire-triples.  Only forced seams route (voluntary extra unmasking was
    measured to backfire)."""
    assert n % w == 0 and mask_len >= w and w >= 3
    k = len(C)
    wires = list(range(n))
    rng.shuffle(wires)
    P = [wires[i:i + w] for i in range(0, n, w)]
    B = len(P)
    masks = [_piece(b, mask_len, rng) for b in P]
    units = [list(m) for m in masks]
    rng.shuffle(units)
    D = [gg for u in units for gg in u]
    max_unit, bare_wires, coblocked = 0, [], 0
    for j in range(k):
        g = C[j]
        idx = {wi: bi for bi, b in enumerate(P) for wi in b}
        involved = {idx[wi] for wi in set(g)}
        coblocked += len(involved) == 1
        if lookahead:
            if len(involved) > 1:
                inv = sorted(involved)
                W = [u for bi in inv for u in P[bi]]
                newb = _route_partition(W, w, C, j, rng, horizon)
                P2 = [list(b) for b in P]
                for bi, nb in zip(inv, newb):
                    P2[bi] = nb
            else:
                P2 = P
        else:
            P2 = _evolve(P, g, rng)
            involved |= {bi for bi in range(B) if set(P[bi]) != set(P2[bi])}
        involved = sorted(involved)
        u = []
        for bi in involved:
            u += inv_word(masks[bi])
        u.append(g)
        for bi in involved:
            masks[bi] = _piece(P2[bi], mask_len, rng)
            u += masks[bi]
        supp = len({ww for gg in u for ww in gg})
        max_unit = max(max_unit, supp)
        bare_wires.append(sum(len(P[bi]) for bi in involved))
        units = [u]
        for bi in range(B):
            if bi not in involved and acc:
                r = _block_word(P[bi], acc, rng)
                masks[bi] = masks[bi] + r
                units.append(r)
        rng.shuffle(units)
        D += [gg for uu in units for gg in uu]
        P = P2
    units = [inv_word(m) for m in masks]
    rng.shuffle(units)
    D += [gg for u in units for gg in u]
    rewrites = 0
    if launder_steps:
        rs = rs or Resynth(max_wires=4, radius=4)
        D, rewrites = launder_word(D, launder_steps, rs, rng)
    return {"D": D, "arm": "brick2" + ("+look" if lookahead else ""),
            "w": w, "mask_len": mask_len, "acc": acc,
            "max_unit_support": max_unit,
            "mean_bare_width": round(sum(bare_wires) / max(1, len(bare_wires)), 2),
            "coblocked_frac": round(coblocked / max(1, k), 3),
            "launder_rewrites": rewrites}


# ---------------------------------------------------------------------------
# arm 5: mono -- monolithic w-wire unit synthesis (the section-3.2 slot)
# ---------------------------------------------------------------------------
def mono_synth(unit, rs, rng, back_radius=4, len_lo=None, len_hi=None):
    """Fresh random g57 word equal to `unit` as a permutation on its support
    (<= rs.max_wires wires), whose interior states are a random walk -- NOT the
    raw factorization inv(M).g.M' interior, which passes through the bare source
    state.  The ENCODED_CHAIN_DESIGN.md 3.2 monolithic-synthesis slot at the
    widths the ball engine reaches.

    Asymmetric meet-in-the-middle: the reachable w-wire g57 group is far larger
    than any radius-r ball, so a random residual never lands in the ball.  We
    instead BFS *suffix* words s from the empty word to radius `back_radius`;
    each s needs prefix perm PA = argsort(Ps)[T], which we look up in the
    precomputed forward ball (T = A.s reachable iff dist(id,PA) <= ball radius).
    A meet at forward-depth p, back-depth q realizes T in p+q gates.  We collect
    every meet and pick one whose total length lands in [len_lo,len_hi] so units
    carry no length tell.  Returns (word, ok)."""
    cw, order = g57.canonical_support(unit)
    m = len(order)
    if m > rs.max_wires or m < 2:
        return unit, False
    rs._build(m)
    T = g57.word_table(cw, m)
    back = {i: order[i] for i in range(m)}
    fwd = rs._ball[m]                    # perm_bytes -> shortest prefix word
    gates = rs._gates(m)
    gts = {g: g57.gate_table(g, m) for g in gates}
    lo = len_lo if len_lo is not None else len(unit)
    hi = len_hi if len_hi is not None else len(unit) + 2 * rs.radius
    # BFS suffix words s (as (perm Ps, word)); for each, PA = argsort(Ps)[T].
    ID = np.arange(1 << m, dtype=np.int64)
    frontier = {ID.tobytes(): ()}
    seen = {ID.tobytes()}
    hits = []
    for _q in range(back_radius + 1):
        for pb, s in list(frontier.items()):
            Ps = np.frombuffer(pb, dtype=np.int64)
            PA = np.argsort(Ps)[T]
            pw = fwd.get(PA.tobytes())
            if pw is not None:
                W = g57.relabel(list(pw) + [tuple(gg) for gg in s], back)
                if lo <= len(W) <= hi:
                    return W, True
                hits.append(W)
        nf = {}
        for pb, s in frontier.items():
            Ps = np.frombuffer(pb, dtype=np.int64)
            for g, tb in gts.items():
                npb = tb[Ps].tobytes()
                if npb not in seen:
                    seen.add(npb)
                    nf[npb] = s + (g,)
        frontier = nf
        if not frontier:
            break
    if hits:
        mid = (lo + hi) / 2
        return min(hits, key=lambda W: abs(len(W) - mid)), True
    return unit, False


_MMD = None


def _mmd_engine():
    """Lazy handle to the MMD constructive synthesizer (synth_mmd/)."""
    global _MMD
    if _MMD is None:
        import os
        import sys
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_mmd")
        if d not in sys.path:
            sys.path.insert(0, d)
        import synth as _s
        _MMD = _s
    return _MMD


def mono_synth_mmd(unit, ancilla_pool, verify=False):
    """Constructive monolithic synthesis of a unit via MMD -> Barenco MCT ->
    g57 templates (synth_mmd/).  Unlike the ball MITM, this produces a fresh
    computation-trajectory word whose interior NEVER materializes the bare
    source state (measured 0% born-bare at every width vs the ball's ~51% at
    w=4), and lifts the comeback wall to the block width.  Cost: ~5k g57
    gates/unit at w=6 (MCT blow-up) -- the optimization frontier.

    Ancillas are DIRTY (restored to their input value for ALL 2^(w+n_anc)
    states), so `ancilla_pool` may be any n_anc wires disjoint from the unit's
    support; one source gate per era means the shared pool never conflicts.
    Returns (word, ok)."""
    cw, order = g57.canonical_support(unit)
    m = len(order)
    if m < 3:
        return unit, True                      # nothing to synthesize
    na = max(0, m - 3)
    if len(ancilla_pool) < na:
        return unit, False                     # not enough ancilla wires
    eng = _mmd_engine()
    gw, stt = eng.synth_unit(cw, m, n_anc=na, verify=verify)
    mp = {i: order[i] for i in range(m)}
    for i in range(na):
        mp[m + i] = ancilla_pool[i]
    return [(mp[a], mp[x], mp[y]) for (a, x, y) in gw], True


class _Deferred:
    """Placeholder for a unit whose synthesis is deferred to a batch pass
    (backend='db': one round-trip to the .242 DB for all units at once).
    Carries what the batch synthesizer needs: the raw realization to match, the
    accreted mask `M` that was inverted (so born-bare-freeness can be checked
    against the masked-input distribution), and the block wires."""
    __slots__ = ("raw", "M", "block")

    def __init__(self, raw, M, block):
        self.raw, self.M, self.block = list(raw), list(M), list(block)


def _coord_degrees(word, m):
    """Per-output-coordinate ANF (algebraic) degree of the permutation `word` on
    m wires -- via the Mobius transform on each output bit's truth table."""
    T = g57.word_table(word, m)
    N = 1 << m
    idx = np.arange(N)
    degs = []
    for bit in range(m):
        a = ((T >> bit) & 1).astype(np.uint8)
        for i in range(m):                      # in-place Mobius (ANF) transform
            step = 1 << i
            sel = (idx & step) > 0
            a[sel] ^= a[idx[sel] ^ step]
        degs.append(max((bin(i).count("1") for i in range(N) if a[i]), default=0))
    return degs


def _inv_min_degree(M, block):
    """Min over coordinates of the ANF degree of inv(M) restricted to `block`.
    This is the adversary's unmask degree: if 1, the bare source bit on that
    coordinate is affinely recoverable from the masked boundary (the deg-1
    break).  Enforce >= 2 (clears affine) or = w-1 for the full wall."""
    m = len(block)
    rel = {wv: i for i, wv in enumerate(block)}
    inv = [(rel[a], rel[x], rel[y]) for (a, x, y) in inv_word(M)]
    return min(_coord_degrees(inv, m)) if inv else 0


def _bornbare_free(word, M, block, pool_bits=6, seed=0):
    """True if `word`, fed the masked-input distribution M(pool), never passes
    through the bare source state (== pool) at any interior cut -- the born-bare
    audit restricted to the block, exact/complement over a full 2^|block| pool."""
    m = len(block)
    relabel = {wv: i for i, wv in enumerate(block)}
    Wl = [(relabel[a], relabel[x], relabel[y]) for (a, x, y) in word]
    Ml = [(relabel[a], relabel[x], relabel[y]) for (a, x, y) in M]
    pool = np.arange(1 << m, dtype=np.int64)          # every block state
    Z = g57.apply_word_batch(Ml, pool)                # masked inputs
    for gi in range(len(Wl) + 1):
        if np.array_equal(Z, pool):                   # bare state materialized
            return False
        if gi < len(Wl):
            Z = g57.apply_word_batch([Wl[gi]], Z)
    return True


def _standin_db_batch(pending, rng):
    """Local stand-in for the remote DB batch synth (verifies the two-pass
    plumbing without .242): re-synthesize each unit via the w-wire ball, prefer
    a born-bare-free realization, fall back to the raw word.  The real remote
    backend replaces this by querying the diameter-6 DB with MITM."""
    rs = Resynth(max_wires=5, radius=5)
    out = []
    for sent in pending:
        best = sent.raw
        for _ in range(6):
            cand, ok = mono_synth(sent.raw, rs, rng)
            if ok and _bornbare_free(cand, sent.M, sent.block):
                best = cand
                break
            if ok:
                best = cand
        out.append(best)
    return out


def compile_mono(C, n, rng, w=4, mask_len=6, acc=3, synth_radius=5, refresh=0,
                 launder_steps=0, rs=None, backend="ball", n_anc=None,
                 shuffle_init=True, synth_batch=None, min_mask_degree=0):
    """Persistent accreting masks (brick2 discipline) + monolithic synthesis of
    every CO-BLOCKED gate's unit, so its interior no longer materializes the
    bare source state and it stops looking wider than a mask unit.

    Routing keeps each gate's 3 wires inside one w-wire block whenever possible
    (register-allocation lookahead).  A co-blocked gate's unit lives on that one
    block: unmask it (inv of the accreted stack), apply g, take a fresh mask --
    then re-synthesize the whole w-wire word via `mono_synth`, whose interior is
    a random walk instead of the bare-state-crossing factorization.  Gates that
    cannot be co-blocked unmask the involved blocks and emit the raw braided
    unit -- the honest residual, counted `raw_units`.  Accretion words and
    head/tail (un)masking are emitted raw: their interiors are masked or public,
    never a source intermediate.

    `refresh` (item 2 -- the explicit layer schedule): every `refresh` eras a
    stale (un-visited) block is retired-and-refreshed *via mono_synth*
    (inv(stack).newpiece on w wires, no bare interior), so mask depth is
    scheduled independently of g-visits -- the "share several adjacent legs"
    knob, made safe by synthesis.  refresh=0 = brick2's retire-at-g-visit."""
    assert n % w == 0 and mask_len >= w and w >= 3
    assert backend != "ball" or w <= 5, "ball backend: w<=5 (ball explodes above)"
    k = len(C)

    # Degree enforcement (answers the w=3 affine-unmask leak): keep every mask's
    # inverse at coordinate-degree >= min_mask_degree.  Fresh pieces are already
    # >= 2; ACCRETION is what randomly drops a coordinate to affine (deg 1), so
    # we reject-and-resample the accretion word until the composed mask stays
    # above the floor.  min_mask_degree=0 disables (legacy behaviour).
    def mk_piece(block):
        for _ in range(40):
            p = _piece(block, mask_len, rng)
            if not min_mask_degree or _inv_min_degree(p, block) >= min_mask_degree:
                return p
        return p

    def mk_accretion(block, cur_mask):
        for _ in range(40):
            r = _block_word(block, acc, rng)
            if not min_mask_degree or _inv_min_degree(cur_mask + r, block) >= min_mask_degree:
                return r
        return r
    if backend == "ball":
        rs = rs or Resynth(max_wires=w, radius=synth_radius)
    # dirty-ancilla pool for the MMD backend: n_anc extra wires above the n
    # circuit wires, held at 0 (each unit restores them).  One source gate per
    # era => the shared pool never conflicts.
    NANC = (max(0, w - 3) if n_anc is None else n_anc) if backend == "mmd" else 0
    ancilla_pool = list(range(n, n + NANC))
    n_total = n + NANC
    pending = []                                       # backend='db' deferrals
    wires = list(range(n))
    if shuffle_init:
        rng.shuffle(wires)      # random block groups (obscures block membership)
    # else: contiguous aligned blocks [0..w-1],[w..2w-1],... -- when the source
    # respects those windows (a block-aligned sliced sandwich), nearly every
    # gate is co-blocked, so nearly every gate gets monolithic (born-bare-free)
    # synthesis.  Block membership is public then, but the mask VALUES stay
    # secret, so this is a routing choice, not a leak.
    P = [wires[i:i + w] for i in range(0, n, w)]
    B = len(P)
    masks = [mk_piece(b) for b in P]                     # current accreted stack/block
    last_seen = [0] * B
    D = [gg for u in [list(m) for m in masks] for gg in u]   # head: mask (public)
    coblocked = raw_units = mono_units = refreshed = mono_fail = 0

    def synth(word, M, block):
        # M = the accreted stack being inverted (raw = inv(M).g.piece or
        # inv(M).piece); block = the w wires -- both only needed by the db batch.
        nonlocal mono_units, mono_fail
        if backend == "db":
            sent = _Deferred(word, M, block)
            pending.append(sent)
            return [sent]                              # spliced in the batch pass
        if backend == "mmd":
            s, ok = mono_synth_mmd(word, ancilla_pool)
        else:
            s, ok = mono_synth(word, rs, rng, len_lo=mask_len, len_hi=mask_len + 2 * w)
        mono_units += ok
        mono_fail += not ok
        return s

    for j in range(k):
        g = C[j]
        idx = {wi: bi for bi, b in enumerate(P) for wi in b}
        involved = {idx[wi] for wi in set(g)}
        # One word per block this era, keyed by block id (the multi-block raw
        # unit under a distinct sentinel key).  Distinct keys touch disjoint
        # blocks => commute => the final shuffle is sound.  (Bug fixed 2026-08-29:
        # a refreshed block that also accretes must emit ONE ordered word, not
        # two shuffled non-commuting units on the same wires.)
        era = {}
        if len(involved) == 1:                       # co-blocked -> monolithic
            coblocked += 1
            b = next(iter(involved))
            M_old = masks[b]                           # stack inverted by the unit
            piece = mk_piece(P[b])
            raw = inv_word(M_old) + [g] + piece
            masks[b] = piece                          # new stack = the fresh mask
            era[b] = list(synth(raw, M_old, P[b]))
            last_seen[b] = j
        else:                                         # residual: raw multi-block
            inv = sorted(involved)
            W = [u for bi in inv for u in P[bi]]
            newb = _route_partition(W, w, C, j, rng)  # co-block for the future
            u = [gg for bi in inv for gg in inv_word(masks[bi])] + [g]
            P2 = [list(b) for b in P]
            for bi, nb in zip(inv, newb):
                P2[bi] = nb
            for bi in inv:
                masks[bi] = mk_piece(P2[bi])
                u += masks[bi]
                last_seen[bi] = j
            era[("multi", tuple(inv))] = u
            raw_units += 1
            P = P2
        if refresh:                                  # scheduled mono-refresh
            for bi in range(B):
                if bi not in involved and j - last_seen[bi] >= refresh:
                    M_old = masks[bi]
                    piece = mk_piece(P[bi])
                    ref = inv_word(M_old) + piece
                    masks[bi] = piece
                    era[bi] = list(synth(ref, M_old, P[bi]))
                    refreshed += 1
                    last_seen[bi] = j
        for bi in range(B):                          # accretion (masked, raw ok)
            if bi not in involved and acc:
                r = mk_accretion(P[bi], masks[bi])
                masks[bi] = masks[bi] + r
                era.setdefault(bi, []).extend(r)      # after refresh if both
        units = list(era.values())
        rng.shuffle(units)                            # keys => disjoint blocks
        D += [gg for u in units for gg in u]
    tail = [inv_word(m) for m in masks]              # tail: unmask (public C(x))
    rng.shuffle(tail)
    D += [gg for u in tail for gg in u]

    bb_free = None
    if backend == "db" and pending:
        fn = synth_batch or _standin_db_batch
        syn = fn(pending, rng)                        # words aligned with pending
        assert len(syn) == len(pending), "synth_batch returned wrong count"
        bb_free = 0
        rep = {}
        for s, wsyn in zip(pending, syn):
            m = len(s.block)
            rel = {wv: i for i, wv in enumerate(s.block)}
            traw = g57.word_table([(rel[a], rel[x], rel[y]) for a, x, y in s.raw], m)
            tsyn = g57.word_table([(rel[a], rel[x], rel[y]) for a, x, y in wsyn], m)
            assert np.array_equal(traw, tsyn), \
                "db batch synth returned a non-equivalent unit"
            bb_free += _bornbare_free(wsyn, s.M, s.block)
            rep[id(s)] = wsyn
        newD = []                                     # splice synthesized words in
        for item in D:
            newD.extend(rep[id(item)] if isinstance(item, _Deferred) else [item])
        D = newD
        mono_units, mono_fail = len(pending), 0

    rewrites = 0
    if launder_steps:
        rs = rs or Resynth(max_wires=min(w, 5), radius=synth_radius)
        D, rewrites = launder_word(D, launder_steps, rs, rng)
    out = {"D": D, "arm": "mono", "backend": backend, "w": w, "mask_len": mask_len,
           "acc": acc, "n_total": n_total, "n_anc": NANC,
           "coblocked_frac": round(coblocked / max(1, k), 3),
           "mono_units": mono_units, "mono_fail": mono_fail,
           "raw_units": raw_units, "refreshed": refreshed,
           "launder_rewrites": rewrites}
    if bb_free is not None:
        out["bornbare_free"] = f"{bb_free}/{len(pending)}"
    return out


# ---------------------------------------------------------------------------
# the cut-exposure audit
# ---------------------------------------------------------------------------
def _col_bits(Z, n):
    return ((Z[None, :] >> np.arange(n, dtype=np.int64)[:, None]) & 1).astype(np.uint8)


def _col_keys(Z, n):
    B = _col_bits(Z, n)
    pk = np.packbits(B, axis=1)
    pkc = np.packbits(1 - B, axis=1)
    return ([pk[i].tobytes() for i in range(n)],
            [pkc[i].tobytes() for i in range(n)])


def cut_exposure(D, C, n, pool_size=256, seed=0, affine_cuts=120, affine_seed=1):
    """Measure materialization of source intermediates at cuts of D.

    Returns per-(cut, wire) exact/complement exposure vs every informative
    source-intermediate column, full-state exposure, and (sampled) GF(2)-affine
    exposure of cut columns in span{columns of s_j, 1}.
    """
    assert n <= 62
    pool = g57.make_state_pool(n, trials=pool_size, seed=seed)
    k, L = len(C), len(D)
    S = [pool.copy()]
    for g in C:
        S.append(g57.apply_word_batch([g], S[-1]))
    # public columns: the input s_0 and the output s_k (the adversary holds
    # both), plus complements; only interior intermediates count as leaks.
    trivial = set()
    for j in (0, k):
        kj, ckj = _col_keys(S[j], n)
        trivial |= set(kj) | set(ckj)
    targets = {}
    for j in range(1, k):
        kj, _ = _col_keys(S[j], n)
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
    covered = set()          # informative (j, wi) source slots seen at any cut
    full_hits = []
    Z = pool.copy()
    for p in range(L + 1):
        if p > 0:
            Z = g57.apply_word_batch([D[p - 1]], Z)
        kp, ckp = _col_keys(Z, n)
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
        B = _col_bits(Z, n)
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
        # density can be diluted by length: always read these two with it
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


def to_mpmct1(D, n):
    """Serialize a g57 word to the mpmct1 plain-text format fmix/fcompress read
    (--input-format mpmct1).  A g57 gate (a,x,y) = a ^= (x OR NOT y) is, per
    src/circuit/xgate.rs from_g57, the XGate target=a comp=1 ctrls=(x,0)(y,1)
    -- i.e. fires = 1 XOR (NOT x AND y) = x OR NOT y.  Verified against the Rust
    reader's field order (target comp k [wire pol]*)."""
    lines = [f"mpmct1 {n} {len(D)}"]
    lines += [f"{a} 1 2 {x} 0 {y} 1" for (a, x, y) in D]
    return "\n".join(lines) + "\n"


def print_exposure(rep, label):
    a = rep["affine"]
    at = max(1, a["sampled_columns"])
    print(f"--- cut exposure: {label}  (len {rep['len']}, k={rep['k']}, "
          f"n={rep['n']}, informative cols {rep['informative_columns']}) ---")
    cov, slots = rep["source_coverage"]
    print(f"  exact/comp exposure    : {rep['exposure_fraction']*100:6.2f}% of (cut,wire)   "
          f"cuts touched {rep['cuts_with_any_exposure']}/{rep['cuts_total']}   "
          f"mean/max wires per cut {rep['mean_exposed_wires_per_cut']:.2f}/{rep['max_exposed_wires']}")
    print(f"  absolute / coverage    : {rep['exposed_pairs']} exposed (cut,wire) pairs   "
          f"source values covered {cov}/{slots}"
          f" = {cov/max(1,slots)*100:.0f}%  (density alone can flatter: long circuits dilute)")
    print(f"  full-state bare cuts   : {rep['full_state_hits']}"
          + (f"   at {rep['full_state_positions']}" if rep["full_state_hits"] else ""))
    print(f"  affine (sampled)       : public-span {a['in_public_span']/at*100:.1f}%   "
          f"intermediate-span-only {a['in_intermediate_span_only']/at*100:.1f}%")
