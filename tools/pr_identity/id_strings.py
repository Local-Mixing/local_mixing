#!/usr/bin/env python3
"""(d,k)-strings of locally geodesic identities.

A (d,k)-string is a sequence C_1..C_k of locally geodesic identity circuits
(drawn from identities_m1m11_le12_dedup.csv, each independently wire-relabeled
and optionally rotated/reversed — both preserve identity-ness and cyclic local
geodesicity) such that
  (a) the d-gate suffix of C_i equals, gate for gate, the REVERSE of the
      d-gate prefix of C_{i+1}  (g57 gates are involutions, so those 2d gates
      cancel exactly when the C_i are concatenated), and
  (b) every C_i has at least 3d+1 gates.

Concatenating and cancelling seams yields the identity
    W = C_1[:-d] ++ C_2[d:-d] ++ ... ++ C_k[d:]
of size sum(L_i) - 2d(k-1)  (= k(d+1) + 2d when all L_i = 3d+1).

d-simplicity ("no chords of fewer than d gates"): for every contiguous window
w of W, geodesic(fn(w)) >= min(len(w), |W|-len(w), d).  (The |W|-len term is
forced: W is itself an identity, so a window of length |W|-c always has the
c-gate complement as a bypass.)

U-claim: with g_i = the (d+1)-th gate of C_i and P_i = C_i's d-gate prefix,
removing g_1..g_k from W leaves U with
    U  ==  P_1 g_1 P_1* . P_2 g_2 P_2* . ... . P_k g_k P_k*   (X* = reverse)
i.e. a product of prefix-CONJUGATED gates.  U == g_1..g_k exactly iff each
perm(P_i) commutes with g_i — the --commuting-prefix mode enforces that.
(Exact equality via a common prefix permutation is impossible: perm(P_i)=p and
perm(suffix)=p^-1 forces the middle L-2d gates to be a proper sub-identity,
contradicting local geodesicity.)
"""

import argparse
import random
import sys
from collections import Counter, defaultdict

from g57 import canonical_support, relabel, wires_of

LIB = "identities_m1m11_le12_dedup.csv"


# ---------------------------------------------------------------- library ---

def load_library(path):
    lib = []
    with open(path) as f:
        header = f.readline()
        assert header.startswith("length,")
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, _, circ = line.split(",", 2)
            word = tuple(tuple(int(t) for t in tok.split(",")) for tok in circ.split(";"))
            lib.append(word)
    return lib


def rev_word(word):
    """Inverse of a word (g57 gates are involutions)."""
    return tuple(reversed(word))


def rotations(word):
    L = len(word)
    for r in range(L):
        yield r, word[r:] + word[:r]


def canon_word(word):
    """Relabel wires by first occurrence -> canonical tuple (relabel-invariant)."""
    seen = {}
    out = []
    for g in word:
        out.append(tuple(seen.setdefault(w, len(seen)) for w in g))
    return tuple(out)


def oriented(word, o, r):
    w = rev_word(word) if o else word
    return w[r:] + w[:r]


# ---------------------------------------------------------- exact小 tables ---

def word_perm_table(word):
    """Exact permutation of the word on its own support (<= ~16 wires).

    Returns (support list, table) with table[s] over 2^len(support) states,
    wire i of the table = support[i].
    """
    sup = wires_of(word)
    pos = {w: i for i, w in enumerate(sup)}
    lw = [(pos[a], pos[x], pos[y]) for (a, x, y) in word]
    n = len(sup)
    tbl = list(range(1 << n))
    for s in range(1 << n):
        cur = s
        for (a, x, y) in lw:
            if ((cur >> x) & 1) | (1 - ((cur >> y) & 1)):
                cur ^= 1 << a
        tbl[s] = cur
    return sup, tbl


def _moved_deps_idx(n, tbl):
    """Exact moved-wire and dependency-wire indices of a table permutation."""
    moved = 0
    for s, t in enumerate(tbl):
        moved |= s ^ t
    deps = 0
    for i in range(n):
        for s in range(1 << n):
            if tbl[s] ^ tbl[s ^ (1 << i)] != (1 << i):
                deps |= 1 << i
                break
    return ([i for i in range(n) if (moved >> i) & 1],
            [i for i in range(n) if (deps >> i) & 1])


def _short_equiv_tbl(n, tbl, cmax):
    """Length of a shortest word of <= cmax g57 gates realizing table `tbl` on
    n wires, or None.  Candidate gates target a moved wire with controls among
    moved+dep wires — complete for cmax <= 2 (a control outside deps leaves a
    net dependency, so it can never help); MITM for cmax 3..4."""
    ident = list(range(1 << n))
    if tbl == ident:
        return 0
    if cmax < 1:
        return None
    movedi, depsi = _moved_deps_idx(n, tbl)
    if len(movedi) > cmax:         # write-counting lemma: minlen >= #moved
        return None
    dep = sorted(set(movedi) | set(depsi))
    gates = []
    for a in movedi:
        for x in dep:
            if x == a:
                continue
            for y in dep:
                if y == a or y == x:
                    continue
                gates.append((a, x, y))
    gtbl = {}
    for g in gates:
        a, x, y = g
        gtbl[g] = [s ^ (1 << a) if (((s >> x) & 1) | (1 - ((s >> y) & 1))) else s
                   for s in range(1 << n)]

    def compose(t1, t2):           # apply t1 then t2
        return [t2[v] for v in t1]

    for g in gates:
        if gtbl[g] == tbl:
            return 1
    if cmax >= 2:
        for g1 in gates:
            t1 = gtbl[g1]
            for g2 in gates:
                if g2 != g1 and compose(t1, gtbl[g2]) == tbl:
                    return 2
    if cmax >= 3:
        front = {tuple(gtbl[g]): 1 for g in gates}
        for g1 in gates:
            for g2 in gates:
                if g1 != g2:
                    front.setdefault(tuple(compose(gtbl[g1], gtbl[g2])), 2)
        for c in (3, 4):
            if c > cmax:
                break
            bl = c // 2            # back length: c=3 -> front2+back1, c=4 -> 2+2
            if bl == 1:
                backs = [gtbl[g] for g in gates]
            else:
                backs = []
                for g1 in gates:
                    for g2 in gates:
                        if g1 != g2:
                            backs.append(compose(gtbl[g1], gtbl[g2]))
            for bt in backs:
                binv = [0] * (1 << n)
                for s, t in enumerate(bt):
                    binv[t] = s
                fl = front.get(tuple(compose(tbl, binv)))
                if fl is not None and fl + bl == c:
                    return c
    return None


def has_short_equiv(word, cmax):
    sup, tbl = word_perm_table(word)
    return _short_equiv_tbl(len(sup), tbl, cmax)


def word_table_on(word, wires, background=0):
    """Table of `word` restricted to `wires`; the word's other support wires
    are pinned to the bits of `background` (indexed in support order)."""
    sup = wires_of(word)
    pos = {w: i for i, w in enumerate(sup)}
    lw = [(pos[a], pos[x], pos[y]) for (a, x, y) in word]
    idx = [pos[w] for w in wires]
    n = len(wires)
    tbl = []
    for s in range(1 << n):
        full = background
        for j, i in enumerate(idx):
            if (s >> j) & 1:
                full |= 1 << i
            else:
                full &= ~(1 << i)
        cur = full
        for (a, x, y) in lw:
            if ((cur >> x) & 1) | (1 - ((cur >> y) & 1)):
                cur ^= 1 << a
        r = 0
        for j, i in enumerate(idx):
            if (cur >> i) & 1:
                r |= 1 << j
        tbl.append(r)
    return tbl


def perm_key(word):
    """Exact global-permutation key of a small word: (moved+dep wires, table
    restricted to them).  Two words are the same permutation iff keys match."""
    sup, tbl = word_perm_table(word)
    movedi, depsi = _moved_deps_idx(len(sup), tbl)
    R = sorted({sup[i] for i in movedi} | {sup[i] for i in depsi})
    if not R:
        return ((), ())
    return (tuple(R), tuple(word_table_on(word, R)))


def window_chord_ok(win, need, rng=None):
    """True iff geodesic(fn(win)) >= need.  Ladder: sampled moved-wire count
    (write-counting, sound) -> exact small-support check -> certified-deps
    counting -> exact check on the reduced moved+deps support.
    Returns True / False / None (None = could not certify)."""
    if need <= 0:
        return True
    if rng is None:
        rng = _LB_RNG
    win = tuple(canonical_support(list(win))[0])   # compact labels: cheap ints
    sup = wires_of(win)
    top = max(sup) + 1
    moved = 0
    for _ in range(32):
        s = rng.getrandbits(top)
        moved |= run_word(win, s) ^ s
        if bin(moved).count("1") >= need:
            return True
    if len(sup) <= 14:
        return has_short_equiv(win, need - 1) is None
    # wide support, few moved wires: certify dependencies by bit flips
    M = {i for i in range(top) if (moved >> i) & 1}
    D = set()
    for w in sup:
        for _ in range(24):
            s = rng.getrandbits(top)
            delta = run_word(win, s) ^ run_word(win, s ^ (1 << w))
            if delta & ~(1 << w):
                D.add(w)
                break
    if len(M | D) > 3 * (need - 1):
        return True             # a <need-gate word touches <= 3(need-1) wires
    R = sorted(M | D)
    if not R:
        return False            # identity window: chord of 0 gates
    if len(R) > 12:
        return None
    t1 = word_table_on(win, R, 0)
    t2 = word_table_on(win, R, rng.getrandbits(len(sup)))
    if t1 != t2:
        return None             # missed a dependency; cannot certify
    return _short_equiv_tbl(len(R), t1, need - 1) is None


# ------------------------------------------------------------- d-simplicity ---

_LB_RNG = random.Random(424242)


def geodesic_lb_ok(window, need):
    """True iff geodesic(fn(window)) >= need, exact on the window support."""
    if need <= 0:
        return True
    sup = wires_of(window)
    # fast sound pre-screen: wires observed moved on random states lower-bound
    # the geodesic (write-counting lemma)
    top = max(sup) + 1
    moved = 0
    for _ in range(16):
        s = _LB_RNG.getrandbits(top)
        cur = s
        for (a, x, y) in window:
            if ((cur >> x) & 1) | (1 - ((cur >> y) & 1)):
                cur ^= 1 << a
        moved |= cur ^ s
        if bin(moved).count("1") >= need:
            return True
    if len(sup) > 14:
        # cheap sound bound: count wires observed moved on random states
        rng = random.Random(12345)
        moved = set()
        top = max(sup) + 1
        for _ in range(48):
            s = rng.getrandbits(top)
            cur = s
            for (a, x, y) in window:
                if ((cur >> x) & 1) | (1 - ((cur >> y) & 1)):
                    cur ^= 1 << a
            dif = cur ^ s
            i = 0
            while dif:
                if dif & 1:
                    moved.add(i)
                dif >>= 1
                i += 1
            if len(moved) >= need:
                return True
        return None                 # undecided (wide window, few moved wires)
    return has_short_equiv(window, need - 1) is None


def cyclic_d_simple(word, d, cache=None, key=None):
    """All cyclic windows w of the identity: geodesic >= min(l, L-l, d)."""
    if cache is not None and key in cache:
        return cache[key]
    L = len(word)
    ok = True
    for l in range(2, L - 1 + 1):
        need = min(l, L - l, d)
        if need < 1:
            continue
        for r in range(L):
            win = (word[r:] + word[:r])[:l]
            res = window_chord_ok(win, need)
            if res is not True:
                ok = False
                break
        if not ok:
            break
    if cache is not None:
        cache[key] = ok
    return ok


# ------------------------------------------------------------------ graph ---

def build_index(lib, d, cap=300, max_rot=None, seed=0, minlen=None):
    """Edges a->b over canon d-patterns; a=canon(prefix_d), b=canon(rev suffix_d)."""
    minlen = (3 * d + 1) if minlen is None else minlen
    rng = random.Random(seed)
    adj_edges = defaultdict(list)     # a -> [(idx,o,r,b)] capped reservoir
    out_count = Counter()
    node_adj = defaultdict(set)       # a -> {b}
    n_edges = 0
    n_ident = 0
    for idx, word in enumerate(lib):
        L = len(word)
        if L < minlen:
            continue
        n_ident += 1
        rots = range(L) if max_rot is None else rng.sample(range(L), min(L, max_rot))
        for o in (0, 1):
            for r in rots:
                w = oriented(word, o, r)
                a = canon_word(w[:d])
                b = canon_word(rev_word(w[-d:]))
                n_edges += 1
                out_count[a] += 1
                node_adj[a].add(b)
                lst = adj_edges[a]
                if len(lst) < cap:
                    lst.append((idx, o, r, b))
                else:
                    j = rng.randrange(out_count[a])
                    if j < cap:
                        lst[j] = (idx, o, r, b)
    return dict(adj_edges), out_count, dict(node_adj), n_edges, n_ident


def scc_sizes(node_adj):
    """Iterative Tarjan SCC over the pattern graph."""
    nodes = set(node_adj)
    for vs in node_adj.values():
        nodes |= vs
    index = {}
    low = {}
    onstk = set()
    stk = []
    sccs = []
    counter = [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(node_adj.get(root, ()))))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stk.append(root)
        onstk.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stk.append(w)
                    onstk.add(w)
                    work.append((w, iter(sorted(node_adj.get(w, ())))))
                    advanced = True
                    break
                elif w in onstk:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stk.pop()
                    onstk.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
    return sorted((len(c) for c in sccs), reverse=True), sccs


# ------------------------------------------------------------------- walk ---

def match_relabel(pref, port):
    """Injective wire map sending word `pref` onto word `port`, or None."""
    m = {}
    used = set()
    for gp, gq in zip(pref, port):
        for wp, wq in zip(gp, gq):
            if wp in m:
                if m[wp] != wq:
                    return None
            else:
                if wq in used:
                    return None
                m[wp] = wq
                used.add(wq)
    return m


def perms_commute(word1, word2):
    """Exact commutation of two small words as permutations."""
    sup = sorted(set(wires_of(word1)) | set(wires_of(word2)))
    if len(sup) > 16:
        return False
    pos = {w: i for i, w in enumerate(sup)}
    def run(word, s):
        for (a, x, y) in word:
            a, x, y = pos[a], pos[x], pos[y]
            if ((s >> x) & 1) | (1 - ((s >> y) & 1)):
                s ^= 1 << a
        return s
    return all(run(word2, run(word1, s)) == run(word1, run(word2, s))
               for s in range(1 << len(sup)))


def assemble_run(parts, d):
    """Reduced word of the current walk (last part keeps its suffix)."""
    W = []
    for i, p in enumerate(parts):
        block = list(p) if i == len(parts) - 1 else list(p[:len(p) - d])
        if i > 0:
            block = block[d:]
        W.extend(block)
    return W


def build_string(lib, adj_edges, d, k, seed=None, commuting=False,
                 dsimple_cache=None, exact_len=False, max_tries=400,
                 n_wires=None, lcap=24, screen_d=None):
    """Random seam-graph walk with two d-simplicity enforcement mechanisms:

    * SEAM LEDGER — the running permutation of W at the seam cut after C_i is
      exactly perm(port_i) (everything earlier telescopes away, each C_j being
      an identity).  So a chord between two seam cuts is port_i^-1 . port_j:
      the k-1 seam permutations must be pairwise distinct (equal => an interior
      identity window, a 0-gate chord) and pairwise >= d apart in the Cayley
      metric.  Enforced incrementally; disjoint-support pairs are auto-far.
    * LOCAL WINDOW SCREEN — every window of length <= lcap crossing a new seam
      must have geodesic >= min(len, d); candidates violating it are rejected.
    """
    rng = random.Random(seed)
    if dsimple_cache is None:
        dsimple_cache = {}
    if screen_d is None:
        screen_d = d     # simplicity threshold may exceed the seam width
    minlen = 3 * d + 1   # (a,b,c)-strings; composition mode only needs 2d+2
    parts = []          # relabeled identity words
    metas = []          # (idx, o, r)
    Wrun = []           # reduced word so far
    port = None         # literal d-word the next prefix must equal
    ledger = {}         # perm_key -> seam index
    seam_words = []     # port word per materialized seam
    wire2seams = defaultdict(list)
    fresh = [0]

    def alloc():
        fresh[0] += 1
        return fresh[0] - 1

    def seam_ok_vs_ledger(pw):
        key = perm_key(pw)
        if key in ledger:
            return False
        cand = set()
        for w0 in key[0]:
            cand.update(wire2seams.get(w0, ()))
        for j in cand:
            ww = tuple(list(rev_word(seam_words[j])) + list(pw))
            if window_chord_ok(ww, screen_d, rng) is not True:
                return False
        return True

    def register_seam(pw):
        jidx = len(seam_words)
        ledger[perm_key(pw)] = jidx
        seam_words.append(tuple(pw))
        for w0 in perm_key(pw)[0]:
            wire2seams[w0].append(jidx)

    def undo_last_part():
        parts.pop()
        metas.pop()
        if parts and seam_words:
            jidx = len(seam_words) - 1
            pw = seam_words.pop()
            key = perm_key(pw)
            del ledger[key]
            for w0 in key[0]:
                wire2seams[w0].remove(jidx)

    def local_screen(W2, seam_pos):
        m2 = len(W2)
        lo = max(0, seam_pos - lcap + 1)
        hi = min(m2, seam_pos + lcap)
        region = W2[lo:hi]
        cw, _ = canonical_support(region)      # compact labels: cheap probes
        topw = (max(wires_of(cw)) + 1) if cw else 1
        for s0 in range(0, seam_pos - lo):
            vals = [rng.getrandbits(topw) for _ in range(10)]
            base = vals[:]
            movedmask = 0
            e = s0
            while e < min(len(cw), s0 + lcap):
                g = cw[e]
                vals = [run_word((g,), s) for s in vals]
                e += 1
                for b0, v in zip(base, vals):
                    movedmask |= b0 ^ v
                l = e - s0
                if e + lo <= seam_pos or l < 2:
                    continue
                need = min(l, screen_d)
                if bin(movedmask).count("1") >= need:
                    continue
                if window_chord_ok(tuple(cw[s0:e]), need, rng) is not True:
                    return False
        return True

    nodes = list(adj_edges)
    while len(parts) < k:
        # the pending port becomes a real seam once the next part attaches:
        # it must be admissible against the ledger, else back out
        if port is not None and not seam_ok_vs_ledger(port):
            undo_last_part()
            Wrun = assemble_run(parts, d)
            port = rev_word(parts[-1][-d:]) if parts else None
            max_tries -= 1
            if max_tries <= 0:
                raise RuntimeError("walk stalled (ledger dead ends)")
            continue
        a = canon_word(port) if port is not None else rng.choice(nodes)
        cands = adj_edges.get(a, [])
        cands = rng.sample(cands, len(cands))
        placed = False
        for (idx, o, r, b) in cands:
            word = lib[idx]
            if exact_len and len(word) != minlen:
                continue
            if not cyclic_d_simple(word, screen_d, dsimple_cache,
                                   (idx, screen_d)):
                continue
            w = oriented(word, o, r)
            if commuting and not perms_commute(w[:d], (w[d],)):
                continue
            if port is None:
                m = {}
                used = set()
            else:
                m = match_relabel(w[:d], port)
                if m is None:
                    continue
                m = dict(m)
                used = set(m.values())
            bad = False
            for wi in wires_of(w):
                if wi not in m:
                    if n_wires is None:
                        while True:
                            c = alloc()
                            if c not in used:
                                break
                    else:
                        free = [c for c in range(n_wires) if c not in used]
                        if not free:
                            bad = True
                            break
                        c = rng.choice(free)
                    m[wi] = c
                    used.add(c)
            if bad:
                continue
            rw = tuple(relabel(w, m))
            # cascade guard: cancellation must stop at exactly d gates
            if parts:
                prev = parts[-1]
                if len(prev) > d and len(rw) > d and prev[-d - 1] == rw[d]:
                    continue
                if rev_word(prev[-d - 1:]) == rw[:d + 1]:
                    continue
            if parts:
                W2 = Wrun[:-d] + list(rw[d:])
                if not local_screen(W2, len(Wrun) - d):
                    continue
            else:
                W2 = list(rw)
            if parts:
                register_seam(port)
            parts.append(rw)
            metas.append((idx, o, r))
            Wrun = W2
            if n_wires is None:
                fresh[0] = max(fresh[0], max(wires_of(rw)) + 1)
            port = rev_word(rw[-d:])
            placed = True
            break
        if not placed:
            if len(parts) == 0:
                raise RuntimeError(f"no usable start edge for d={d}")
            undo_last_part()
            Wrun = assemble_run(parts, d)
            port = rev_word(parts[-1][-d:]) if parts else None
            max_tries -= 1
            if max_tries <= 0:
                raise RuntimeError("walk stalled (too many dead ends)")
    return parts, metas


def assemble(parts, d):
    """Cancel the 2d seam gates; return (W, g_positions, prefixes, gs)."""
    W = []
    gpos = []
    prefixes = []
    gs = []
    for i, p in enumerate(parts):
        block = list(p[:len(p) - d] if i < len(parts) - 1 else p)
        if i > 0:
            block = block[d:]
        if i == 0:
            gpos.append(len(W) + d)
        else:
            gpos.append(len(W))
        prefixes.append(p[:d])
        gs.append(p[d])
        W.extend(block)
    return W, gpos, prefixes, gs


# ------------------------------------------------------------ verification ---

def run_word(word, s):
    for (a, x, y) in word:
        if ((s >> x) & 1) | (1 - ((s >> y) & 1)):
            s ^= 1 << a
    return s


def verify_string(W, gpos, prefixes, gs, d, parts, trials=1024, lcap=24,
                  seed=987, log=print):
    n = max(wires_of(W)) + 1
    rng = random.Random(seed)
    report = {}

    # 1. identity
    ok = all(run_word(W, s := rng.getrandbits(n)) == s for _ in range(trials))
    report["identity"] = ok
    log(f"  identity ({trials} random states, n={n}): {'PASS' if ok else 'FAIL'}")

    # 2. reduced word: no adjacent equal gates (incl. across seams)
    adj = any(W[i] == W[i + 1] for i in range(len(W) - 1))
    report["no_adjacent_cancel"] = not adj
    log(f"  no adjacent cancelling pair: {'PASS' if not adj else 'FAIL'}")

    # 3. chord-0 scan: no interior window is an identity (probe-cut collisions)
    probes = [rng.getrandbits(n) for _ in range(40)]
    sigs = {}
    cur = list(probes)
    dup = 0
    for i in range(len(W) + 1):
        key = hash(tuple(cur))
        if key in sigs and 0 < i < len(W):
            j = sigs[key]
            # confirm the candidate identity-window on fresh states
            win = W[j:i]
            if all(run_word(win, s) == s
                   for s in (rng.getrandbits(n) for _ in range(24))):
                dup += 1
        sigs.setdefault(key, i)
        if i < len(W):
            cur = [run_word((W[i],), s) for s in cur]
    report["no_identity_window"] = dup == 0
    log(f"  interior identity-windows (chord 0): {dup} {'PASS' if dup == 0 else 'FAIL'}")

    # 4. window screen for chords < d, all windows up to lcap gates
    m = len(W)
    suspicious = []
    checked = 0
    for i in range(m):
        vals = list(probes[:10])
        base = vals[:]
        movedmask = 0
        for l in range(1, min(lcap, m - i) + 1):
            g = W[i + l - 1]
            vals = [run_word((g,), s) for s in vals]
            for b, v in zip(base, vals):
                movedmask |= b ^ v
            if l < 2:
                continue
            need = min(l, m - l, d)
            checked += 1
            if bin(movedmask).count("1") >= need:
                continue
            win = tuple(W[i:i + l])
            res = window_chord_ok(win, need, rng)
            if res is not True:
                suspicious.append((i, l, res))
        # cheap reset for next start
    report["window_screen"] = (checked, suspicious)
    ok4 = not suspicious
    log(f"  chord screen (windows len<= {lcap}, {checked} checked): "
        f"{'PASS' if ok4 else f'FAIL {suspicious[:5]}'}")

    # 5. sampled long windows (chords < d must not exist at any length)
    long_bad = []
    n_long = 1200
    for _ in range(n_long):
        l = rng.randrange(lcap + 1, max(lcap + 2, m // 2))
        i = rng.randrange(0, m - l)
        win = W[i:i + l]
        moved = set()
        for s in probes[:8]:
            t = run_word(win, s)
            dif = s ^ t
            j = 0
            while dif:
                if dif & 1:
                    moved.add(j)
                dif >>= 1
                j += 1
        if len(moved) < d:
            res = window_chord_ok(tuple(win), min(l, m - l, d), rng)
            if res is not True:
                long_bad.append((i, l, res))
    report["long_window_screen"] = long_bad
    log(f"  long-window sample screen ({n_long} windows): "
        f"{'PASS' if not long_bad else f'FAIL {long_bad[:5]}'}")

    # 5b. seam-cut audit: running perm at the seam after C_i is perm(port_i);
    # all seam pairs must be distinct (else 0-chord) and >= d apart
    seam_words = [rev_word(p[-d:]) for p in parts[:-1]]
    keys = [perm_key(w) for w in seam_words]
    eq_dup = len(keys) - len(set(keys))
    wmap = defaultdict(list)
    for i2, k2 in enumerate(keys):
        for w2 in k2[0]:
            wmap[w2].append(i2)
    pairs = set()
    for lst in wmap.values():
        for ii in range(len(lst)):
            for jj in range(ii + 1, len(lst)):
                pairs.add((lst[ii], lst[jj]))
    # disjoint-support pairs whose combined moved set could still be < d
    movedsets = []
    for w2 in seam_words:
        sup2, tbl2 = word_perm_table(w2)
        mi, _ = _moved_deps_idx(len(sup2), tbl2)
        movedsets.append({sup2[i3] for i3 in mi})
    small = [i2 for i2, ms in enumerate(movedsets) if len(ms) <= d - 2]
    for ii in range(len(small)):
        for jj in range(ii + 1, len(small)):
            i2, j2 = small[ii], small[jj]
            if len(movedsets[i2] | movedsets[j2]) < d:
                pairs.add((i2, j2))
    seam_fail = []
    for (i2, j2) in pairs:
        ww = tuple(list(rev_word(seam_words[i2])) + list(seam_words[j2]))
        if window_chord_ok(ww, d, rng) is not True:
            seam_fail.append((i2, j2))
    report["seam_audit"] = eq_dup == 0 and not seam_fail
    log(f"  seam-cut audit ({len(seam_words)} seams, {len(pairs)} close pairs "
        f"deep-checked): {'PASS' if report['seam_audit'] else f'FAIL dup={eq_dup} {seam_fail[:5]}'}")

    # 6. U-claim
    gset = set(gpos)
    U = [g for i, g in enumerate(W) if i not in gset]
    t_naive = list(gs)
    t_conj = []
    for P, g in zip(prefixes, gs):
        t_conj.extend(list(P) + [g] + list(rev_word(P)))
    same_naive = all(run_word(U, s) == run_word(t_naive, s)
                     for s in probes)
    same_conj = all(run_word(U, s) == run_word(t_conj, s)
                    for s in probes)
    report["U_eq_gproduct"] = same_naive
    report["U_eq_conjugated"] = same_conj
    log(f"  U == g_1..g_k (naive claim):        {same_naive}")
    log(f"  U == prod P_i g_i P_i* (conjugated): {same_conj}")

    # 7. size accounting
    exp = sum(len(p) for p in parts) - 2 * d * (len(parts) - 1)
    report["size"] = (len(W), exp)
    log(f"  |W| = {len(W)} (expected {exp}; k(d+1)+2d = "
        f"{len(parts) * (d + 1) + 2 * d} if all |C_i|=3d+1)")
    return report


def all_cut_audit(W, D, rng=None, probes=24):
    """Full cyclic D-simplicity of an identity word: every cyclic window is a
    cut pair (u,v) with permutation p_v.p_u^-1 and need = min(arc, m-arc, D);
    wrap windows are inverses of linear ones (same geodesic), so scanning all
    linear pairs with that need is exhaustive.  Returns failing (u, arc, need).
    """
    W = tuple(W)
    m = len(W)
    if rng is None:
        rng = random.Random(31337)
    n = max(wires_of(W)) + 1
    st = [[rng.getrandbits(n) for _ in range(probes)]]
    for g in W:
        st.append([run_word((g,), s) for s in st[-1]])
    bad = []
    for u in range(m):
        for v in range(u + 1, m + 1):
            arc = v - u
            need = min(arc, m - arc, D)
            if need < 1:
                continue
            moved = 0
            for su, sv in zip(st[u], st[v]):
                moved |= su ^ sv
            if bin(moved).count("1") >= need:
                continue
            if window_chord_ok(W[u:v], need, rng) is not True:
                bad.append((u, arc, need))
    return bad


def compose_atoms(lib, d_in, length_min, seed=0, max_seeds=30, cap=300,
                  dsimple_cache=None, adj=None, min_support=None):
    """Compose library atoms at seam width d_in (screens at d_in) into a
    certified cyclically-d_in-simple identity of length >= length_min.
    Composition closure: seam-to-boundary chords equal d_in exactly, so the
    output is D-simple only for D <= d_in — compose AT the target simplicity.
    Returns (W, parts, metas) or raises."""
    if adj is None:
        adj = build_index(lib, d_in, cap=cap, minlen=2 * d_in + 2)[0]
    if dsimple_cache is None:
        dsimple_cache = {}
    for s in range(seed, seed + max_seeds):
        # enough parts that sum(L) - 2 d_in (k-1) >= length_min, assuming L~11
        k = max(2, (length_min - 2 * d_in) // (11 - 2 * d_in) + 1)
        try:
            parts, metas = build_string(lib, adj, d_in, k, seed=s,
                                        dsimple_cache=dsimple_cache,
                                        max_tries=600)
        except RuntimeError:
            continue
        W = tuple(assemble_run(parts, d_in))
        if len(W) < length_min:
            continue
        if min_support is not None and len(wires_of(W)) < min_support:
            continue
        rng = random.Random(s * 7 + 1)
        n = max(wires_of(W)) + 1
        if not all(run_word(W, st := rng.getrandbits(n)) == st
                   for _ in range(1024)):
            continue
        if all_cut_audit(W, d_in, rng):
            continue
        return W, parts, metas
    raise RuntimeError(f"compose_atoms: no certified atom in {max_seeds} seeds")


# -------------------------------------------------- targeted (wrt g_1..g_k) ---
#
# RC's refined definition (2026-08-30): given a target gate sequence g_1..g_k,
# a d-string of identities WRT g_1..g_k is C_1..C_k, each locally geodesic with
# >= 3d+1 gates, such that (a) the FIRST gate of C_i is g_i, and (b) the d-gate
# suffix of C_i equals the reverse of gates 2..d+1 of C_{i+1}.
# Writing C_i = g_i P_i M_i S_i (P_i = gates 2..d+1, S_i = last d gates,
# S_i = rev(P_{i+1})), identity-ness gives  M_i == rev(P_i) g_i P_{i+1}, so the
# concatenated middles telescope:
#     M_1 M_2 .. M_k  ==  rev(P_1) . g_1 g_2 .. g_k . rev(S_k)
# i.e. RC's uniform strip is exact up to the two boundary d-gate words; keeping
# P_1 in front and S_k in back yields EXACTLY g_1..g_k:
#     U  =  P_1 ++ M_1 ++ M_2 ++ ... ++ M_k ++ S_k  ==  g_1 g_2 .. g_k .

def build_joint_index(lib, d, cap=300, max_rot=None, seed=0):
    """Index identities by canon of their first d+1 gates (gate + d-prefix)."""
    minlen = 3 * d + 1
    rng = random.Random(seed)
    jindex = defaultdict(list)
    count = Counter()
    for idx, word in enumerate(lib):
        L = len(word)
        if L < minlen:
            continue
        rots = range(L) if max_rot is None else rng.sample(range(L), min(L, max_rot))
        for o in (0, 1):
            for r in rots:
                w = oriented(word, o, r)
                a = canon_word(w[:d + 1])
                count[a] += 1
                lst = jindex[a]
                if len(lst) < cap:
                    lst.append((idx, o, r))
                else:
                    j = rng.randrange(count[a])
                    if j < cap:
                        lst[j] = (idx, o, r)
    return dict(jindex), count


def build_targeted(lib, jindex, d, target, seed=None, dsimple_cache=None,
                   max_tries=4000, dsimple=True):
    """Sample a d-string of identities WRT the given target gate sequence.

    Free wires of each identity are mapped to fresh ancilla wires above the
    target's wires; ancillas net-restore through the telescoping."""
    rng = random.Random(seed)
    if dsimple_cache is None:
        dsimple_cache = {}
    k = len(target)
    minlen = 3 * d + 1
    parts = []
    metas = []
    fresh = [max(max(g) for g in target) + 1]

    def alloc():
        fresh[0] += 1
        return fresh[0] - 1

    all_keys = list(jindex)
    tries = 0
    dead = Counter()      # joint-pattern classes that keep failing
    while len(parts) < k:
        i = len(parts)
        g = tuple(target[i])
        if i == 0:
            prescribed = (g,)
            cands = None      # any identity/rotation matches a single gate
        else:
            prescribed = (g,) + tuple(rev_word(parts[-1][-d:]))
            cands = jindex.get(canon_word(prescribed), [])
        placed = False
        pool = (rng.sample(cands, len(cands)) if cands is not None
                else [rng.choice(jindex[rng.choice(all_keys)])
                      for _ in range(200)])
        for (idx, o, r) in pool:
            word = lib[idx]
            if dsimple and not cyclic_d_simple(word, d, dsimple_cache, (idx, d)):
                continue
            w = oriented(word, o, r)
            m0 = match_relabel(w[:len(prescribed)], prescribed)
            if m0 is None:
                continue
            base_used = set(m0.values())
            # free wires of w appearing in the suffix: steering them onto the
            # NEXT target gate's wires controls the next joint-pattern class
            suf_free = []
            for g0 in w[-d:]:
                for wi in g0:
                    if wi not in m0 and wi not in suf_free:
                        suf_free.append(wi)
            if i < k - 1:
                gw_next = [c for c in target[i + 1] if c not in base_used]
            else:
                gw_next = []
            options = [{}]
            if gw_next and suf_free:
                import itertools
                for jn in range(1, min(len(gw_next), len(suf_free)) + 1):
                    for sub in itertools.combinations(suf_free, jn):
                        for perm in itertools.permutations(gw_next, jn):
                            options.append(dict(zip(sub, perm)))
                options.sort(key=len, reverse=True)   # richer overlap first
                if len(options) > 240:
                    options = options[:240]
            for opt in options:
                m = dict(m0)
                m.update(opt)
                used = set(m.values())
                if len(used) != len(m):
                    continue
                ok = True
                for wi in wires_of(w):
                    if wi not in m:
                        while True:
                            c = alloc()
                            if c not in used:
                                break
                        m[wi] = c
                        used.add(c)
                rw = tuple(relabel(w, m))
                if i < k - 1:
                    nxt = (tuple(target[i + 1]),) + tuple(rev_word(rw[-d:]))
                    nk = canon_word(nxt)
                    if dead[nk] >= 2 or not jindex.get(nk):
                        continue
                parts.append(rw)
                metas.append((idx, o, r))
                placed = True
                break
            if placed:
                break
        if not placed:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"targeted walk stalled at part {i}")
            if cands is not None:
                dead[canon_word(prescribed)] += 1
            if tries % 400 == 0:
                del parts[:], metas[:]      # occasional full restart
            else:
                for _ in range(min(len(parts), 1 + rng.randrange(3))):
                    parts.pop()
                    metas.pop()
    return parts, metas


def assemble_targeted(parts, d):
    """U = P_1 ++ middles ++ S_k  (== g_1..g_k exactly)."""
    U = list(parts[0][1:d + 1])
    for p in parts:
        U.extend(p[d + 1:len(p) - d])
    U.extend(parts[-1][-d:])
    Urc = [g for p in parts for g in p[d + 1:len(p) - d]]
    return U, Urc


def verify_targeted(parts, target, d, trials=512, seed=321, log=print):
    k = len(target)
    ok_first = all(parts[i][0] == tuple(target[i]) for i in range(k))
    ok_len = all(len(p) >= 3 * d + 1 for p in parts)
    ok_seam = all(parts[i][-d:] == tuple(rev_word(parts[i + 1][1:d + 1]))
                  for i in range(k - 1))
    rng = random.Random(seed)
    n = max(max(wires_of(p)) for p in parts) + 1
    ok_id = True
    for p in parts:
        for _ in range(64):
            s0 = rng.getrandbits(n)
            if run_word(p, s0) != s0:
                ok_id = False
                break
        if not ok_id:
            break
    U, Urc = assemble_targeted(parts, d)
    tgt = [tuple(g) for g in target]
    ok_U = all(run_word(U, s := rng.getrandbits(n)) == run_word(tgt, s)
               for _ in range(trials))
    rc_naive = all(run_word(Urc, s := rng.getrandbits(n)) == run_word(tgt, s)
                   for _ in range(128))
    bound = (list(rev_word(parts[0][1:d + 1])) + tgt
             + list(rev_word(parts[-1][-d:])))
    rc_bound = all(run_word(Urc, s := rng.getrandbits(n)) == run_word(bound, s)
                   for _ in range(128))
    log(f"  parts: first-gate {'PASS' if ok_first else 'FAIL'}, "
        f"len>=3d+1 {'PASS' if ok_len else 'FAIL'}, "
        f"seam(b) {'PASS' if ok_seam else 'FAIL'}, "
        f"identities {'PASS' if ok_id else 'FAIL'}")
    log(f"  U = P_1++middles++S_k == g_1..g_k ({trials} states): "
        f"{'PASS' if ok_U else 'FAIL'}")
    log(f"  uniform-strip U_RC == g_1..g_k: {rc_naive}   "
        f"U_RC == rev(P_1).target.rev(S_k): {rc_bound}")
    log(f"  |U|={len(U)} |U_RC|={len(Urc)} target k={k} "
        f"(overhead {len(U)/max(1,k):.2f} gates/gate), "
        f"ancilla wires used: {n - (max(max(g) for g in target) + 1)}")
    return (ok_first and ok_len and ok_seam and ok_id and ok_U and rc_bound,
            U, Urc)


# -------------------------------------------------------------------- CLI ---

def cmd_analyze(args, lib):
    for d in args.d:
        minlen = 3 * d + 1
        elig = sum(1 for w in lib if len(w) >= minlen)
        if elig == 0:
            print(f"d={d}: NO identities of length >= {minlen} in the library "
                  f"-> (d,k)-strings impossible at this d from this database")
            continue
        adj_edges, out_count, node_adj, n_edges, n_ident = build_index(
            lib, d, cap=args.cap, max_rot=args.max_rot)
        sizes, sccs = scc_sizes(node_adj)
        selfloops = sum(1 for a, bs in node_adj.items() if a in bs)
        giant = sizes[0] if sizes else 0
        # cycle exists iff some SCC >1 or a self-loop
        cyc = giant > 1 or selfloops > 0
        # d-simplicity sample
        rng = random.Random(7)
        idxs = [i for i, w in enumerate(lib) if len(w) >= minlen]
        samp = rng.sample(idxs, min(args.dsample, len(idxs)))
        cache = {}
        good = sum(cyclic_d_simple(lib[i], d, cache, (i, d)) for i in samp)
        print(f"d={d}: eligible identities (len>={minlen}): {elig}"
              f" | edges (rot x orient): {n_edges}"
              f" | pattern-nodes: {len(node_adj)}"
              f" | giant SCC: {giant}"
              f" | nodes with self-loop: {selfloops}"
              f" | cycles exist: {cyc}  =>  k {'UNBOUNDED' if cyc else 'bounded'}"
              f" | cyclic-d-simple sample: {good}/{len(samp)}")


def cmd_build(args, lib):
    d, k = args.build_d, args.k
    adj_edges, *_ = build_index(lib, d, cap=args.cap, max_rot=args.max_rot)
    cache = {}
    parts, metas = build_string(lib, adj_edges, d, k, seed=args.seed,
                                commuting=args.commuting_prefix,
                                dsimple_cache=cache,
                                exact_len=args.exact_len,
                                n_wires=args.n_wires)
    W, gpos, prefixes, gs = assemble(parts, d)
    print(f"built (d={d},k={k}) string: |W|={len(W)}, "
          f"support={len(set(wires_of(W)))} wires, "
          f"{len(set(m[0] for m in metas))} distinct identities used")
    rep = verify_string(W, gpos, prefixes, gs, d, parts, trials=args.trials,
                        lcap=args.lcap)
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"# (d,k)-string identity W  d={d} k={k}  |W|={len(W)}\n")
            f.write(f"# g positions (0-based): {','.join(map(str, gpos))}\n")
            f.write(f"# parts (library idx,orient,rot): "
                    f"{';'.join(f'{i},{o},{r}' for i, o, r in metas)}\n")
            f.write(";".join(f"{a},{x},{y}" for (a, x, y) in W) + "\n")
        print(f"wrote {args.out}")
    hard_ok = (rep["identity"] and rep["no_adjacent_cancel"]
               and rep["no_identity_window"] and not rep["window_screen"][1])
    return 0 if hard_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--max-rot", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("analyze", help="seam-graph + feasibility per d")
    a1.add_argument("--d", type=int, nargs="+", default=[1, 2, 3, 4])
    a1.add_argument("--dsample", type=int, default=120)

    a2 = sub.add_parser("build", help="construct + verify a (d,k)-string")
    a2.add_argument("--d", dest="build_d", type=int, required=True)
    a2.add_argument("--k", type=int, required=True)
    a2.add_argument("--seed", type=int, default=None)
    a2.add_argument("--commuting-prefix", action="store_true")
    a2.add_argument("--exact-len", action="store_true")
    a2.add_argument("--n-wires", type=int, default=None)

    a3 = sub.add_parser("target", help="sample a d-string WRT a target circuit")
    a3.add_argument("--d", dest="t_d", type=int, required=True)
    a3.add_argument("--k", type=int, default=64, help="random target size")
    a3.add_argument("--n", type=int, default=128, help="target wire count")
    a3.add_argument("--target-file", default=None,
                    help="CSV word t,c1,c2;... instead of a random target")
    a3.add_argument("--seed", type=int, default=None)
    a3.add_argument("--trials", type=int, default=512)
    a3.add_argument("--out", default=None)
    a2.add_argument("--trials", type=int, default=1024)
    a2.add_argument("--lcap", type=int, default=24)
    a2.add_argument("--out", default=None)

    args = ap.parse_args()
    lib = load_library(args.lib)
    print(f"library: {len(lib)} identities, lengths "
          f"{dict(sorted(Counter(len(w) for w in lib).items()))}")
    if args.cmd == "analyze":
        cmd_analyze(args, lib)
    elif args.cmd == "build":
        sys.exit(cmd_build(args, lib))
    elif args.cmd == "target":
        rng = random.Random(args.seed)
        if args.target_file:
            with open(args.target_file) as f:
                line = [l for l in f if l.strip() and not l.startswith("#")][0]
            target = [tuple(int(t) for t in tok.split(","))
                      for tok in line.strip().split(";")]
        else:
            target = []
            for _ in range(args.k):
                a, x, y = rng.sample(range(args.n), 3)
                target.append((a, x, y))
        jindex, _ = build_joint_index(lib, args.t_d, cap=args.cap,
                                      max_rot=args.max_rot)
        print(f"joint index d={args.t_d}: {len(jindex)} (d+1)-gate patterns")
        parts, metas = build_targeted(lib, jindex, args.t_d, target,
                                      seed=args.seed)
        ok, U, Urc = verify_targeted(parts, target, args.t_d,
                                     trials=args.trials)
        if args.out:
            with open(args.out, "w") as f:
                f.write(f"# targeted d-string U == target  d={args.t_d} "
                        f"k={len(target)} |U|={len(U)}\n")
                f.write("# target=" +
                        ";".join(f"{a},{x},{y}" for (a, x, y) in target) + "\n")
                f.write("# parts=" +
                        ";".join(f"{i},{o},{r}" for i, o, r in metas) + "\n")
                f.write(";".join(f"{a},{x},{y}" for (a, x, y) in U) + "\n")
            print(f"wrote {args.out}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
