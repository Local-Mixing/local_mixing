"""Atom library and small-support resynthesis oracle for the PR-identity generator.

An *atom* is a g57 word that computes the identity permutation on its own
support (a "locally geodesic" / minimal identity, or any verified identity).
Atoms come from three sources:

  1. the curated CSV  (26 hand-verified minimal identities, 6-15 gates),
  2. minted class-O identities  (doubled even-degree cycles, any size),
  3. the BFS-MITM sampler in sampler.py  (random minimal identities <= ~12 gates).

The resynthesis oracle BFS-explores g57 words on a small canonical support and
answers "give me a different word for this permutation" -- the engine behind
local-rewrite laundering.  It is exact (permutation equality on all 2^m states).
"""
import csv
import itertools
import random
from functools import lru_cache

import numpy as np

import g57


# ----------------------------------------------------------------------------
# 1. curated CSV atoms
# ----------------------------------------------------------------------------
def load_curated(path):
    atoms = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("name"):
                continue
            word = g57.parse_csv_circuit(r["circuit"])
            n = g57.support_size(word)
            assert g57.is_identity_exact(word, max(g57.wires_of(word)) + 1), \
                f"{r['name']} not an identity"
            atoms.append({"name": r["name"], "word": word, "len": len(word),
                          "support": n, "source": "curated"})
    return atoms


def load_identity_library(path, max_len=None, per_len_cap=None, rng=None):
    """Load the DB-extracted locally-geodesic identities (format from
    extract_identities.rs: `length,n_wires,circuit` where circuit is
    `t,c1,c2;...`, split on the first two commas). Optionally cap by length and
    subsample per length for a manageable, length-balanced atom pool."""
    from collections import defaultdict
    buckets = defaultdict(list)
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            L, _nw, circ = line.split(",", 2)
            L = int(L)
            if max_len is not None and L > max_len:
                continue
            word = [tuple(int(z) for z in tok.split(",")) for tok in circ.split(";")]
            buckets[L].append(word)
    atoms = []
    for L, words in buckets.items():
        if per_len_cap is not None and len(words) > per_len_cap:
            words = (rng or random).sample(words, per_len_cap)
        for w in words:
            atoms.append({"name": f"id{L}", "word": w, "len": L,
                          "support": g57.support_size(w), "source": "db-library"})
    return atoms


# ----------------------------------------------------------------------------
# 2. class-O minting: doubled even-degree cycles
# ----------------------------------------------------------------------------
def mint_doubled_cycle(k, rng, target=0, ctrl_wires=None):
    """Both orientations of each edge of a random k-cycle, all targeting `target`.

    Yields a 2k-gate identity on wires {target} + k control wires.  Every gate is
    distinct (no repeated-gate fingerprint); the cancellation is linear-algebraic
    (quadratic monomials pair off, linear parts cancel since every vertex has
    even degree 2).  Verified before returning.
    """
    if ctrl_wires is None:
        ctrl_wires = list(range(1, k + 1))
    cyc = rng.sample(ctrl_wires, k)
    gates = []
    for i in range(k):
        u, v = cyc[i], cyc[(i + 1) % k]
        gates += [(target, u, v), (target, v, u)]
    rng.shuffle(gates)  # commuting family: any order is still the identity
    n = max(max(g) for g in gates) + 1
    assert g57.is_identity_exact(gates, n), "minted doubled cycle is not identity"
    return gates


def mint_class_o_library(rng, sizes=(3, 4, 5, 6), per_size=4, span=8):
    """A batch of minted class-O atoms on a canonical span of `span` control wires."""
    atoms = []
    for k in sizes:
        for _ in range(per_size):
            word = mint_doubled_cycle(k, rng, target=0, ctrl_wires=list(range(1, span + 1)))
            word, _ = g57.canonical_support(word)
            atoms.append({"name": f"O{k}", "word": word, "len": len(word),
                          "support": g57.support_size(word), "source": "minted-O"})
    return atoms


# ----------------------------------------------------------------------------
# 3. resynthesis oracle: BFS words on a canonical small support
# ----------------------------------------------------------------------------
class Resynth:
    """BFS-based equivalent-word oracle on <= max_wires canonical wires.

    For a permutation of 2^m states (m wires, m small), returns alternative g57
    words realising it.  Used to (a) locally rewrite windows to break syntactic
    fingerprints, and (b) complete meet-in-the-middle identity samples.
    """

    def __init__(self, max_wires=4, radius=4):
        # ball sizes (measured): m=4 -> r4:165k(0.2s), r5:2.8M(4s); m=5 explodes.
        # Defaults kept cheap; the parent DB (complete <=6 gates) replaces this
        # oracle with a lookup when available.
        self.max_wires = max_wires
        self.radius = radius
        self._ball = {}       # m -> {perm_bytes: shortest_word}
        self._by_perm = {}    # m -> {perm_bytes: [words up to radius]}

    def _gates(self, m):
        return list(itertools.permutations(range(m), 3))

    def _build(self, m):
        if m in self._ball:
            return
        ID = np.arange(1 << m, dtype=np.int64)
        gts = {g: g57.gate_table(g, m) for g in self._gates(m)}
        ball = {ID.tobytes(): ()}
        by_perm = {ID.tobytes(): [()]}
        frontier = {ID.tobytes(): ()}
        for _ in range(self.radius):
            nf = {}
            for pb, w in frontier.items():
                P = np.frombuffer(pb, dtype=np.int64)
                for g, tb in gts.items():
                    npb = (tb[P]).tobytes()
                    by_perm.setdefault(npb, [])
                    if len(by_perm[npb]) < 8:
                        by_perm[npb].append(w + (g,))
                    if npb not in ball:
                        ball[npb] = w + (g,)
                        nf[npb] = w + (g,)
            frontier = nf
            if not frontier:
                break
        self._ball[m] = ball
        self._by_perm[m] = by_perm

    def perm_of(self, word, m):
        return g57.word_table(word, m).tobytes()

    def alternatives(self, word, exclude_self=True, max_len=None, exact_len=None):
        """BFS words equal to `word` as a permutation on its canonical support.

        exact_len : if set, return only alternatives of exactly this length
                    (length-preserving swaps -- the laundering default, so the
                    circuit decorrelates without growing).
        max_len   : if set, cap alternative length.
        """
        cword, order = g57.canonical_support(word)
        m = len(order)
        if m > self.max_wires:
            return []
        self._build(m)
        pb = self.perm_of(cword, m)
        alts = self._by_perm[m].get(pb, [])
        back = list(order)  # canonical index -> original wire
        out = []
        cset = tuple(cword)
        for w in alts:
            if exclude_self and tuple(w) == cset:
                continue
            if exact_len is not None and len(w) != exact_len:
                continue
            if max_len is not None and len(w) > max_len:
                continue
            out.append(g57.relabel(list(w), {i: back[i] for i in range(m)}))
        return out

    def shortest_len(self, word):
        cword, order = g57.canonical_support(word)
        m = len(order)
        if m > self.max_wires:
            return None
        self._build(m)
        w = self._ball[m].get(self.perm_of(cword, m))
        return None if w is None else len(w)
