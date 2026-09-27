"""Randomized rejection-minting search for repeat-free multi-target locally-geodesic
g57 identities ("atoms").

STRATEGY: birthday meet-in-the-middle with structure bias toward SMALL support.

Method
------
For a fixed support s (small: 2^s <= 256) and half-lengths (m1, m2):
  1. draw N random REDUCED words X of length m1; bucket them by the permutation
     they compute:  store[perm(X)] .append(X)   (cap K words per perm),
  2. draw N random REDUCED words Y of length m2; look up store[ inv(perm(Y)) ];
     for every stored X there, perm(Y) o perm(X) = perm(Y) o perm(Y)^{-1} = id,
     so the concatenation  X + Y  is an EXACT identity by construction,
  3. REJECT unless: repeat-free (all gate triples distinct), >=3 distinct targets,
     locally-geodesic (no PROPER contiguous subword is the identity).
X and Y are INDEPENDENT random walks that meet at a common permutation, so their
gate sets differ -> repeat-free and multi-target survive rejection (unlike a
walk+its-own-inverse, which reuses gates).  Repeat-free => automatically reduced
(no adjacent-equal / commuting-equal pair), so locally-geodesic reduces to the
no-internal-identity-window check.

Why support 4 wins: on 4 wires a repeat-free identity with >=3 distinct targets is
forced to overlap densely, so it is automatically NON-FACTORING (connected
gate-commutation graph).  On support >=5 the birthday collisions concentrate on
low-support permutations, so the identities that survive are almost never 3-target.

Every accepted atom is EXACTLY re-verified (all 5 properties) before it is written.
"""
import itertools
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g57


# ---------------------------------------------------------------------------
# exact property checks
# ---------------------------------------------------------------------------
def repeat_free(word):
    return len(set(word)) == len(word)


def commutes(g, h):
    return (g[0] not in (h[0], h[1], h[2])) and (h[0] not in (g[0], g[1], g[2]))


def is_reduced(word):
    L = len(word)
    for i in range(L):
        for j in range(i + 1, L):
            if word[i] == word[j] and all(commutes(word[i], word[k])
                                          for k in range(i + 1, j)):
                return False
    return True


def locally_geodesic(word, n):
    """No PROPER contiguous subword computes the identity (single gates skipped:
    a g57 gate is never the identity)."""
    L = len(word)
    for i in range(L):
        for j in range(i + 2, L + 1):
            if i == 0 and j == L:
                continue
            if g57.is_identity_exact(word[i:j], n):
                return False
    return True


def targets(word):
    return {g[0] for g in word}


def non_factoring(word):
    """Gate-commutation graph connected via NON-commuting edges."""
    L = len(word)
    if L == 0:
        return False
    adj = [[] for _ in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            if not commutes(word[i], word[j]):
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * L
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                stack.append(v)
    return cnt == L


def full_verify(word):
    """(ok, non_factor): ok iff all 5 core properties hold, exactly."""
    if not word:
        return False, False
    n = max(w for g in word for w in g) + 1
    if not repeat_free(word):
        return False, False
    if len(targets(word)) < 3:
        return False, False
    if not g57.is_identity_exact(word, n):
        return False, False
    if not is_reduced(word):
        return False, False
    if not locally_geodesic(word, n):
        return False, False
    return True, non_factoring(word)


def fmt_atom(word):
    return ";".join(f"{a},{x},{y}" for (a, x, y) in word)


# ---------------------------------------------------------------------------
# birthday meet-in-the-middle engine
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self, support):
        self.support = support
        self.D = 1 << support
        assert self.D <= 256
        self.gates = list(itertools.permutations(range(support), 3))
        self.gts = {g: g57.gate_table(g, support).astype(np.uint8) for g in self.gates}
        self.ID = np.arange(self.D, dtype=np.uint8)

    def table(self, w):
        T = self.ID.copy()
        for g in w:
            T = self.gts[g][T]
        return T

    def rand_word(self, L, rng):
        w = []
        gates = self.gates
        for _ in range(L):
            g = rng.choice(gates)
            while w and g == w[-1]:
                g = rng.choice(gates)
            w.append(g)
        return w

    def run(self, m1, m2, N, K, rng, seen_canon, sink):
        """One config. Returns (attempts, collisions, found, nonfac)."""
        ID = self.ID
        store = {}
        for _ in range(N):
            X = self.rand_word(m1, rng)
            b = self.table(X).tobytes()
            lst = store.get(b)
            if lst is None:
                store[b] = [X]
            elif len(lst) < K:
                lst.append(X)
        attempts = N
        coll = found = nonfac = 0
        for _ in range(N):
            Y = self.rand_word(m2, rng)
            fY = self.table(Y)
            inv = np.empty_like(fY)
            inv[fY] = ID
            lst = store.get(inv.tobytes())
            if not lst:
                continue
            for X in lst:
                coll += 1
                w = X + Y
                if not repeat_free(w):
                    continue
                if len({g[0] for g in w}) < 3:
                    continue
                ok, nf = full_verify(w)
                if not ok:
                    continue
                cw, _ = g57.canonical_support(w)
                key = tuple(cw)
                if key in seen_canon:
                    continue
                seen_canon.add(key)
                found += 1
                if nf:
                    nonfac += 1
                sink.append((w, nf))
        return attempts, coll, found, nonfac


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randrange(1 << 30)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    outdir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(outdir, "search-mint.csv")

    seen_canon = set()
    sink = []
    K = 8

    # (support, m1, m2): support 4 is the productive regime.  A couple of
    # support-5 configs are kept only as a documented (near-zero-yield) control.
    focus = os.environ.get("MINT_FOCUS", "1") == "1"
    configs = []
    if focus:
        for (m1, m2) in [(5, 5), (6, 6), (6, 7), (7, 7), (7, 8)]:
            configs.append((4, m1, m2))
        configs.append((5, 4, 4))  # control: support-5 collides but ~never 3-target
    else:
        for (m1, m2) in [(5, 5), (5, 6), (6, 6), (6, 7), (7, 7), (7, 8), (8, 8)]:
            configs.append((4, m1, m2))
        for (m1, m2) in [(4, 4), (4, 5), (5, 5), (5, 6)]:
            configs.append((5, m1, m2))

    engines = {}
    total_attempts = 0
    # per-config cumulative stats across rounds
    from collections import defaultdict
    cstat = defaultdict(lambda: [0, 0, 0, 0])  # att, coll, found, nonfac
    print(f"# SEED={seed} N={N} rounds={rounds} K={K}", file=sys.stderr, flush=True)
    for rnd in range(rounds):
        rng = random.Random((seed + 1) * 1000003 + rnd)
        for (support, m1, m2) in configs:
            eng = engines.get(support)
            if eng is None:
                eng = engines[support] = Engine(support)
            att, coll, found, nf = eng.run(m1, m2, N, K, rng, seen_canon, sink)
            total_attempts += att
            s = cstat[(support, m1, m2)]
            s[0] += att; s[1] += coll; s[2] += found; s[3] += nf
        print(f"# round {rnd}: cumulative unique atoms = {len(sink)}",
              file=sys.stderr, flush=True)
    for (support, m1, m2), (att, coll, found, nf) in sorted(cstat.items()):
        yld = found / att * 10000 if att else 0.0
        print(f"s={support} m=({m1},{m2}) N={att} coll={coll:8d} "
              f"found={found:4d} nonfac={nf:4d} yield/10k={yld:6.3f}",
              file=sys.stderr, flush=True)

    # write all unique verified atoms
    with open(outfile, "w") as f:
        for (w, nf) in sink:
            f.write(fmt_atom(w) + "\n")

    total = len(sink)
    total_nf = sum(1 for (_, nf) in sink if nf)
    lens = [len(w) for (w, _) in sink]
    tg = [len({g[0] for g in w}) for (w, _) in sink]
    print("=" * 60, file=sys.stderr)
    print(f"TOTAL attempts={total_attempts}", file=sys.stderr)
    print(f"TOTAL unique verified atoms: {total}", file=sys.stderr)
    print(f"  non-factoring (connected): {total_nf}", file=sys.stderr)
    if total:
        print(f"  length range: {min(lens)}..{max(lens)}", file=sys.stderr)
        print(f"  target range: {min(tg)}..{max(tg)}", file=sys.stderr)
        print(f"  overall yield/10k attempts: {total/total_attempts*10000:.4f}",
              file=sys.stderr)
    print(f"written to {outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
