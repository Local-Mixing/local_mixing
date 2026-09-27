"""Fingerprint dashboard: how far a generated identity is from a random one.

A random identity word of length m (at poly(n) length, far below the group
diameter) essentially never contains a short contiguous sub-identity, has no
repeated gates beyond chance, and does not factor into commuting identity
blocks.  Each metric below measures a deviation an adversary could exploit.

  1. id_windows        contiguous sub-words that compute the identity
                       (shortest ones are the sharpest fingerprint)
  2. repeated_gates    exact oriented-gate repeats beyond the random baseline
  3. write_collision   fraction of length-t windows whose targets collide
  4. factorization     the real attack: greedily peel off commuting identity
                       blocks; how much of the word is recovered as atoms

All identity tests are bit-parallel over a fixed random state pool (exact when
they report False; one-sided error the other way).  Each metric is reported
against a length-matched uniform-random word baseline.
"""
import random
from collections import Counter

import numpy as np

import g57


def id_windows(word, n, max_len=12, pool=None):
    """Contiguous windows (len 2..max_len) that compute the identity."""
    if pool is None:
        pool = g57.make_state_pool(n, trials=256, seed=0)
    L = len(word)
    hits = []
    for i in range(L):
        cur = pool.copy()
        jmax = min(max_len, L - i)
        for step in range(jmax):
            g = word[i + step]
            a, x, y = g
            fire = ((cur >> x) & 1) | (1 - ((cur >> y) & 1))
            cur = cur ^ (fire << a)
            wlen = step + 1
            if wlen >= 2 and np.array_equal(cur, pool):
                hits.append((i, wlen))
    shortest = min((h[1] for h in hits), default=None)
    return {"count": len(hits), "shortest": shortest, "hits": hits[:20]}


def repeated_gates(word):
    c = Counter(word)
    return sum(v - 1 for v in c.values() if v > 1)


def write_collision(word, t):
    if len(word) < t:
        return 0.0
    hits = sum(1 for i in range(len(word) - t + 1)
               if len({g[0] for g in word[i:i + t]}) < t)
    return hits / (len(word) - t + 1)


def factorization_attack(word, n, max_block=15, pool=None, max_passes=8):
    """Greedy identity-block peel: how much of the word an adversary can carve
    into isolated contiguous identity sub-blocks.  High = transparent structure.

    Left-to-right greedy; after peeling at i, continue near i (removal can expose
    a new boundary) rather than restarting -- bounds work to ~max_passes*L*max_block
    vectorised tests.  Contiguous only (a commuting-aware peeler is stronger; this
    is a fast proxy).
    """
    if pool is None:
        pool = g57.make_state_pool(n, trials=64, seed=1)
    remaining = list(word)
    peeled = 0
    for _ in range(max_passes):
        L = len(remaining)
        progressed = False
        i = 0
        while i < L - 1:
            cur = pool.copy()
            found = 0
            for step in range(min(max_block, L - i)):
                a, x, y = remaining[i + step]
                fire = ((cur >> x) & 1) | (1 - ((cur >> y) & 1))
                cur = cur ^ (fire << a)
                if step + 1 >= 2 and np.array_equal(cur, pool):
                    found = step + 1
                    break
            if found:
                del remaining[i:i + found]
                peeled += found
                L -= found
                progressed = True
                i = max(0, i - max_block)
            else:
                i += 1
        if not progressed:
            break
    return {"peeled_gates": peeled, "total": len(word),
            "fraction_recovered": peeled / max(1, len(word)),
            "residual": len(remaining)}


def random_word(n, length, rng):
    out = []
    for _ in range(length):
        a, x, y = rng.sample(range(n), 3)
        out.append((a, x, y))
    return out


def full_report(word, n, label="word", seed=0):
    rng = random.Random(seed)
    rand = random_word(n, len(word), rng)
    poolA = g57.make_state_pool(n, trials=256, seed=0)
    poolB = g57.make_state_pool(n, trials=128, seed=1)
    return {
        "label": label,
        "length": len(word),
        "id_windows": id_windows(word, n, pool=poolA),
        "repeated_gates": {"word": repeated_gates(word), "random": repeated_gates(rand)},
        "write_collision_gen_vs_rand": {
            t: (round(write_collision(word, t), 3), round(write_collision(rand, t), 3))
            for t in (8, 12, 16)},
        "factorization": {"word": factorization_attack(word, n, pool=poolB),
                          "random": factorization_attack(rand, n, pool=poolB)},
    }


def print_report(rep):
    print(f"--- audit: {rep['label']}  (length {rep['length']}) ---")
    idw = rep["id_windows"]
    print(f"  id-windows            : {idw['count']:>4}  shortest={idw['shortest']}")
    rg = rep["repeated_gates"]
    print(f"  repeated gates        : {rg['word']:>4}  (random baseline {rg['random']})")
    print(f"  write-collision t=8/12/16 (gen vs rand): "
          + ", ".join(f"{t}:{v[0]}/{v[1]}" for t, v in rep["write_collision_gen_vs_rand"].items()))
    fw, fr = rep["factorization"]["word"], rep["factorization"]["random"]
    print(f"  factorization recovered: {fw['fraction_recovered']*100:5.1f}%  "
          f"(residual {fw['residual']}/{fw['total']})   "
          f"random baseline {fr['fraction_recovered']*100:.1f}%")
