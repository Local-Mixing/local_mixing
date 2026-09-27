"""g57 core semantics, shared by the PR-identity generator and its audit harness.

Gate notation "a x y" (as in the curated CSV):  wire a ^= (x OR NOT y).
Bit-level:  s[a] ^= ((s>>x)&1) | (1 - ((s>>y)&1)).
All 26 curated atoms verify as identities under this rule, and it matches
src/circuit/xgate.rs.  (The sqlite DB's [x,z,y]=x+y(z+1)+1 is the same family
with the two controls' roles swapped; we standardise on the CSV rule here.)

A "word" is a list of (a, x, y) int tuples applied left-to-right.
"""
import numpy as np

Gate = tuple  # (a, x, y)


def apply_gate(g, s):
    a, x, y = g
    if ((s >> x) & 1) | (1 - ((s >> y) & 1)):
        return s ^ (1 << a)
    return s


def apply_word_int(word, s):
    for g in word:
        s = apply_gate(g, s)
    return s


def gate_table(g, n):
    """Permutation of 2^n states induced by a single gate."""
    a, x, y = g
    st = np.arange(1 << n, dtype=np.int64)
    fire = ((st >> x) & 1) | (1 - ((st >> y) & 1))
    return st ^ (fire << a)


def word_table(word, n):
    """Permutation table (2^n,) for a word on n wires; exact, small n only."""
    T = np.arange(1 << n, dtype=np.int64)
    for g in word:
        T = gate_table(g, n)[T]
    return T


def is_identity_exact(word, n):
    return np.array_equal(word_table(word, n), np.arange(1 << n, dtype=np.int64))


def is_identity_random(word, n, trials=4096, seed=None):
    """Monte-Carlo identity check on n wires, ANY n (arbitrary-precision ints).

    Runs `trials` random states through the word; True iff every one is fixed.
    Exact when it returns False; one-sided error when it returns True. Uses
    Python big-ints so it is correct for n > 63 (the n=128 target) where int64
    bit-parallel evaluation overflows on `1 << a`.
    """
    import random as _random
    rng = _random.Random(seed)
    for _ in range(trials):
        s = rng.getrandbits(n)
        cur = s
        for (a, x, y) in word:
            if ((cur >> x) & 1) | (1 - ((cur >> y) & 1)):
                cur ^= (1 << a)
        if cur != s:
            return False
    return True


def make_state_pool(n, trials=256, seed=0):
    """A fixed pool of random n-bit states as an int64 array, for batch checks."""
    rng = np.random.default_rng(seed)
    st = rng.integers(0, 1 << 62, size=trials, dtype=np.int64)
    if n > 62:
        st = st | (rng.integers(0, 1 << (n - 62), size=trials, dtype=np.int64) << 62)
    return st % (1 << n) if n < 62 else st


def apply_word_batch(word, states):
    """Apply a word to an int64 array of states (bit-parallel). Returns new array."""
    st = states
    for (a, x, y) in word:
        fire = ((st >> x) & 1) | (1 - ((st >> y) & 1))
        st = st ^ (fire << a)
    return st


def window_is_identity(word, pool):
    """True iff `word` fixes every state in `pool` (one-sided: exact when False)."""
    return bool(np.array_equal(apply_word_batch(word, pool), pool))


def wires_of(word):
    return sorted({w for g in word for w in g})


def support_size(word):
    return len(wires_of(word))


def targets_of(word):
    return [g[0] for g in word]


def relabel(word, mapping):
    return [(mapping[a], mapping[x], mapping[y]) for (a, x, y) in word]


def canonical_support(word):
    """Relabel a word's wires to 0..m-1 (first-seen order). Returns (word', order)."""
    order = []
    seen = {}
    for g in word:
        for w in g:
            if w not in seen:
                seen[w] = len(order)
                order.append(w)
    return relabel(word, seen), order


def parse_csv_circuit(field):
    """'012;021;...' -> [(0,1,2),(0,2,1),...]."""
    out = []
    for tok in field.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        out.append((int(tok[0]), int(tok[1]), int(tok[2])))
    return out


def fmt_word(word):
    return ";".join(f"{a}{x}{y}" if max(a, x, y) < 10 else f"{a},{x},{y}"
                    for (a, x, y) in word)
