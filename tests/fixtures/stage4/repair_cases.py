#!/usr/bin/env python3
"""repair_cases.py OUT [NAME SEED ...] : deterministic synthetic inputs for the
repair fixtures (docs/STAGE4_RUST_DESIGN.md §10.3, the crafted repair cases).

Each case is a source C (g57, n = 8 wires, random non-X gates) and a
"packed" circuit P on 4n = 32 wires with a window lattice (bounds). P runs
C's gates verbatim on the data wires 0..n, so every value C computes sits on
a P wire while it lives: interior boundaries expose non-output values, which
is what the fix, merge and split stages act on. Between C's gates P has
noise gates on the ancilla wires n..4n (controls on data and ancilla wires,
both polarities, some complemented): a noise gate that reads a data wire
blocks C's writes of that wire from sliding past it. Identical gate pairs
(also nested A B B A, which need two cancel passes) are inserted where they
commute. P is not C's function on the ancillas; the repair only needs
(P, bounds, C).

`orderdep` is different: P (160 wires) has windows where t ^= c1*c2 with
c1, c2 copies of the same sum of k inputs (anf_cases.py's construction), so
the product's CPython 3.9 prefix exceeds the split's product cap while the
finished product fits: whether such a window fits the budget (and so whether
it is split) is decided by the set iteration order. The second product and
the uncopies are written as different but equal gates (comp flipped, or
`c ^= NOT(NOT x)`), so the cancel pass keeps them.

Seeds are fixed (CALIBRATION material only). Without arguments every case
of CASES is written; `NAME SEED` overrides a case's seed (the seed search).
Writes OUT/<name>.c.g57, OUT/<name>.packed.mpmct1, OUT/<name>.packed.bounds.
"""
import os
import random
import sys

N = 8
NW = 4 * N

# name: (seed, C gates, noise per C gate (mean), P(noise reads a data wire),
#        cancel pairs, bound spacing in gates)
CASES = {
    "leaky": (1, 24, 6, 0.25, 30, 9),
    "dense": (1, 16, 12, 0.6, 20, 7),
    "sparse": (1, 16, 3, 0.05, 10, 5),
}


def g57_gate(rng):
    a = rng.randrange(N)
    x, y = rng.sample([w for w in range(N) if w != a], 2)
    return a, x, y


def from_g57(a, x, y):
    """XGate::from_g57: a ^= x OR NOT y, i.e. comp=1 on (x=0 AND y=1)."""
    lits = sorted([(x, 0), (y, 1)])
    return (a, 1, tuple(c for c, _ in lits), tuple(p for _, p in lits))


def noise_gate(rng, p_data):
    t = rng.randrange(N, NW)
    k = rng.choice([1, 1, 2, 2, 3])
    cps = set()
    while len(cps) < k:
        if rng.random() < p_data:
            cps.add(rng.randrange(N))
        else:
            c = rng.randrange(N, NW)
            if c != t:
                cps.add(c)
    cps = tuple(sorted(cps))
    return (t, int(rng.random() < 0.2), cps, tuple(rng.randrange(2) for _ in cps))


def commutes(g, h):
    return g[0] not in h[2] and h[0] not in g[2]


def write_mpmct(path, n, gates):
    with open(path, "w") as f:
        f.write(f"mpmct1 {n} {len(gates)}\n")
        for t, comp, cps, pol in gates:
            parts = [str(t), str(comp), str(len(cps))]
            for c, p in zip(cps, pol):
                parts += [str(c), str(p)]
            f.write(" ".join(parts) + "\n")


def make(out, name, seed, nc, noise, p_data, pairs, spacing):
    rng = random.Random(seed)
    c = [g57_gate(rng) for _ in range(nc)]
    with open(os.path.join(out, f"{name}.c.g57"), "w") as f:
        f.write("".join(f"{a}{x}{y};" for a, x, y in c))
    p = []
    for g in c:
        for _ in range(rng.randrange(2 * noise + 1)):
            p.append(noise_gate(rng, p_data))
        p.append(from_g57(*g))
    for _ in range(rng.randrange(2 * noise + 1)):
        p.append(noise_gate(rng, p_data))
    # cancelling pairs around commuting stretches; some nested (A B B A)
    for _ in range(pairs):
        j = rng.randrange(len(p) + 1)
        a = noise_gate(rng, p_data)
        span = rng.randrange(0, 6)
        mid = p[j:j + span]
        if not all(commutes(a, h) for h in mid):
            continue
        if rng.random() < 0.4:
            b = noise_gate(rng, p_data)
            if not commutes(a, b):
                p[j:j + span] = [a, b] + mid + [b, a]
                continue
        p[j:j + span] = [a] + mid + [a]
    m = len(p)
    bounds = []
    b = 0
    while True:
        b += max(1, spacing + rng.randrange(-2, 3))
        if b >= m:
            break
        bounds.append(b)
    write_mpmct(os.path.join(out, f"{name}.packed.mpmct1"), NW, p)
    with open(os.path.join(out, f"{name}.packed.bounds"), "w") as f:
        f.write("\n".join(map(str, bounds)) + "\n")


def make_orderdep(out, name, seed):
    rng = random.Random(seed)
    c = [g57_gate(rng) for _ in range(16)]
    with open(os.path.join(out, f"{name}.c.g57"), "w") as f:
        f.write("".join(f"{a}{x}{y};" for a, x, y in c))
    nw = 160
    p, bounds = [], []
    for k in (48, 64, 80):
        xs = rng.sample(range(0, 150), k)
        c1, c2, t = rng.sample([w for w in range(nw) if w not in xs], 3)
        copy = [(c1, 0, (x,), (1,)) for x in xs] + [(c2, 0, (x,), (1,)) for x in xs]
        rng.shuffle(copy)
        uncopy = [(w, 1, cps, (0,)) for w, _, cps, _ in copy[::-1]]
        comp = rng.randrange(2)
        ctl = tuple(sorted([c1, c2]))
        prod = [(t, comp, ctl, (1, 1)), (t, 1 - comp, ctl, (1, 1))]
        tail = [(t, 0, (xs[0],), (1,))]
        p += copy + prod + tail + uncopy
        bounds.append(len(p))
    write_mpmct(os.path.join(out, f"{name}.packed.mpmct1"), nw, p)
    with open(os.path.join(out, f"{name}.packed.bounds"), "w") as f:
        f.write("\n".join(map(str, bounds)) + "\n")


def main():
    out = sys.argv[1]
    over = dict(zip(sys.argv[2::2], map(int, sys.argv[3::2])))
    os.makedirs(out, exist_ok=True)
    for name, (seed, *rest) in CASES.items():
        make(out, name, over.get(name, seed), *rest)
    make_orderdep(out, "orderdep", over.get("orderdep", 1))


if __name__ == "__main__":
    main()
