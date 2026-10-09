#!/usr/bin/env python3
"""pack_cases.py OUT : deterministic synthetic (A, B) pairs for the packer
fixtures (docs/STAGE4_RUST_DESIGN.md §10.3, `wide300` / `wide128` / `w1024`).

A is a random circuit; B is A rewritten by legal commutations (adjacent swaps
of gates that do not read each other's target), with inserted cancelling
pairs and, for `wide128`, copy/uncopy CNOT pairs into extra ancilla wires
(B is wider than A). B computes A's function on A's wires, so the packer's
VERIFY holds. Gates use 0..4 controls, both polarities and complements, so
the window ANF grows quickly and small monomial budgets force compose
aborts. Seeds are fixed (CALIBRATION material only).
"""
import os
import random
import sys


def rand_gate(rng, wires, kdist):
    t = rng.choice(wires)
    k = rng.choice(kdist)
    others = [w for w in wires if w != t]
    cps = sorted(rng.sample(others, k))
    return (t, rng.randrange(2), tuple(cps), tuple(rng.randrange(2) for _ in cps))


def commutes(g, h):
    return g[0] not in h[2] and h[0] not in g[2]


def shuffle(rng, gates, passes):
    for _ in range(passes):
        i = rng.randrange(len(gates) - 1)
        if commutes(gates[i], gates[i + 1]):
            gates[i], gates[i + 1] = gates[i + 1], gates[i]


def write(path, n, gates):
    with open(path, "w") as f:
        f.write(f"mpmct1 {n} {len(gates)}\n")
        for t, comp, cps, pol in gates:
            parts = [str(t), str(comp), str(len(cps))]
            for c, p in zip(cps, pol):
                parts += [str(c), str(p)]
            f.write(" ".join(parts) + "\n")


def make(out, name, seed, na, nb, wires, m, kdist, pairs, copies, passes, local):
    rng = random.Random(seed)
    # A: gates on a sliding block of `local` wires so values stay local
    # (short spans, real intervals), drifting across all of `wires`.
    A = []
    for i in range(m):
        lo = (i * len(wires) // m) % max(1, len(wires) - local + 1)
        A.append(rand_gate(rng, wires[lo:lo + local], kdist))
    B = list(A)
    for _ in range(pairs):
        j = rng.randrange(len(B) + 1)
        lo = rng.randrange(max(1, len(wires) - local + 1))
        g = rand_gate(rng, wires[lo:lo + local], kdist)
        B[j:j] = [g, g]
    anc = list(range(na, nb))
    for _ in range(copies):
        j = rng.randrange(len(B) + 1)
        w = rng.choice(wires)
        a = rng.choice(anc)
        cp = (a, 0, (w,), (1,))
        B[j:j] = [cp, cp]
    shuffle(rng, B, passes)
    write(os.path.join(out, f"{name}_a.mpmct1"), na, A)
    write(os.path.join(out, f"{name}_b.mpmct1"), nb, B)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    # 300 wires (W = 8; wire ids >= 61, >= 128, >= 256: data/mid/band classes)
    make(out, "wide300", 3001, 300, 300, list(range(300)), 2400,
         [0, 1, 2, 2, 3, 3, 4], 300, 0, 60000, 24)
    # A on 128 wires, B on 160 (W = 4 for n = 160): ancilla copy/uncopy pairs
    make(out, "wide128", 1281, 128, 160, list(range(128)), 2000,
         [1, 2, 2, 3, 3, 4], 200, 150, 50000, 20)
    # 1024 wires (W = 16; wire ids up to 1023)
    make(out, "w1024", 10241, 1024, 1024, list(range(1024)), 1500,
         [0, 1, 2, 3, 3], 100, 0, 30000, 40)


if __name__ == "__main__":
    main()
