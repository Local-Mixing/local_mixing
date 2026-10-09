#!/usr/bin/env python3
"""synth.py OUT KIND SEED : random small stage-4 inputs for the CI differential
test of the Python oracle against tdp_packing (docs/STAGE4_RUST_DESIGN.md §10.6).

KIND=pack   OUT/a.mpmct1, OUT/b.mpmct1 (B = A rewritten by legal commutations,
            cancelling pairs and ancilla copy/uncopy pairs; B computes A on A's
            wires) and prints the packer argv `W_cap G B_mono span_cap`, drawn
            around the zone where compose aborts (and CPython-order-dependent
            aborts) happen.
KIND=repair OUT/c.g57 (n in {4, 6, 8, 12}), OUT/p.mpmct1 on 4n wires (C's gates
            verbatim on the data wires plus noise, cancelling pairs, nested
            pairs) and OUT/p.bounds; prints `n split_budget fix_iterations
            fallback_iterations`.
The generators are the fixture generators (tests/fixtures/stage4/pack_cases.py,
repair_cases.py) with randomized parameters. Seeds are CALIBRATION material
(derived from the CI run id), printed by the harness, never deliverable seeds.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fixtures", "stage4"))
import pack_cases  # noqa: E402
import repair_cases  # noqa: E402

# repair_cases writes C as f"{a}{x}{y};", which is the base-83 g57 text only
# for wires < 10; wires >= 10 need their base-83 digit (circuit/formats/g57.rs).
G57_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"


class Wire(int):
    """An int that formats (f-strings) as its base-83 g57 digit; str() stays decimal."""

    def __format__(self, spec):
        return G57_ALPHABET[int(self)]


_g57_gate = repair_cases.g57_gate
repair_cases.g57_gate = lambda rng: tuple(Wire(v) for v in _g57_gate(rng))


def main():
    out, kind, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    if kind == "pack":
        na = rng.choice([8, 12, 16, 24, 40, 70])
        nb = na + rng.choice([0, 0, 0, 8, 24])
        kdist = rng.choice([[0, 1, 1, 2], [1, 2, 2], [2, 3, 3, 4], [0, 1, 1, 1, 2, 2]])
        local = min(na, max(max(kdist) + 2, rng.choice([4, 6, 8, 12, 16])))
        m = rng.randrange(150, 1200)
        pack_cases.make(out, "x", rng.randrange(1 << 30), na, nb, list(range(na)), m, kdist,
                        rng.randrange(0, 60), rng.randrange(0, 40) if nb > na else 0,
                        rng.randrange(1000, 20000), local)
        os.replace(os.path.join(out, "x_a.mpmct1"), os.path.join(out, "a.mpmct1"))
        os.replace(os.path.join(out, "x_b.mpmct1"), os.path.join(out, "b.mpmct1"))
        w_cap = rng.choice([8, 16, 32, 64, 512])
        g = rng.choice([4, 16, 32, 64, 512])
        b_mono = rng.choice([8, 32, 48, 256, 1024, 3072, 24576])
        span = rng.choice([40, 300, 3000])
        print(w_cap, g, b_mono, span)
    elif kind == "repair":
        n = rng.choice([4, 6, 8, 12])
        repair_cases.N = n
        repair_cases.NW = 4 * n
        repair_cases.make(out, "x", rng.randrange(1 << 30), rng.randrange(6, 30),
                          rng.randrange(2, 12), rng.choice([0.05, 0.25, 0.6]),
                          rng.randrange(0, 30), rng.randrange(4, 12))
        os.replace(os.path.join(out, "x.c.g57"), os.path.join(out, "c.g57"))
        os.replace(os.path.join(out, "x.packed.mpmct1"), os.path.join(out, "p.mpmct1"))
        os.replace(os.path.join(out, "x.packed.bounds"), os.path.join(out, "p.bounds"))
        # A small state with a large budget makes the oracle's ANF products
        # (no cap abort below 2^(4n) monomials) take hours in Python: keep
        # the budget below the state's monomial ceiling.
        budgets = [16, 64, 104, 256, 1024] + ([65536] if n >= 8 else [])
        print(n, rng.choice(budgets), rng.randrange(1, 5), rng.randrange(1, 5))
    else:
        sys.exit(f"unknown kind {kind}")


if __name__ == "__main__":
    main()
