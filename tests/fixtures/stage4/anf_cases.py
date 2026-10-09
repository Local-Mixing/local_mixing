"""Crafted (circuit, bounds) cases for the stage-4 ANF / serialization fixtures
(docs/STAGE4_RUST_DESIGN.md §10.3). Deterministic: random.Random(SEED) only.

usage: python3 -I anf_cases.py OUTDIR
Writes OUTDIR/<name>.mpmct1, OUTDIR/<name>.bounds and OUTDIR/cases.txt
(`name budget...`: the anf_compose.py budgets to record for that case).

Coverage targets: monomial widths W = 1, 2, 4, 8, 16 limbs (state widths up
to 1024, wire ids >= 61, >= 64, >= 128, >= 256, 1023); k = 0 gates,
complemented gates, negative polarities; bounds with 0, m, > m, negative,
duplicate and unsorted entries; empty windows (all changes cancel, f = c = 0)
and repeated bounds (zero-gate windows); a single window; a zero-gate circuit;
small budgets that overflow in all three ways (product, running sum, final)
and products in the ambiguous zone.
"""
import random
import sys

SEED = 20261008


def gate_line(t, comp, ctrls):
    s = f"{t} {comp} {len(ctrls)}"
    for c, p in ctrls:
        s += f" {c} {p}"
    return s


def rand_gate(r, wires, kmax=4, pcomp=0.3):
    k = r.choices(range(kmax + 1), weights=[4, 30, 40, 20, 6][: kmax + 1])[0]
    k = min(k, len(wires) - 1)
    pick = r.sample(wires, k + 1)
    t = pick[0]
    ctrls = sorted((c, r.randrange(2)) for c in pick[1:])
    return (t, 1 if r.random() < pcomp else 0, ctrls)


def local_circuit(r, nw, m, region=(3, 6), span=(30, 120), kmax=3, pcomp=0.3, forced=()):
    """Gates in runs over random wire subsets (so windows stay small), with
    `forced` wire ids always mixed into some runs."""
    gates = []
    while len(gates) < m:
        size = r.randint(*region)
        wires = r.sample(range(nw), min(size, nw))
        if forced and r.random() < 0.5:
            for f in forced:
                if f not in wires and r.random() < 0.5:
                    wires[r.randrange(len(wires))] = f
            wires = list(dict.fromkeys(wires))
        for _ in range(r.randint(*span)):
            gates.append(rand_gate(r, wires, kmax, pcomp))
    return gates[:m]


def bounds_for(r, m, every=(25, 90), extras=True):
    b = []
    x = 0
    while True:
        x += r.randint(*every)
        if x >= m:
            break
        b.append(x)
    if extras and b:
        b += [0, m, m + 7, -3, r.choice(b), r.choice(b)]
        r.shuffle(b)
    return b


def write(out, name, n_hdr, gates, bounds):
    with open(f"{out}/{name}.mpmct1", "w") as f:
        f.write(f"mpmct1 {n_hdr} {len(gates)}\n")
        for g in gates:
            f.write(gate_line(*g) + "\n")
    with open(f"{out}/{name}.bounds", "w") as f:
        f.write("".join(f"{x}\n" for x in bounds))


def cancel_case(r):
    """Windows whose changes cancel (g g), NOT pairs, zero-gate windows."""
    nw = 16
    gates, bounds = [], []
    for blk in range(40):
        wires = r.sample(range(nw), 5)
        kind = blk % 4
        if kind == 0:      # g X g with X commuting: all changes cancel
            g = rand_gate(r, wires)
            gates += [g, g]
        elif kind == 1:    # NOT pair (k = 0)
            t = r.choice(wires)
            gates += [(t, 0, []), (t, 0, [])]
        elif kind == 2:    # a k = 0 complemented gate (no-op) and a real gate
            t = r.choice(wires)
            gates += [(t, 1, []), rand_gate(r, wires)]
        else:              # a short random run
            gates += [rand_gate(r, wires) for _ in range(r.randint(1, 6))]
        bounds.append(len(gates))
        if blk % 5 == 0:
            bounds.append(len(gates))   # duplicate: an empty window
    return nw, gates, bounds


def orderdep_case(r):
    """Windows where t ^= c1*c2 with c1 = x_c1 + s, c2 = x_c2 + s (s a sum of
    k inputs): the product's CPython prefix grows to about (k+2)^2/4
    monomials while the finished product has 3k+1, so for 5k+4 <= 4*budget
    and 8*budget between the two, the product aborts under CPython 3.9 order
    but not under the canonical rule. The gate is applied twice and the
    copies are undone, so the window's final ANF stays small."""
    nw = 160
    gates, bounds = [], []
    for k in (24, 40, 48, 64, 80):
        xs = r.sample(range(0, 150), k)
        c1, c2, t = r.sample([w for w in range(nw) if w not in xs], 3)
        copy = [(c1, 0, [(x, 1)]) for x in xs] + [(c2, 0, [(x, 1)]) for x in xs]
        r.shuffle(copy)
        prod = (t, r.randrange(2), sorted([(c1, 1), (c2, 1)]))
        tail = [(t, 0, [(xs[0], 1)])]   # keeps one changed wire
        gates += copy + [prod, prod] + tail + copy[::-1]
        bounds.append(len(gates))
    return nw, gates, bounds


def main():
    out = sys.argv[1]
    r = random.Random(SEED)
    cases = []

    g = local_circuit(r, 32, 2500, region=(3, 6), span=(40, 120))
    write(out, "w32_dense", 32, g, bounds_for(r, len(g), (40, 110)))
    cases.append(("w32_dense", [6, 12, 40, 200]))

    g = local_circuit(r, 64, 1500, forced=(60, 61, 62, 63))
    g = [(t % 24 + 40, c, [(w % 24 + 40, p) for w, p in ctl]) for t, c, ctl in g]
    g = [(t, c, sorted(dict(ctl).items())) for t, c, ctl in g if t not in dict(ctl)]
    write(out, "w64_top", 64, g, bounds_for(r, len(g)))
    cases.append(("w64_top", [16, 100]))

    g = local_circuit(r, 128, 2000, forced=(61, 63, 64, 127))
    write(out, "w128", 128, g, bounds_for(r, len(g)))
    cases.append(("w128", [30, 300]))

    g = local_circuit(r, 256, 2000, forced=(61, 127, 128, 255))
    write(out, "w256", 256, g, bounds_for(r, len(g)))
    cases.append(("w256", [50]))

    g = local_circuit(r, 300, 1500, forced=(61, 191, 256, 299))
    write(out, "w300", 300, g, bounds_for(r, len(g)))
    cases.append(("w300", [40]))

    g = local_circuit(r, 512, 1500, forced=(63, 256, 448, 511))
    write(out, "w512", 512, g, bounds_for(r, len(g)))
    cases.append(("w512", [40]))

    g = local_circuit(r, 1024, 1500, forced=(61, 512, 1000, 1023))
    write(out, "w1024", 1024, g, bounds_for(r, len(g)))
    cases.append(("w1024", [40]))

    nw, g, b = cancel_case(r)
    write(out, "cancel", nw, g, b)
    cases.append(("cancel", [4, 64]))

    g = local_circuit(r, 20, 300, region=(5, 5), span=(150, 150))
    write(out, "single", 20, g, [])
    cases.append(("single", [64, 4096]))

    nw, g, b = orderdep_case(r)
    write(out, "orderdep", nw, g, b)
    cases.append(("orderdep", [30, 56, 70, 90, 104, 200]))

    write(out, "nogates", 8, [], [])
    cases.append(("nogates", [8]))

    with open(f"{out}/cases.txt", "w") as f:
        for name, budgets in cases:
            f.write(name + " " + " ".join(map(str, budgets)) + "\n")


if __name__ == "__main__":
    main()
