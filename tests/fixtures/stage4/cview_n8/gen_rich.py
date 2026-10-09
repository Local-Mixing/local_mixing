"""Generate the synthetic 'rich' C-view fixture (rich_c.mpmct1, rich_p.mpmct1).

C: 24 gates on n=8 wires (CNOTs, NOTs, Toffolis, complemented gates).
P: C's gates verbatim, interleaved with linear traffic between the data
wires 0..7 and the aux wires 8..47 (copies, aux-into-data masking, XORs of
aux wires, NOTs = constants on clean wires) and nonlinear junk on aux. P
does not compute C; it only has to share values with it. This
exercises complement classes, constants, nullspace walks (nullity <= 14 and
> 14), t* > 1 and ties, which the real n=8 G2 fixture does not.
Deterministic: python3 gen_rich.py (any Python 3)."""
import random

r = random.Random(20261008)
N, AUX = 8, 40
NW = N + AUX


def gate(t, lits, comp=0):
    lits = sorted(set(lits))
    ws = [w for w, _ in lits]
    assert t not in ws and len(ws) == len(set(ws))
    return (t, comp, lits)


def c_gate():
    while True:
        t = r.randrange(N)
        k = r.choice([0, 1, 1, 1, 2, 2])
        ws = r.sample([w for w in range(N) if w != t], k)
        return gate(t, [(w, r.randrange(2)) for w in ws], int(r.random() < 0.2))


C = [c_gate() for _ in range(24)]
P = []
for i, g in enumerate(C):
    P.append(g)
    linear = i < 12  # first half: linear-heavy traffic (big nullity)
    for _ in range(r.randrange(2, 12 if linear else 6)):
        a = r.randrange(N, NW)
        kind = r.randrange(6 if linear else 8)
        if kind <= 1:
            P.append(gate(a, [(r.randrange(N), 1)]))  # aux ^= data
        elif kind == 2:
            P.append(gate(a, []))  # NOT aux
        elif kind == 3:
            b = r.randrange(N, NW)
            if b != a:
                P.append(gate(a, [(b, 1)]))  # aux ^= aux
        elif kind <= 5:
            P.append(gate(r.randrange(N), [(a, 1)]))  # data ^= aux (masks C)
        else:
            ws = r.sample([w for w in range(NW) if w != a], 2)
            P.append(gate(a, [(w, r.randrange(2)) for w in ws], int(r.random() < 0.3)))


def write(path, gates, nw):
    with open(path, "w") as f:
        f.write("mpmct1 %d %d\n" % (nw, len(gates)))
        for t, comp, lits in gates:
            f.write(" ".join(str(x) for x in [t, comp, len(lits)] + [v for wp in lits for v in wp]) + "\n")


write("rich_c.mpmct1", C, N)
write("rich_p.mpmct1", P, NW)
print(len(C), len(P))
