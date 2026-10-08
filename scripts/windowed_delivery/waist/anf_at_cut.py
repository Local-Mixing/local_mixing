"""Exact ANF (over the C input bits) of a C value and of delivered wires at a
cut, in the C-view evaluation domain (wires 0..63 = C inputs, all other
wires start at 0).

usage: anf_at_cut.py TAG C_GATE CUT WIRE...
"""
import sys, random
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load

ALPH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
D = {c: i for i, c in enumerate(ALPH)}
tag, cg, cut = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
wires = [int(w) for w in sys.argv[4:]]
R = f'/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist/{tag}'
body = open(f'{R}/tdp.mpmct1.source_c.g57').read().strip().strip(';')
C = [tuple(D[ch] for ch in tok) for tok in body.split(';') if tok]
npk, P = load(f'{R}/packed.mpmct1')
P = P[:cut]
NI = 64

def run_g(lanes, x):           # x: list of NI lane-ints
    m = (1 << lanes) - 1
    st = list(x) + [0] * (npk - NI)
    for t, comp, cps, pol in P:
        acc = m
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ m
        if comp:
            acc ^= m
        st[t] ^= acc
    return st
def run_c(lanes, x):
    m = (1 << lanes) - 1
    st = list(x)
    for g, (a, xx, y) in enumerate(C[:cg + 1]):
        st[a] ^= st[xx] | (st[y] ^ m)
    return st[C[cg][0]]

# input support by flips on 64 random points (bit-sliced, 64 lanes)
rng = random.Random(5)
L = 64
x0 = [rng.getrandbits(L) for _ in range(NI)]
g0, c0 = run_g(L, x0), run_c(L, x0)
gs = {w: set() for w in wires}; cs = set()
for i in range(NI):
    x1 = list(x0); x1[i] ^= (1 << L) - 1
    g1, c1 = run_g(L, x1), run_c(L, x1)
    for w in wires:
        if g1[w] != g0[w]: gs[w].add(i)
    if c1 != c0: cs.add(i)

def anf(supp, fn):
    supp = sorted(supp)
    k = len(supp)
    if k > 16:
        return f'support {k} bits (too large to expand)'
    lanes = 1 << k
    x = [0] * NI
    for j, b in enumerate(supp):
        v = 0
        for a in range(lanes):
            if (a >> j) & 1:
                v |= 1 << a
        x[b] = v
    tt = fn(lanes, x)
    coef = [(tt >> a) & 1 for a in range(lanes)]
    for j in range(k):                     # Moebius transform
        for a in range(lanes):
            if (a >> j) & 1:
                coef[a] ^= coef[a ^ (1 << j)]
    monos = []
    for a in range(lanes):
        if coef[a]:
            ms = [f'x{supp[j]}' for j in range(k) if (a >> j) & 1]
            monos.append('*'.join(ms) if ms else '1')
    monos.sort(key=lambda s: (s.count('*'), s))
    deg = max((m.count('*') + 1 if m != '1' else 0) for m in monos) if monos else 0
    return f'deg {deg}: ' + ' + '.join(monos)

print(f'{tag} C gate {cg} (gate {C[cg]}): {anf(cs, run_c)}')
for w in wires:
    print(f'  delivered wire {w:3d} at cut {cut}: {anf(gs[w], lambda l, x, w=w: run_g(l, x)[w])}')
