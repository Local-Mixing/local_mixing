"""Input support (number of input bits a C value depends on) per C gate,
from the g57 source; g57 gate [a,x,y]: a ^= x OR NOT y."""
import sys, random
ALPH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
D = {c: i for i, c in enumerate(ALPH)}
def load(p):
    body = open(p).read().strip().strip(';')
    return [tuple(D[ch] for ch in tok) for tok in body.split(';') if tok]
def run(C, x):
    st = list(x); vals = []
    for a, xx, y in C:
        st[a] ^= (st[xx] | (1 - st[y]))
        vals.append(st[a])
    return vals
tag = sys.argv[1]; gates = [int(g) for g in sys.argv[2:]]
C = load(f'{tag}/tdp.mpmct1.source_c.g57'); n = 64
rng = random.Random(1)
supp = [set() for _ in C]
for _ in range(64):
    x = [rng.getrandbits(1) for _ in range(n)]
    base = run(C, x)
    for i in range(n):
        y = list(x); y[i] ^= 1
        for gi, (u, v) in enumerate(zip(base, run(C, y))):
            if u != v: supp[gi].add(i)
sizes = [len(s) for s in supp]
q = sorted(sizes)
print(f'{tag}: C gates {len(C)}; input-support size median {q[len(q)//2]}, p10 {q[len(q)//10]}, '
      f'first-50-gates median {sorted(sizes[:50])[25]}')
for g in gates:
    print(f'  leaked C gate {g:3d}: depends on {sizes[g]:2d} of 64 input bits')
