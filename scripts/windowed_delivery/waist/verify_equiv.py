"""Functional equivalence of two mpmct1 circuits on 1024 random states over ALL wires."""
import sys, random
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist')
from order_repair import load
L = 1024; M = (1 << L) - 1
na, A = load(sys.argv[1]); nb, B = load(sys.argv[2])
n = max(na, nb)
rng = random.Random(77)
x = [rng.getrandbits(L) for _ in range(n)]
def run(G):
    st = list(x)
    for t, comp, cps, pol in G:
        acc = M
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ M
        if comp: acc ^= M
        st[t] ^= acc
    return st
sa, sb = run(A), run(B)
bad = sum(1 for u, v in zip(sa, sb) if u != v)
print(f'{sys.argv[2]}: {len(A)} vs {len(B)} gates; {"EQUIVALENT" if bad == 0 else f"MISMATCH on {bad} wires"} (1024 random states)')
