"""Band-wire complexity in the C-view domain (wires 0..63 = C inputs, rest 0)
at given stage-2 cuts: input-support size and degree class per band wire."""
import sys, random, re
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load
tag = sys.argv[1]
R = f'/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist/{tag}'
na, A = load(f'{R}/tdp.mpmct1')
g = open(f'{R}/gen.log').read()
seed_n, reseed_n, guard_n = map(int, re.search(r'(\d+) band-seed \+ (\d+) band-reseed \+ (\d+) slice-guard', g).groups())
NI, L = 64, 64
rng = random.Random(9)
x0 = [rng.getrandbits(L) for _ in range(NI)]
def run(x, upto):
    m = (1 << L) - 1; st = list(x) + [0] * (na - NI)
    for t, comp, cps, pol in A[:upto]:
        acc = m
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ m
        if comp: acc ^= m
        st[t] ^= acc
    return st
cuts = [('after guard', guard_n), ('after band seed', guard_n + seed_n),
        ('+2k compute', guard_n + seed_n + 2000), ('+10k compute', guard_n + seed_n + 10000),
        ('+50k compute', guard_n + seed_n + 50000)]
for name, cut in cuts:
    s0 = run(x0, cut)
    supp = {w: 0 for w in range(128, na)}
    for i in range(NI):
        x1 = list(x0); x1[i] ^= (1 << L) - 1
        s1 = run(x1, cut)
        for w in range(128, na):
            if s1[w] != s0[w]: supp[w] += 1
    const = sum(1 for w in supp if supp[w] == 0)
    small = sum(1 for w in supp if 1 <= supp[w] <= 3)
    sz = sorted(supp.values())
    print(f'{tag} {name:16s} (stage-2 gate {cut:6d}): band wires constant {const:3d}/128, '
          f'support 1-3 bits {small:3d}, median support {sz[64]}')
