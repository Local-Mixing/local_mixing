"""For delivered wires at a cut, find stage-2 gates producing the identical
function (up to complement) in the C-view domain (wires 0..63 random, rest 0).
usage: trace_domain.py TAG CUT WIRE..."""
import sys, random, re, hashlib
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load
tag, cut = sys.argv[1], int(sys.argv[2]); wires = [int(w) for w in sys.argv[3:]]
R = f'/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist/{tag}'
na, A = load(f'{R}/tdp.mpmct1'); npk, P = load(f'{R}/packed.mpmct1')
g = open(f'{R}/gen.log').read()
seed_n, reseed_n, guard_n = map(int, re.search(r'(\d+) band-seed \+ (\d+) band-reseed \+ (\d+) slice-guard', g).groups())
def region(i):
    if i < guard_n: return 'guard-in'
    if i < guard_n + seed_n: return 'band-seed'
    return f'compute+{i - guard_n - seed_n}'
NI, L = 64, 2048
m = (1 << L) - 1
rng = random.Random(11)
x = [rng.getrandbits(L) for _ in range(NI)]
def canon(v): return v ^ m if v & 1 else v
st = list(x) + [0] * (npk - NI)
for t, comp, cps, pol in P[:cut]:
    acc = m
    for c, p in zip(cps, pol): acc &= st[c] if p else st[c] ^ m
    if comp: acc ^= m
    st[t] ^= acc
want = {canon(st[w]): w for w in wires}
found = {w: [] for w in wires}
inputs = {canon(v): i for i, v in enumerate(x)}
for w in wires:
    if canon(st[w]) in inputs: found[w].append(f'= raw input x{inputs[canon(st[w])]}')
s2 = list(x) + [0] * (na - NI)
for i, (t, comp, cps, pol) in enumerate(A):
    acc = m
    for c, p in zip(cps, pol): acc &= s2[c] if p else s2[c] ^ m
    if comp: acc ^= m
    s2[t] ^= acc
    v = canon(s2[t])
    if v in want and len(found[want[v]]) < 4:
        found[want[v]].append(f'stage-2 gate {i} [{region(i)}] wire {t} ctrls {list(zip(cps, pol))}')
for w in wires:
    print(f'  delivered wire {w}: ' + ('; '.join(found[w]) if found[w] else 'NEVER produced in stage 2 (mixer/packer-born)'))
