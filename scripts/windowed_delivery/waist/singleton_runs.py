"""For linkable proxy values born ONCE in stage 2: are their runs on the
same wire (value perturbed then restored) or on different wires (copied)?"""
import sys, random, hashlib
from collections import defaultdict, Counter
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from boundary_cluster import load, NBITS, MASK, NBYTES, SEED
a_path, b_path, bounds_path = sys.argv[1:4]
na, A = load(a_path); nb, B = load(b_path)
bounds = sorted(int(x) for x in open(bounds_path).read().split())
n = max(na, nb)
rng = random.Random(SEED)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def digest(v):
    fp = v ^ MASK if v & 1 else v
    return None if fp == 0 else hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
amult = defaultdict(int)
st = list(inputs[:na]) + [0] * (n - na)
for t, comp, cps, pol in A:
    acc = MASK
    for c, p in zip(cps, pol): acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    d = digest(st[t])
    if d: amult[d] += 1
st = list(inputs[:nb]) + [0] * (n - nb)
cur = [digest(v) for v in st]; start = [-1] * n
runs = defaultdict(list); lo = 0; dirty = set()
for bi, cut in enumerate(bounds):
    for t, comp, cps, pol in B[lo:cut]:
        acc = MASK
        for c, p in zip(cps, pol): acc &= st[c] if p else st[c] ^ MASK
        if comp: acc ^= MASK
        st[t] ^= acc; dirty.add(t)
    for w in dirty:
        d = digest(st[w])
        if d != cur[w]:
            if cur[w]: runs[cur[w]].append((w, start[w], bi - 1))
            cur[w] = d; start[w] = bi
    dirty.clear(); lo = cut
for w in range(n):
    if cur[w]: runs[cur[w]].append((w, start[w], len(bounds) - 1))
kinds = Counter(); gaps = []
for d, rr in runs.items():
    if len(rr) >= 2 and amult.get(d) == 1:
        wires = {w for w, _, _ in rr}
        if len(wires) == 1:
            kinds['same wire'] += 1
            rr.sort(key=lambda x: x[1])
            gaps += [b[1] - a[2] - 1 for a, b in zip(rr, rr[1:])]
        else:
            kinds[f'{len(wires)} wires'] += 1
print('linkable stage-2 singletons by wire pattern:', dict(kinds))
if gaps:
    g = sorted(gaps)
    print(f'same-wire gaps (boundaries in between): median {g[len(g)//2]}, '
          f'share with gap 1: {sum(1 for x in g if x == 1)/len(g):.0%}, gap<=3: {sum(1 for x in g if x <= 3)/len(g):.0%}')
