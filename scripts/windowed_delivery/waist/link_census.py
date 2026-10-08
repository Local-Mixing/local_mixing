"""Linkable-exposure census of a windowed deliverable (RC 2026-10-07).

A value is LINKABLE iff it is visible in >= 2 separate boundary runs (a run =
consecutive boundaries where one wire keeps the same value; inputs seed the
first runs). Linkable values are recognizable from the artifact alone and
are thread ends for an attack; values seen once are near-innocuous.

usage: link_census.py A.mpmct1 packed.mpmct1 bounds label
Prints, for proxy (= stage-2 born values) and froth (mixer-born):
  L2   linkable values (cluster size >= 2)
  M2   linkable mass   (sum of runs over clusters of size >= 2)
  histogram of cluster sizes, and for linkable proxy values their stage-2
  birth multiplicity classes.
"""
import sys, random, hashlib
from collections import defaultdict, Counter
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from boundary_cluster import load, NBITS, MASK, NBYTES, SEED

a_path, b_path, bounds_path, label = sys.argv[1:5]
na, A = load(a_path)
nb, B = load(b_path)
bounds = sorted(int(x) for x in open(bounds_path).read().split())
n = max(na, nb)
rng = random.Random(SEED)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]

def digest(v):
    fp = v ^ MASK if v & 1 else v
    if fp == 0:
        return None
    return hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()

amult = defaultdict(int)
st = list(inputs[:na]) + [0] * (n - na)
for t, comp, cps, pol in A:
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp:
        acc ^= MASK
    st[t] ^= acc
    d = digest(st[t])
    if d:
        amult[d] += 1

st = list(inputs[:nb]) + [0] * (n - nb)
last = [digest(v) for v in st]
counts = defaultdict(int)
for d in last:
    if d:
        counts[d] += 1
lo = 0
dirty = set()
for cut in bounds:
    for t, comp, cps, pol in B[lo:cut]:
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        st[t] ^= acc
        dirty.add(t)
    for w in dirty:
        d = digest(st[w])
        if d != last[w]:
            last[w] = d
            if d:
                counts[d] += 1
    dirty.clear()
    lo = cut

prox = {d: c for d, c in counts.items() if d in amult}
froth = {d: c for d, c in counts.items() if d not in amult}
def bucket(c):
    for lo_, hi_, name in ((1, 1, '1'), (2, 2, '2'), (3, 4, '3-4'), (5, 9, '5-9'),
                           (10, 19, '10-19'), (20, 49, '20-49'), (50, 99, '50-99'), (100, 10**9, '100+')):
        if lo_ <= c <= hi_:
            return name
print(f'{label}: {len(bounds)} boundaries, {sum(counts.values())} runs')
for name, dct in (('proxy', prox), ('froth', froth)):
    L2 = sum(1 for c in dct.values() if c >= 2)
    M2 = sum(c for c in dct.values() if c >= 2)
    h = Counter(bucket(c) for c in dct.values())
    order = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
    print(f'  {name}: values {len(dct)}  L2 {L2}  M2 {M2}  | sizes ' +
          ' '.join(f'{k}:{h.get(k, 0)}' for k in order))
lp = [d for d, c in prox.items() if c >= 2]
mh = Counter(bucket(amult[d]) for d in lp)
print('  linkable proxy by stage-2 birth multiplicity: ' +
      ' '.join(f'{k}:{mh.get(k, 0)}' for k in ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']))
mass = Counter()
for d in lp:
    mass[bucket(amult[d])] += prox[d]
print('  linkable proxy MASS by stage-2 birth multiplicity: ' +
      ' '.join(f'{k}:{mass.get(k, 0)}' for k in ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']))
