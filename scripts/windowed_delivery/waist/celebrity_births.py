"""Birth-site anatomy of celebrity values in a stage-2 output stream.
Classify each birth gate of every mult>=THRESH value:
  pair2   = second CNOT of an adjacent reciprocal transfer pair (value hop)
  pair1   = first CNOT of such a pair
  cnot    = other single-control CNOT (drain / seed / port swap)
  multi   = multi-control gate (fire / bracket / guard)"""
import sys, random, hashlib
from collections import defaultdict, Counter
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES
path = sys.argv[1]; THRESH = int(sys.argv[2]) if len(sys.argv) > 2 else 257
n, G = load(path)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def norm(v): return v ^ MASK if v & 1 else v
births = defaultdict(list)   # digest -> [gate indices]
st = list(inputs)
for i, (t, comp, cps, pol) in enumerate(G):
    acc = MASK
    for c, p in zip(cps, pol): acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        births[hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()].append(i)
celebs = {d: bs for d, bs in births.items() if len(bs) >= THRESH}
print(f"{path}: {len(celebs)} values with mult >= {THRESH}")
def kind(i):
    t, comp, cps, pol = G[i]
    if len(cps) != 1 or comp: return "multi"
    c = cps[0]
    for j in (i-1, i+1):
        if 0 <= j < len(G):
            t2, comp2, cps2, pol2 = G[j]
            if len(cps2) == 1 and not comp2 and t2 == c and cps2[0] == t:
                return "pair2" if j == i-1 and G[j][0] == c else ("pair1" if j == i+1 else "pair2")
    return "cnot"
total = Counter(); wires = Counter()
percel = []
for d, bs in sorted(celebs.items(), key=lambda kv: -len(kv[1]))[:400]:
    ks = Counter(kind(i) for i in bs)
    total.update(ks)
    ws = len(set(G[i][0] for i in bs))
    percel.append((len(bs), dict(ks), ws))
print("birth-kind totals over celebrity class:", dict(total))
print("top 8 celebrities: (mult, kinds, distinct birth wires)")
for row in percel[:8]: print("  ", row)
# what reads these values? sample reads of top celeb's wires during spans: skip (heavy)
