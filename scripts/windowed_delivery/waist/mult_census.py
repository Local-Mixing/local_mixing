import sys, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES
a_path, label = sys.argv[1:3]
na, A = load(a_path)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(na)]
def norm(v): return v ^ MASK if v & 1 else v
mult = defaultdict(int); born = 0
st = list(inputs)
for t, comp, cps, pol in A:
    acc = MASK
    for c, p in zip(cps, pol): acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        born += 1
        mult[hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()] += 1
vals = sorted(mult.values(), reverse=True)
big = [v for v in vals if v >= 65]
print(f" {label}: born={born:>8} distinct={len(mult):>8} repeats={100*(born-len(mult))/born:>5.1f}%  "
      f"top={vals[0]:>5}  vals>=65: {len(big):>4} (cover {sum(big):>7} = {100*sum(big)/born:.1f}%)  "
      f">=257: {sum(1 for v in vals if v>=257)}")
