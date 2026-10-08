import sys, random, hashlib, bisect
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

a_path, p_path, b_path, label = sys.argv[1:5]
na, A = load(a_path)
npk, P = load(p_path)
total = int(open(p_path).readline().split()[2])
bounds = sorted(set(int(x) for x in open(b_path) if x.strip()) | {total})
n = max(na, npk)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def norm(v): return v ^ MASK if v & 1 else v
mult = defaultdict(int)
st = list(inputs[:na]) + [0]*(n-na)
for t, comp, cps, pol in A:
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        mult[hashlib.blake2b(fp.to_bytes(NBYTES,"little"), digest_size=16).digest()] += 1
born = sum(mult.values())
pending = {}; exposed = set()
st = list(inputs[:npk]) + [0]*(n-npk)
def close(t, i):
    prev = pending.pop(t, None)
    if prev is not None:
        b, d = prev
        k = bisect.bisect_left(bounds, b + 1)
        if k < len(bounds) and bounds[k] <= i:
            exposed.add(d)
for i, (t, comp, cps, pol) in enumerate(P):
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    close(t, i)
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        d = hashlib.blake2b(fp.to_bytes(NBYTES,"little"), digest_size=16).digest()
        if d in mult:
            pending[t] = (i, d)
for t in list(pending): close(t, total)
seg = sum(mult[d] for d in exposed)
print(f"{label}: born={born} distinct={len(mult)} repeats={100*(born-len(mult))/born:.1f}%")
print(f"  exposed distinct: {len(exposed)} ({100*len(exposed)/len(mult):.2f}%) covering {seg} segments ({100*seg/born:.1f}%)")
for lo, hi, lab in [(1,1,'1'),(2,4,'2-4'),(5,64,'5-64'),(65,256,'65-256'),(257,10**9,'>=257')]:
    vals = [d for d in exposed if lo <= mult[d] <= hi]
    print(f"    mult {lab:>7}: {len(vals):>7} values / {sum(mult[d] for d in vals):>7} segments")
nonexp = [mult[d] for d in mult if d not in exposed]
print(f"  non-exposed max mult: {max(nonexp) if nonexp else 0}")
top = sorted(exposed, key=lambda d: -mult[d])[:5]
print("  top exposed:", [mult[d] for d in top])
