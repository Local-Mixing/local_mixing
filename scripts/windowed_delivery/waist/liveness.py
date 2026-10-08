"""Per-position celebrity liveness + best-case redraw cost: greedily cut at the
thinnest position within each 256-stretch, summing distinct (span,gap) hits."""
import sys, random, hashlib, bisect
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES
a_path, p_path = sys.argv[1:3]
MULTMIN = 5; W = 254
na, A = load(a_path); npk, P = load(p_path)
n = max(na, npk)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def norm(v): return v ^ MASK if v & 1 else v
mult = defaultdict(int)
st = list(inputs[:na]) + [0]*(n-na)
for t,comp,cps,pol in A:
    acc = MASK
    for c,p in zip(cps,pol): acc &= st[c] if p else st[c]^MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if fp: mult[hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()] += 1
pending = {}; spans = []
st = list(inputs[:npk]) + [0]*(n-npk)
for i,(t,comp,cps,pol) in enumerate(P):
    acc = MASK
    for c,p in zip(cps,pol): acc &= st[c] if p else st[c]^MASK
    if comp: acc ^= MASK
    prev = pending.pop(t, None)
    if prev is not None:
        b,d = prev
        if mult[d] >= MULTMIN: spans.append((b,i))
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        d = hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()
        if d in mult: pending[t] = (i,d)
tot = len(P)
live = [0]*(tot+2)
for b,e in spans:
    live[b+1] += 1; live[min(e,tot)+1] -= 1
for i in range(1, tot+1): live[i] += live[i-1]
L = sorted(live[1:tot+1])
print(f"celebrity liveness per cut position: min {L[0]}, p5 {L[len(L)//20]}, median {L[len(L)//2]}, p95 {L[int(.95*len(L))]}, max {L[-1]}")
# greedy thin-cut redraw: in each stretch, cut at min-liveness position
prev = 0; cost = 0; cuts = 0
while prev + W < tot:
    seg = range(prev+1, prev+W+1)
    p = min(seg, key=lambda c: live[c])
    cost += live[p]; cuts += 1; prev = p
print(f"thin-greedy redraw: {cuts} cuts, span-crossings >= {cost} -> pairs <= that; "
      f"gates ~{2*cost} = {200*cost/tot:.1f}% overhead (upper bd via crossings; distinct gaps <= crossings)")
