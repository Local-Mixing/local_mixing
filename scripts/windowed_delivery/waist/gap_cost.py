"""Cost census for targeted boundary refresh: for each celebrity span, split by
reads of the carrier into read-free gaps; count gaps containing >=1 boundary.
Each such gap needs one nonlinear toggle/untoggle pair (2 gates)."""
import sys, random, hashlib, bisect
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

a_path, p_path, b_path = sys.argv[1:4]
MULTMIN = 5
na, A = load(a_path)
npk, P = load(p_path)
bounds = sorted(int(x) for x in open(b_path).read().split())
n = max(na, npk)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def norm(v): return v ^ MASK if v & 1 else v

mult = defaultdict(int)
st = list(inputs[:na]) + [0]*(n-na)
for t, comp, cps, pol in A:
    acc = MASK
    for c,p in zip(cps,pol): acc &= st[c] if p else st[c]^MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if fp: mult[hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()] += 1

reads = defaultdict(list)
for i,(t,comp,cps,pol) in enumerate(P):
    for c,_ in zip(cps,pol): reads[c].append(i)

pending = {}
pairs = 0        # (span,gap) toggle pairs needed
spans_cut = 0
crossings = 0
vals_cut = set()
st = list(inputs[:npk]) + [0]*(n-npk)
def process(b, e, t, d):
    global pairs, spans_cut, crossings
    # boundaries strictly inside: cut c slices iff b < c <= e  (cut between gate c-1 and c)
    lo = bisect.bisect_right(bounds, b)
    hi = bisect.bisect_right(bounds, e)
    cuts = bounds[lo:hi]
    if not cuts: return
    spans_cut += 1; crossings += len(cuts); vals_cut.add(d)
    r = reads.get(t, ())
    rl = r[bisect.bisect_right(r, b):bisect.bisect_left(r, e)]
    # gaps: (b, rl0], (rl0, rl1], ..., (rl[-1], e]; cut c in gap (g0,g1] iff g0 < c-? ...
    # cut at position c sits between gates c-1 and c; it's inside read-free gap
    # (after read at g0, before read at g1) iff g0 < c <= g1  (read at index g reads BEFORE cut c iff g < c)
    gs = [b] + rl + [e]
    used = set()
    for c in cuts:
        k = bisect.bisect_left(gs, c)   # first g >= c -> gap index k
        used.add(k)
    pairs += len(used)
for i,(t,comp,cps,pol) in enumerate(P):
    acc = MASK
    for c,p in zip(cps,pol): acc &= st[c] if p else st[c]^MASK
    if comp: acc ^= MASK
    prev = pending.pop(t, None)
    if prev is not None:
        b,d = prev
        if mult[d] >= MULTMIN: process(b,i,t,d)
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        d = hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()
        if d in mult: pending[t] = (i,d)
tot = len(P)
for t,(b,d) in pending.items():
    if mult[d] >= MULTMIN: process(b,tot,t,d)  # output spans: last "cut" = end
print(f"gates {tot}, windows {len(bounds)+1}")
print(f"celebrity spans cut: {spans_cut}, total crossings {crossings}, distinct values cut {len(vals_cut)}")
print(f"toggle pairs needed: {pairs}  -> {2*pairs} inserted gates = {200*pairs/tot:.2f}% overhead")
