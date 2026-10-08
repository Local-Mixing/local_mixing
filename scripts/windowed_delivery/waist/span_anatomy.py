"""Anatomy of celebrity spans in the emitted order: lengths, and read-activity
of the carrier wire inside each span (collapsible iff zero intervening reads)."""
import sys, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

a_path, p_path = sys.argv[1:3]
MULTMIN = 5
na, A = load(a_path)
npk, P = load(p_path)
total = len(P)
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
    if fp: mult[hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()] += 1

# reads index: for each wire, sorted gate indices where it is a control
reads = defaultdict(list)
for i,(t,comp,cps,pol) in enumerate(P):
    for c,_ in zip(cps,pol): reads[c].append(i)
import bisect
def nreads(w, b, e):
    r = reads.get(w, ())
    return bisect.bisect_left(r, e) - bisect.bisect_right(r, b)

pending = {}
spans = defaultdict(list)   # digest -> [(b,e,wire,reads_in)]
st = list(inputs[:npk]) + [0]*(n-npk)
for i,(t,comp,cps,pol) in enumerate(P):
    acc = MASK
    for c,p in zip(cps,pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    prev = pending.pop(t, None)
    if prev is not None:
        b,d = prev
        if mult[d] >= MULTMIN:
            spans[d].append((b,i,t,nreads(t,b,i)))
    st[t] ^= acc
    fp = norm(st[t])
    if fp:
        d = hashlib.blake2b(fp.to_bytes(NBYTES,"little"),digest_size=16).digest()
        if d in mult: pending[t] = (i,d)
# output-pending spans dropped (unprotectable regardless)

allspans = [s for ss in spans.values() for s in ss]
lens = sorted(e-b for b,e,_,_ in allspans)
import statistics
print(f"celebrity values {len(spans)}, interior spans {len(allspans)}")
print(f"span lengths: min {lens[0]}, p25 {lens[len(lens)//4]}, median {lens[len(lens)//2]}, "
      f"p75 {lens[3*len(lens)//4]}, p95 {lens[int(.95*len(lens))]}, max {lens[-1]}")
for W in (254, 510, 1022, 2046, 4094):
    nlong = sum(1 for L in lens if L > W)
    vbad = sum(1 for ss in spans.values() if any(e-b > W for b,e,_,_ in ss))
    print(f"  W={W+2}: spans too long {nlong} ({100*nlong/len(lens):.1f}%), values unprotectable {vbad}/{len(spans)}")
zr = sum(1 for _,_,_,r in allspans if r == 0)
print(f"spans with ZERO intervening reads of carrier (collapsible): {zr}/{len(allspans)} ({100*zr/len(allspans):.1f}%)")
longz = [s for s in allspans if s[1]-s[0] > 254]
lz = sum(1 for s in longz if s[3] == 0)
print(f"  among >254 spans: collapsible {lz}/{len(longz)} ({100*lz/max(1,len(longz)):.1f}%)")
rd = sorted(s[3] for s in longz)
if rd: print(f"  reads-in-long-span: median {rd[len(rd)//2]}, p95 {rd[int(.95*len(rd))]}, max {rd[-1]}")
# per-value: would pull-forward + redraw protect it? value OK iff every span is
# (short) or (collapsible)
vok = sum(1 for ss in spans.values() if all(e-b <= 254 or r == 0 for b,e,_,_ in ss))
print(f"values protectable by pull-forward(collapse)+redraw @W=256: {vok}/{len(spans)}")
vout = sum(1 for d,(i,dd) in [] or [])
