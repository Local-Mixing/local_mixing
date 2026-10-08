"""Celebrity-avoiding boundary redraw (post-post-processing, RC 2026-10-07).
Gates untouched; only the window cut set is re-chosen so that no boundary
falls inside any span of a protectable celebrity value (mult >= MULTMIN,
all spans <= WHARD)."""
import sys, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

a_path, p_path, bounds_out = sys.argv[1:4]
WHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 256
MULTMIN = int(sys.argv[5]) if len(sys.argv) > 5 else 5
na, A = load(a_path)
npk, P = load(p_path)
total = int(open(p_path).readline().split()[2])
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

# spans in the emitted order
pending = {}
spans_by_val = defaultdict(list)   # digest -> [(b,e)] for mult>=MULTMIN
soft = []                          # all other spans (for secondary cost)
st = list(inputs[:npk]) + [0]*(n-npk)
def close(t, i):
    prev = pending.pop(t, None)
    if prev is not None:
        b, d = prev
        if mult[d] >= MULTMIN:
            spans_by_val[d].append((b, i))
        else:
            soft.append((b, i))
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
for t in list(pending): close(t, total)   # output-pending: unprotectable anyway

protectable, unprotectable = [], []
for d, ss in spans_by_val.items():
    if all(e - b <= WHARD - 2 for b, e in ss):
        protectable.append(d)
    else:
        unprotectable.append(d)
print(f"celebrities (mult>={MULTMIN}): {len(spans_by_val)}; protectable {len(protectable)}, "
      f"unprotectable (some span > {WHARD-2}): {len(unprotectable)}")

# forbidden zones + owner tracking (span slices cut c iff b+1 <= c <= e)
forb = bytearray(total + 2)
owner_diff = defaultdict(list)     # position -> digests starting/ending (for jams)
for d in protectable:
    for b, e in spans_by_val[d]:
        lo, hi = b + 1, min(e, total)
        if lo <= hi:
            forb[lo] += 1 if forb[lo] < 255 else 0
            # store interval per digest lazily: keep list for jam resolution
            owner_diff[(lo, hi)].append(d)
# coverage count via diff array (how many protected spans cover c)
cov = [0]*(total + 2)
for (lo, hi), ds in owner_diff.items():
    cov[lo] += len(ds)
    cov[hi+1] -= len(ds)
for i in range(1, total + 2):
    cov[i] += cov[i-1]
# soft coverage
scov = [0]*(total + 2)
for b, e in soft:
    lo, hi = b + 1, min(e, total)
    if lo <= hi:
        scov[lo] += 1
        scov[hi+1] -= 1
for i in range(1, total + 2):
    scov[i] += scov[i-1]

cuts = []
prev = 0
jams = 0
sacrificed = set()
while prev < total:
    hi = min(prev + WHARD, total)
    if hi == total:
        break
    # free positions in (prev, hi]
    best = None
    for p in range(hi, prev, -1):
        if cov[p] == 0:
            if best is None or scov[p] < scov[best]:
                best = p
                if scov[p] == 0:
                    break
    if best is None:
        # jam: pick position with min protected coverage, sacrifice owners
        best = min(range(prev + 1, hi + 1), key=lambda p: cov[p])
        jams += 1
    cuts.append(best)
    prev = best
open(bounds_out, 'w').write("\n".join(map(str, cuts)) + "\n")
print(f"placed {len(cuts)} cuts, jams={jams}; window sizes: "
      f"min={min(b-a for a,b in zip([0]+cuts, cuts+[total]))} "
      f"max={max(b-a for a,b in zip([0]+cuts, cuts+[total]))}")
