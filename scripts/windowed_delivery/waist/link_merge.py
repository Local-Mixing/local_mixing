"""Link-breaking boundary deletion (subset-safe post-pass).

Deleting boundaries never creates a run (it can only remove runs or merge
two runs of the same value on one wire), so no value of any kind becomes
newly visible or newly linkable. Greedy: repeatedly pick the run whose
deletion (all of its boundaries) breaks linkability most cheaply -- a
value with s runs needs s-1 of them removed -- subject to every merged
window staying <= WMAX gates.

usage: link_merge.py A.mpmct1 packed.mpmct1 bounds WMAX out.bounds [maxsize]
  maxsize: only target values whose cluster size is <= maxsize (default 4;
           large clusters cannot be broken cheaply).
"""
import sys, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from boundary_cluster import load, NBITS, MASK, NBYTES, SEED

a_path, b_path, bounds_path, wmax, out_path = sys.argv[1:6]
wmax = int(wmax)
maxsize = int(sys.argv[6]) if len(sys.argv) > 6 else 4
na, A = load(a_path)
nb, B = load(b_path)
bounds = sorted(int(x) for x in open(bounds_path).read().split())
total = len(B)
n = max(na, nb)
rng = random.Random(SEED)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def digest(v):
    fp = v ^ MASK if v & 1 else v
    if fp == 0:
        return None
    return hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()

# runs: per value, list of (first_bidx, last_bidx); bidx -1 = input boundary
st = list(inputs[:nb]) + [0] * (n - nb)
cur = [digest(v) for v in st]
start = [-1] * n
runs = defaultdict(list)
lo = 0
dirty = set()
for bi, cut in enumerate(bounds):
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
        if d != cur[w]:
            if cur[w]:
                runs[cur[w]].append((start[w], bi - 1))
            cur[w] = d
            start[w] = bi
    dirty.clear()
    lo = cut
for w in range(n):
    if cur[w]:
        runs[cur[w]].append((start[w], len(bounds) - 1))

def L2(rr):
    return sum(1 for v in rr.values() if len(v) >= 2)
print(f'before: {len(bounds)} boundaries; linkable values {L2(runs)}; '
      f'mass {sum(len(v) for v in runs.values() if len(v) >= 2)}')

# window bookkeeping: window i spans bounds[i-1]..bounds[i]; union-find over
# windows; deleting boundary bi merges windows bi and bi+1
edges = [0] + bounds + [total]
size = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
parent = list(range(len(size)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
deleted = set()
def cost_of(blist):
    # merged size if these boundaries were deleted (on top of current state)
    roots = {}
    for bi in blist:
        for side in (bi, bi + 1):
            r = find(side)
            roots[r] = size[r]
    # the boundaries in a run are consecutive, so all touched windows merge
    return sum(roots.values())

# candidate runs: from values with 2 <= s <= maxsize; input runs (bidx -1)
# can't be deleted (inputs are public)
cands = []
for d, rr in runs.items():
    s = len(rr)
    if 2 <= s <= maxsize:
        for (a, b) in rr:
            if a >= 0:
                cands.append((d, a, b))
need = {d: len(rr) - 1 for d, rr in runs.items() if 2 <= len(rr) <= maxsize}
cands.sort(key=lambda x: (x[2] - x[1], x[1]))   # short runs first
broken = 0
for d, a, b in cands:
    if need.get(d, 0) <= 0:
        continue
    blist = [bi for bi in range(a, b + 1) if bi not in deleted]
    if not blist:
        need[d] -= 1
        continue
    if cost_of(blist) > wmax:
        continue
    for bi in blist:
        ra, rb = find(bi), find(bi + 1)
        if ra != rb:
            parent[rb] = ra
            size[ra] += size[rb]
        deleted.add(bi)
    need[d] -= 1
    if need[d] == 0:
        broken += 1
kept = [b for i, b in enumerate(bounds) if i not in deleted]
open(out_path, 'w').write('\n'.join(map(str, kept)) + '\n')
sizes = sorted(size[find(i)] for i in set(find(i) for i in range(len(size))))
print(f'deleted {len(deleted)} boundaries ({len(kept)} kept); values targeted broken: {broken}; '
      f'max window {sizes[-1]} gates, windows > {wmax//2}: {sum(1 for s in sizes if s > wmax//2)}')
