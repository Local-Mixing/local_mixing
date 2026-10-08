"""Trace C-view leak witnesses back to stage-2 provenance.

usage: trace_witness.py TAG [max_rows]
Reads waist/TAG/{tdp.mpmct1, packed.mpmct1, exp_C_vs_out.witness.tsv, gen.log}.
For every witness (C gate, cut in packed coordinates, predictor support
wires) it fingerprints each support wire's value at that cut and looks it up
among stage-2 born values: birth gate, stage-2 region, wire class of the
birth, gate shape, and multiplicity. Interior leaks only (outputs skipped).
"""
import sys, re, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

tag = sys.argv[1]
max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 50
R = f'/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist/{tag}'
na, A = load(f'{R}/tdp.mpmct1')
npk, P = load(f'{R}/packed.mpmct1')
n = max(na, npk)

g = open(f'{R}/gen.log').read()
m = re.search(r'(\d+) band-seed \+ (\d+) band-reseed \+ (\d+) slice-guard', g)
seed_n, reseed_n, guard_n = map(int, m.groups())
total = len(A)
compute_n = total - seed_n - reseed_n - 2 * guard_n
edges = [(guard_n, 'guard-in'), (guard_n + seed_n, 'band-seed'),
         (guard_n + seed_n + compute_n, 'compute'),
         (guard_n + seed_n + compute_n + reseed_n, 'reseed'), (total, 'guard-out')]
def region(i):
    for e, name in edges:
        if i < e:
            return name
    return '?'
half = n // 2          # sandwich width; band = half..n
def wclass(w):
    return 'data' if w < half // 2 else ('payload' if w < half else 'band')

rows = []
for line in open(f'{R}/exp_C_vs_out.witness.tsv'):
    if line.startswith('#'):
        continue
    gi, w, t, cut, wires = line.rstrip('\n').split('\t')
    rows.append((int(gi), int(w), int(t), int(cut), [int(x) for x in wires.split(',') if x]))
# outputs = last writer per C wire among all witnessed rows is not enough;
# use the exposure tsv for the full last-writer set
last = {}
for line in open(f'{R}/exp_C_vs_out.exposure.tsv'):
    if line.startswith('#'):
        continue
    a, b, _ = line.split('\t')
    last[int(b)] = max(last.get(int(b), -1), int(a))
outs = set(last.values())
rows = [r for r in rows if r[0] not in outs][:max_rows]
print(f'{tag}: {len(rows)} interior witnesses; stage-2 regions guard={guard_n} seed={seed_n} '
      f'compute={compute_n} reseed={reseed_n}; packed gates={len(P)}')

rng = random.Random(7)
inputs = [rng.getrandbits(NBITS) for _ in range(n)]
def norm(v): return v ^ MASK if v & 1 else v
def dig(v): return hashlib.blake2b(norm(v).to_bytes(NBYTES, 'little'), digest_size=16).digest()

born = {}
mult = defaultdict(int)
st = list(inputs[:na]) + [0] * (n - na)
for i, (t, comp, cps, pol) in enumerate(A):
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp:
        acc ^= MASK
    st[t] ^= acc
    d = dig(st[t])
    mult[d] += 1
    if d not in born:
        born[d] = (i, t, len(cps), comp)
init = {dig(inputs[w]): w for w in range(na)}

need = defaultdict(list)
for k, r in enumerate(rows):
    need[r[3]].append(k)
st = list(inputs[:npk]) + [0] * (n - npk)
snap = {}
for i in range(len(P) + 1):
    if i in need:
        for k in need[i]:
            snap[k] = {w: st[w] for w in rows[k][4]}
    if i == len(P):
        break
    t, comp, cps, pol = P[i]
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp:
        acc ^= MASK
    st[t] ^= acc

summary = defaultdict(int)
for k, (gi, w, t, cut, wires) in enumerate(rows):
    print(f'\nC gate {gi} (wire {w}) t*={t} at packed cut {cut} ({100*cut/len(P):.1f}%)')
    for x in wires:
        d = dig(snap[k][x])
        if d in born:
            i, bt, nc, comp = born[d]
            lab = f'{region(i)}:{wclass(bt)} born@{i} ({100*i/total:.1f}%) ctrls={nc} comp={int(comp)} mult={mult[d]}'
            summary[(region(i), wclass(bt), nc)] += 1
        elif d in init:
            lab = f'INPUT wire {init[d]}'
            summary[('input', '', 0)] += 1
        else:
            lab = 'NOT a stage-2 value (mixer-born)'
            summary[('mixer', '', 0)] += 1
        print(f'   G wire {x:3d} [{wclass(x)}]: {lab}')
print('\nsupport-value provenance totals:')
for key, v in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f'  {v:4d}  {key}')
