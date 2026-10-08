"""Exposure-aware ANF split: like anf_split.py, but each over-budget
window is split at a candidate cut where NO interior C value is exactly
affine (one hmap_affine C-view scan over all candidate cuts) and both
halves are closer to fitting; recursion until all fit.
usage: anf_split2.py TAG G N BUDGET bounds_in bounds_out"""
import sys, subprocess, os
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from anf_compose import load, window_anf
tag, G, N, budget, bin_, bout = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
ROOT = '/Users/rancanetti/code/local_mixing'
R = f'{ROOT}/runs/ncdb_ab_20261001/waist/{tag}'
P = os.environ.get('SPLIT_P', f'{R}/packed_G{G}.mpmct1')
n, gates = load(P)
m = len(gates)
bounds = sorted(int(x) for x in open(bin_).read().split() if x.strip())
# output gates of C (last writer per wire) from a prior full C-view tsv
last = {}
for l in open(os.environ.get('SPLIT_EXP', f'{R}/fin.exposure.tsv')):
    if l.startswith('#'): continue
    g, w, t = map(int, l.split('\t')); last[w] = max(last.get(w, -1), g)
outs = set(last.values())
def leaky_cuts(cands):
    cf = f'{R}/split_cands_{os.getpid()}.txt'
    open(cf, 'w').write('\n'.join(map(str, cands)) + '\n')
    env = dict(os.environ, EXPOSURE_ALLCUTS=f'{R}/split_cands_{os.getpid()}.allcuts.tsv')
    subprocess.run([f'{ROOT}/target/release/hmap_affine', '--c', f'{R}/tdp.mpmct1.source_c.g57',
                    '--c-format', 'g57', '--g', P, '--n', str(N), '--exposure', '--c-step', '1',
                    '--g-cuts', cf, '--out', f'{R}/split_cands_{os.getpid()}'], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bad = set()
    for l in open(f'{R}/split_cands_{os.getpid()}.allcuts.tsv'):
        if l.startswith('#'): continue
        g, w, c, wt = map(int, l.split('\t'))
        if g not in outs: bad.add(c)
    return bad
new = []
def fit(lo, hi, depth=0):
    if window_anf(gates, lo, hi, budget)[0] is not None or hi - lo <= 2:
        return
    k = 32
    cands = sorted({lo + (hi - lo) * i // (k + 1) for i in range(1, k + 1)} - {lo, hi})
    bad = leaky_cuts(cands)
    good = [c for c in cands if c not in bad]
    if not good:
        good = cands      # fall back (reported)
        print(f'  WARNING: no exposure-free split in [{lo},{hi})')
    mid = (lo + hi) // 2
    c = min(good, key=lambda x: abs(x - mid))
    new.append(c)
    fit(lo, c, depth + 1)
    fit(c, hi, depth + 1)
cuts = [0] + [b for b in bounds if 0 < b < m] + [m]
for lo, hi in zip(cuts, cuts[1:]):
    fit(lo, hi)
out = sorted(set(bounds) | set(new))
open(bout, 'w').write('\n'.join(map(str, out)) + '\n')
print(f'{P}: {len(new)} splits added ({len(bounds)} -> {len(out)} boundaries)')
