"""Split every window whose ANF exceeds BUDGET monomials by recursive
bisection (gate midpoint) until each piece fits. Adds boundaries, so
re-score exposure afterwards.
usage: anf_split.py packed.mpmct1 bounds BUDGET out.bounds"""
import sys
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from anf_compose import load, window_anf
p_path, b_path, budget, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
n, gates = load(p_path)
m = len(gates)
bounds = sorted(int(x) for x in open(b_path).read().split() if x.strip())
cuts = [0] + [b for b in bounds if 0 < b < m] + [m]
new = []
splits = 0
def fit(lo, hi):
    global splits
    changed, total = window_anf(gates, lo, hi, budget)
    if changed is not None or hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    splits += 1
    new.append(mid)
    fit(lo, mid)
    fit(mid, hi)
for lo, hi in zip(cuts, cuts[1:]):
    fit(lo, hi)
out = sorted(set(bounds) | set(new))
open(out_path, 'w').write('\n'.join(map(str, out)) + '\n')
print(f'{p_path}: {splits} splits added ({len(bounds)} -> {len(out)} boundaries)')
