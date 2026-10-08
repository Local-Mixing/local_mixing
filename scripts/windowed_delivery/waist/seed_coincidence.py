"""Contingency: first-layer C gates (all three wires still raw inputs) whose
control pair coincides with a band-seed data-data product pair, vs leaked."""
import sys, re
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load
ALPH = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
D = {c: i for i, c in enumerate(ALPH)}
tot = {(m, l): 0 for m in (0, 1) for l in (0, 1)}
for tag in sys.argv[1:]:
    body = open(f'{tag}/tdp.mpmct1.source_c.g57').read().strip().strip(';')
    C = [tuple(D[ch] for ch in tok) for tok in body.split(';') if tok]
    written, first = set(), []
    for g, (a, x, y) in enumerate(C):
        if not ({a, x, y} & written):
            first.append(g)
        written.add(a)
    na, A = load(f'{tag}/tdp.mpmct1')
    gl = open(f'{tag}/gen.log').read()
    seed_n, _, guard_n = map(int, re.search(r'(\d+) band-seed \+ (\d+) band-reseed \+ (\d+) slice-guard', gl).groups())
    seedpairs = {frozenset(cps) for t, comp, cps, pol in A[guard_n:guard_n + seed_n]
                 if len(cps) == 2 and all(c < 64 for c in cps)}
    last = {}
    rows = [tuple(map(int, l.split('\t'))) for l in open(f'{tag}/exp_C_vs_out.exposure.tsv') if not l.startswith('#')]
    for g, w, t in rows:
        last[w] = max(last.get(w, -1), g)
    outs = set(last.values())
    leaked = {g for g, w, t in rows if t > 0 and g not in outs}
    tab = {(m, l): [] for m in (0, 1) for l in (0, 1)}
    for g in first:
        a, x, y = C[g]
        m = int(frozenset((x, y)) in seedpairs)
        l = int(g in leaked)
        tab[(m, l)].append(g)
        tot[(m, l)] += 1
    fl = sorted(leaked & set(first))
    print(f'{tag:8s} first-layer {len(first):3d}, seed data-pairs {len(seedpairs):3d} | matched: {len(tab[(1,0)])+len(tab[(1,1)]):2d} '
          f'(leaked {tab[(1,1)]}) | unmatched leaked {tab[(0,1)]} | all fringe leaks {fl}')
m1 = tot[(1, 0)] + tot[(1, 1)]; m0 = tot[(0, 0)] + tot[(0, 1)]
print(f'TOTAL: matched {m1} -> leaked {tot[(1,1)]} ({100*tot[(1,1)]/max(1,m1):.0f}%); '
      f'unmatched {m0} -> leaked {tot[(0,1)]} ({100*tot[(0,1)]/max(1,m0):.1f}%)')
