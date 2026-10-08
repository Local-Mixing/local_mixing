import sys
from collections import defaultdict
for tag in sys.argv[1:]:
    last = {}
    for l in open(f'{tag}/exp_C_vs_out.exposure.tsv'):
        if l.startswith('#'): continue
        g, w, t = map(int, l.split('\t')); last[w] = max(last.get(w, -1), g)
    outs = set(last.values())
    bounds = [int(x) for x in open(f'{tag}/packed.mpmct1.bounds').read().split()]
    bidx = {b: i for i, b in enumerate(bounds)}
    cuts = defaultdict(list)
    for l in open(f'{tag}/exp_C_vs_out.allcuts.tsv'):
        if l.startswith('#'): continue
        g, w, c, wt = map(int, l.split('\t'))
        if g not in outs:
            cuts[g].append((c, wt))
    print(f'== {tag}: {len(cuts)} interior leaked values')
    for g in sorted(cuts):
        cs = sorted(cuts[g]); idx = sorted(bidx[c] for c, _ in cs if c in bidx)
        runs = 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)
        print(f'  C gate {g:4d}: exposed at {len(cs):3d} boundaries in {runs} run(s); '
              f'span {cs[0][0]}..{cs[-1][0]} ({cs[-1][0]-cs[0][0]} gates); weights {sorted(set(w for _, w in cs))}')
