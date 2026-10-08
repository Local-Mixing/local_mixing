"""Merge-only boundary repair: delete every delivered boundary at which an
interior C value is exactly affinely recoverable (from the all-cuts scan).
Deleting cuts only shrinks the cut lattice, so no value -- C or stage-2 --
can become newly visible; the cost is larger fused windows.

usage: merge_bounds.py TAG   (writes TAG/merged.bounds)
"""
import sys

tag = sys.argv[1]
last = {}
for l in open(f'{tag}/exp_C_vs_out.exposure.tsv'):
    if l.startswith('#'):
        continue
    g, w, t = map(int, l.split('\t'))
    last[w] = max(last.get(w, -1), g)
outs = set(last.values())
bounds = [int(x) for x in open(f'{tag}/packed.mpmct1.bounds').read().split()]
total = int(open(f'{tag}/packed.mpmct1').readline().split()[2])
drop = set()
for l in open(f'{tag}/exp_C_vs_out.allcuts.tsv'):
    if l.startswith('#'):
        continue
    g, w, c, wt = map(int, l.split('\t'))
    if g not in outs:
        drop.add(c)
assert drop <= set(bounds), "exposing cut is not a delivered boundary"
kept = [b for b in bounds if b not in drop]
open(f'{tag}/merged.bounds', 'w').write('\n'.join(map(str, kept)) + '\n')
sizes = [b - a for a, b in zip([0] + kept, kept + [total])]
old = [b - a for a, b in zip([0] + bounds, bounds + [total])]
big = sorted((s for s in sizes if s > max(old)), reverse=True)
print(f'{tag}: removed {len(drop)} of {len(bounds)} boundaries; windows {len(old)} -> {len(sizes)}; '
      f'max window {max(old)} -> {max(sizes)} gates; fused windows over the old cap: {big}')
