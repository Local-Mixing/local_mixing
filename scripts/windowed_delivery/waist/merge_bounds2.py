"""merge_bounds with explicit paths: delete every boundary at which an
interior C value is exactly affine (from an all-cuts scan).
usage: merge_bounds2.py exposure.tsv allcuts.tsv bounds_in bounds_out"""
import sys
exp, allc, bin_, bout = sys.argv[1:5]
last = {}
for l in open(exp):
    if l.startswith('#'): continue
    g, w, t = map(int, l.split('\t')); last[w] = max(last.get(w, -1), g)
outs = set(last.values())
drop = set()
for l in open(allc):
    if l.startswith('#'): continue
    g, w, c, wt = map(int, l.split('\t'))
    if g not in outs: drop.add(c)
bounds = [int(x) for x in open(bin_).read().split()]
kept = [b for b in bounds if b not in drop]
open(bout, 'w').write('\n'.join(map(str, kept)) + '\n')
print(f'merge: removed {len(set(bounds) & drop)} of {len(bounds)} boundaries')
