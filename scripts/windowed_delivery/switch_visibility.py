#!/usr/bin/env python3
"""Classify carrier-refresh switches (mask/unmask toggle pairs) in a
packed deliverable:

  ABSORBED  - both toggles of a pair land in the same window: the pair
              cancels inside the composed ANF; the window polynomial is
              identical to one without the switch (zero trace). If the
              pair's original span was long, the reorder may however
              have defeated the protection (value bare elsewhere) --
              reported via the gap histogram; net exposure effects are
              already in the packer's ground-truth verify.
  STRADDLING- toggles in different windows: the masked value crosses
              the interior boundaries. Hidden from affine/equality
              adversaries; strippable by a quadratic adversary because
              the pair controls sit verbatim at the same boundaries
              (the stability constraint guarantees it).

usage: switch_visibility.py ins_dump order_dump bounds m_original
"""
import sys, bisect
from collections import defaultdict

ins_path, order_path, bounds_path, m_orig = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))

ins = []                        # (anchor, seq, t, c1, c2) sorted like the rebuild
for line in open(ins_path):
    a, s, t, c1, c2 = map(int, line.split())
    ins.append((a, s, t, c1, c2))
ins.sort(key=lambda r: (r[0], r[1]))

# position of insert j in the refreshed circuit = anchor_j + j
# position of original gate g = g + (#inserts with anchor <= g)
ins_pos = [ins[j][0] + j for j in range(len(ins))]

order = [int(x) for x in open(order_path)]
m_ref = len(order)
assert m_ref == m_orig + len(ins), (m_ref, m_orig, len(ins))
empos = [0] * m_ref
for p, g in enumerate(order):
    empos[g] = p
bounds = sorted(int(x) for x in open(bounds_path) if x.strip())

def win_of(emitted_pos):
    return bisect.bisect_right(bounds, emitted_pos)

# pair up toggles in refreshed-circuit execution order via (t,c1,c2) stack
stacks = defaultdict(list)
pairs = []                      # (pos1, pos2, anchor_gap)
for j in range(len(ins)):
    a, s, t, c1, c2 = ins[j]
    key = (t, c1, c2)
    if stacks[key]:
        j0 = stacks[key].pop()
        pairs.append((ins_pos[j0], ins_pos[j], ins[j][0] - ins[j0][0]))
    else:
        stacks[key].append(j)
unmatched = sum(len(v) for v in stacks.values())

same = cross = 0
gap_same, straddle = [], []
for p1, p2, agap in pairs:
    w1, w2 = win_of(empos[p1]), win_of(empos[p2])
    if w1 == w2:
        same += 1
        gap_same.append(agap)
    else:
        cross += 1
        straddle.append(abs(w2 - w1))
tot = len(pairs)
print(f"{tot} toggle pairs ({unmatched} unmatched); "
      f"ABSORBED same-window: {same} ({100*same/tot:.1f}%), "
      f"STRADDLING: {cross} ({100*cross/tot:.1f}%)")
if straddle:
    straddle.sort()
    k = len(straddle)
    print(f"boundaries straddled per crossing pair: med={straddle[k//2]} "
          f"p90={straddle[9*k//10]} max={straddle[-1]}; "
          f"total boundary crossings: {sum(straddle)}")
if gap_same:
    gap_same.sort()
    k = len(gap_same)
    big = sum(1 for g in gap_same if g > 512)
    print(f"absorbed pairs' original anchor gap: med={gap_same[k//2]} "
          f"p90={gap_same[9*k//10]} max={gap_same[-1]}; "
          f"gap>512 (reorder-collapsed, protection likely defeated): {big} "
          f"({100*big/max(tot,1):.1f}% of all pairs)")
