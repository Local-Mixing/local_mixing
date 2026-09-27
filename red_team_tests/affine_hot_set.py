#!/usr/bin/env python3
"""EXACT affine-exposure analysis of a blinded-V5 gadget (mpmct1).

A data wire's value is `w ⊕ c ⊕ s_w ⊕ Q_w`, where the constant and the linear
part `s_w` are sums of single band wires — visible, hence affine — and only the
quadratic part `Q_w = Σ x_i·y_i` hides anything. So an affine adversary reading
the wire state at one instant recovers a GF(2) combination of plaintexts exactly
when some combination of data wires has ALL its quadratic terms cancel:

    Σ_{w∈S} Q_w = 0   over the open mask pairs.

The hot set is therefore every (wire, interval) segment that takes part in such a
cancelling combination, plus the I/O fringe (a wire with no open pair at all is
the |S|=1 case). Deny the adversary those segments and no affine combination of
plaintexts is recoverable at that instant — every surviving combination keeps an
uncancelled quadratic term.

Usage: affine_hot_set.py <gadget.mpmct1> [NP=256] [R=256] [samples=400]
"""
import sys, collections, bisect

path = sys.argv[1]
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R = int(sys.argv[3]) if len(sys.argv) > 3 else 256
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 400

L = [l for l in open(path).read().split("\n")[1:] if l]
G = []
for l in L:
    t = l.split()
    G.append((int(t[0]), int(t[1]), [(int(t[3 + 2 * i]), int(t[4 + 2 * i])) for i in range(int(t[2]))]))
n = len(G)

def is_pair(comp, ls):
    if comp != 1 or len(ls) != 2: return None
    (w0, p0), (w1, p1) = ls
    if p0 == p1: return None
    if not (NP <= w0 < NP + R and NP <= w1 < NP + R): return None
    return (w0, w1) if w0 <= w1 else (w1, w0)

# replay, sampling the open-pair sets of every data wire at NS instants
open_pairs = [set() for _ in range(NP)]
marks = sorted(set(int(i) for i in [j * (n - 1) // (NS - 1) for j in range(NS)]))
mark_at = set(marks)
hot = collections.defaultdict(list)   # wire -> list of (start,end) hot intervals
state = {}                            # wire -> start index of its current hot run
def kernel_wires(snapshot):
    """Wires taking part in ANY cancelling combination = wires whose quadratic
    part lies in the span of the others. Computed as the set of columns that are
    NOT pivots of the pair-incidence matrix, together with everything that
    supports a dependency."""
    # GF(2) elimination over pair-indexed vectors, one column per data wire
    basis = {}          # pivot pair -> (vector, wire-set)
    dependent = set()
    for w, prs in snapshot:
        vec = set(prs); tag = {w}
        for p in sorted(vec):
            if p in basis and p in vec:
                bv, bt = basis[p]
                vec ^= bv; tag ^= bt
        if not vec:
            dependent |= tag        # this wire is a sum of earlier ones: all involved are hot
        else:
            basis[min(vec)] = (vec, tag)
    return dependent

for i, (tg, comp, ls) in enumerate(G):
    if tg < NP:
        pr = is_pair(comp, ls)
        if pr is not None:
            if pr in open_pairs[tg]: open_pairs[tg].remove(pr)
            else: open_pairs[tg].add(pr)
    if i in mark_at:
        snap = [(w, open_pairs[w]) for w in range(NP)]
        bad = kernel_wires(snap)
        for w in range(NP):
            if w in bad:
                if w not in state: state[w] = i
            elif w in state:
                hot[w].append((state.pop(w), i))
for w, st in state.items():
    hot[w].append((st, n))

fringe = [(w,s,e) for w,v in hot.items() for s,e in v if s == 0 or e == n]
inner  = [(w,s,e) for w,v in hot.items() for s,e in v if s != 0 and e != n]
print(f"  -- I/O fringe segments: {len(fringe)}")
print(f"  -- INTERIOR segments (mask cancellation across wires): {len(inner)}")
for w,s,e in sorted(inner, key=lambda t:-(t[2]-t[1]))[:8]:
    print(f"       wire {w:>3}  gates {s:>9,}..{e:<9,}  ({e-s:,} gates, {100*s/n:.0f}%-{100*e/n:.0f}% through)")
tot = sum(e - s for v in hot.values() for s, e in v)
segs = sum(len(v) for v in hot.values())
print(f"{path.split('/')[-1]}: {n:,} gates, {NP} data wires, {NS} sampled instants")
print(f"  hot segments (affine-exposed by cancellation or empty mask): {segs} over {len(hot)} wires")
print(f"  hot wire-time: {tot:,} of {n*NP:,} wire-gate slots = {100*tot/(n*NP):.3f}%")
by = collections.Counter("payload half" if w < NP//2 else "high half" for w in hot for _ in hot[w])
print(f"  by half: {dict(by)}")
