"""Commutation-aware repair of a packed (windowed) circuit. Function-preserving.

Two passes, both legal rewrites of the gate sequence; window boundaries are
markers that stay between the same surviving gates.

  cancel  Delete pairs of IDENTICAL gates that can be brought adjacent by
          commutation (every gate between them commutes with them): a
          do/undo the mixer left in place. Iterated to a fixed point.
  fix     For each interior C leak (witness: predictor support S at a cut),
          the relation lives while all of S's current segments coexist:
          from the last write to S before the cut (the COMPLETER) to the
          first write to S after it (the BREAKER). Slide the completer
          forward and the breaker backward as far as commutation allows;
          the relation then spans fewer (ideally no) boundaries.

Gates commute iff neither's target is a control of the other (same-target
writes commute).

usage: order_repair.py in.mpmct1 in.bounds out.mpmct1 out.bounds cancel
       order_repair.py in.mpmct1 in.bounds out.mpmct1 out.bounds fix witness.tsv exposure.tsv
"""
import sys, bisect
from collections import defaultdict

def load(path):
    with open(path) as f:
        hdr = f.readline().split()
        n = int(hdr[1])
        gates = []
        for line in f:
            q = line.split()
            if not q:
                continue
            t, comp, k = int(q[0]), int(q[1]), int(q[2])
            cps = tuple(int(q[3 + 2 * i]) for i in range(k))
            pol = tuple(int(q[4 + 2 * i]) for i in range(k))
            gates.append((t, comp, cps, pol))
    return n, gates

def save(path, n, gates):
    with open(path, 'w') as f:
        f.write(f'mpmct1 {n} {len(gates)}\n')
        for t, comp, cps, pol in gates:
            f.write(f'{t} {comp} {len(cps)} ' + ' '.join(f'{c} {p}' for c, p in zip(cps, pol)) + '\n')

def to_seq(gates, bounds):
    """Gates interleaved with boundary markers (None)."""
    bset = sorted(set(b for b in bounds if 0 < b < len(gates)))
    seq, k = [], 0
    for i, g in enumerate(gates):
        while k < len(bset) and bset[k] == i:
            seq.append(None); k += 1
        seq.append(g)
    return seq

def from_seq(seq):
    gates, bounds = [], []
    for x in seq:
        if x is None:
            if gates and (not bounds or bounds[-1] != len(gates)):
                bounds.append(len(gates))
        else:
            gates.append(x)
    return gates, [b for b in bounds if b < len(gates)]

def commute(a, b):
    return a[0] not in b[2] and b[0] not in a[2]

def cancel_pass(gates):
    """One pass; returns (kept gates, number of pairs cancelled)."""
    m = len(gates)
    reads = defaultdict(list)    # wire -> positions reading it
    writes = defaultdict(list)   # wire -> positions writing it
    same = defaultdict(list)     # gate -> positions
    for i, g in enumerate(gates):
        writes[g[0]].append(i)
        for c in g[2]:
            reads[c].append(i)
        same[g].append(i)
    def nxt(lst, i):
        k = bisect.bisect_right(lst, i)
        return lst[k] if k < len(lst) else m
    dead = bytearray(m)
    pairs = 0
    for i, g in enumerate(gates):
        if dead[i]:
            continue
        blk = nxt(reads[g[0]], i)
        for c in g[2]:
            blk = min(blk, nxt(writes[c], i))
        lst = same[g]
        k = bisect.bisect_right(lst, i)
        while k < len(lst) and lst[k] < blk and dead[lst[k]]:
            k += 1
        if k < len(lst) and lst[k] < blk:
            j = lst[k]
            dead[i] = dead[j] = 1
            pairs += 1
    return dead, pairs

def run_cancel(seq):
    total = 0
    while True:
        gidx = [i for i, x in enumerate(seq) if x is not None]
        gates = [seq[i] for i in gidx]
        dead, pairs = cancel_pass(gates)
        if pairs == 0:
            return seq, total
        drop = {gidx[k] for k in range(len(gates)) if dead[k]}
        seq = [x for i, x in enumerate(seq) if i not in drop]
        total += pairs
        print(f'  cancel pass: {pairs} pairs')

def slide(seq, pos, direction):
    """Move the gate at seq[pos] as far as commutation allows (past markers
    too); returns its new index."""
    g = seq[pos]
    dest = pos
    j = pos + direction
    while 0 <= j < len(seq):
        x = seq[j]
        if x is not None:
            if not commute(g, x):
                break
            dest = j
        j += direction
    if dest == pos:
        return pos
    # land just before the blocker (forward) / just after it (backward),
    # i.e. after any markers that precede the blocker in the move direction
    land = j - direction if 0 <= j < len(seq) else (len(seq) - 1 if direction > 0 else 0)
    item = seq.pop(pos)
    if direction > 0:
        seq.insert(land, item)     # indices after pos shifted down by one
        return land
    seq.insert(land, item)
    return land

def run_fix(seq, wit_path, exp_path):
    last = {}
    for l in open(exp_path):
        if l.startswith('#'):
            continue
        g, w, t = map(int, l.split('\t'))
        last[w] = max(last.get(w, -1), g)
    outs = set(last.values())
    moved = 0
    for l in open(wit_path):
        if l.startswith('#'):
            continue
        f = l.rstrip('\n').split('\t')
        cg, cut, sup = int(f[0]), int(f[3]), [int(x) for x in f[4].split(',') if x]
        if cg in outs or not sup:
            continue
        # gate positions in seq
        gpos = [i for i, x in enumerate(seq) if x is not None]
        sset = set(sup)
        comp_i = max((gi for k, gi in enumerate(gpos[:cut]) if seq[gi][0] in sset), default=None)
        brk_i = min((gi for gi in gpos[cut:] if seq[gi][0] in sset), default=None)
        if comp_i is not None:
            new = slide(seq, comp_i, +1)
            moved += new != comp_i
        gpos = [i for i, x in enumerate(seq) if x is not None]
        if brk_i is not None:
            # breaker index may have shifted by the completer move: re-find the
            # first write to S after the (unchanged) cut index
            brk_i = min((gi for gi in gpos[cut:] if seq[gi][0] in sset), default=None)
            if brk_i is not None:
                new = slide(seq, brk_i, -1)
                moved += new != brk_i
        print(f'  C gate {cg}: support {sup} completer/breaker slid')
    return seq, moved

def main():
    inp, binp, outp, bout, mode = sys.argv[1:6]
    n, gates = load(inp)
    bounds = [int(x) for x in open(binp).read().split()]
    seq = to_seq(gates, bounds)
    if mode == 'cancel':
        seq, pairs = run_cancel(seq)
        print(f'cancelled {pairs} pairs ({2*pairs} gates)')
    elif mode == 'fix':
        seq, moved = run_fix(seq, sys.argv[6], sys.argv[7])
        print(f'moved {moved} gates')
    g2, b2 = from_seq(seq)
    save(outp, n, g2)
    open(bout, 'w').write('\n'.join(map(str, b2)) + '\n')
    print(f'{len(gates)} -> {len(g2)} gates; {len(bounds)} -> {len(b2)} boundaries')

if __name__ == '__main__':
    main()
