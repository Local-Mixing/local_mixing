#!/usr/bin/env python3
"""Sparse ANF composition per window: the real deliverable-size story.

Each window of a packed circuit becomes the vector ANF of its touched
wires over the window's input wires. Polynomials are sets of monomials;
a monomial is an int bitmask over global wires (bit w = wire w's value at
the WINDOW INPUT); GF(2) cancellation is the set symmetric difference.
Correctness check: each window's ANF is re-evaluated on the shared random
samples and compared against gate-by-gate evaluation of the window.

Budget: a window whose total monomial count exceeds `budget` is counted
as OVERFLOW (in production it would be split); its cost falls back to
its gate-list bits.

Encodings reported (per window):
  header: f input wires (8 bits each) + per changed wire: wire id + count
  monomials: bitmask over the f window inputs (f bits each)
  gate list baseline: 8(target)+1(comp)+4(k)+k*9 bits per gate

usage: anf_compose.py packed.mpmct1 packed.mpmct1.bounds [budget] [label] [check_every]
"""
import sys, random
from collections import defaultdict

LANES = 4                     # 256 samples: plenty for a per-window check
NBITS = LANES * 64
MASK = (1 << NBITS) - 1
SEED = 20261004

def load(path):
    with open(path) as f:
        hdr = f.readline().split()
        n_hdr = int(hdr[1])
        gates = []
        maxw = 0
        for line in f:
            q = line.split()
            if not q:
                continue
            t, comp, k = int(q[0]), int(q[1]), int(q[2])
            cps = tuple(int(q[3 + 2*i]) for i in range(k))
            pol = tuple(int(q[4 + 2*i]) for i in range(k))
            m = max((t,) + cps)
            if m > maxw:
                maxw = m
            gates.append((t, comp, cps, pol))
    return max(n_hdr, maxw + 1), gates

def mul(Ap, Bp, cap):
    if len(Ap) > len(Bp):
        Ap, Bp = Bp, Ap
    res = set()
    add = res.add
    discard = res.discard
    for a in Ap:
        for b in Bp:
            v = a | b
            if v in res:
                discard(v)      # GF(2): colliding monomials cancel pairwise
            else:
                add(v)
        if len(res) > cap:
            return None
    return res

def window_anf(gates, lo, hi, budget):
    polys = {}                          # wire -> set of monomial masks
    for t, comp, cps, pol in gates[lo:hi]:
        prod = {0}
        for c, p in zip(cps, pol):
            cp = polys.get(c)
            if cp is None:
                cp = {1 << c}
            if p == 0:
                cp = cp ^ {0}
            prod = mul(prod, cp, budget * 8)
            if prod is None:
                return None, None
        if comp:
            prod ^= {0}
        tgt = polys.get(t)
        if tgt is None:
            tgt = {1 << t}
        polys[t] = tgt ^ prod
        if sum(len(v) for v in polys.values()) > budget * 4:
            return None, None
    changed = {w: p for w, p in polys.items() if p != {1 << w}}
    total = sum(len(p) for p in changed.values())
    if total > budget:
        return None, None
    return changed, total

def check_window(gates, lo, hi, changed, n, rng):
    xs = [rng.getrandbits(NBITS) for _ in range(n)]
    st = list(xs)
    for t, comp, cps, pol in gates[lo:hi]:
        acc = MASK
        for c, p in zip(cps, pol):
            acc &= st[c] if p else st[c] ^ MASK
        if comp:
            acc ^= MASK
        st[t] ^= acc
    for w, poly in changed.items():
        v = 0
        for mono in poly:
            if mono == 0:
                v ^= MASK
                continue
            term = MASK
            mm = mono
            while mm:
                low = mm & -mm
                term &= xs[low.bit_length() - 1]
                mm ^= low
            v ^= term
        if v != st[w]:
            return False
    for w in range(n):
        if w not in changed and st[w] != xs[w]:
            return False
    return True

def main():
    c_path, b_path = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    label = sys.argv[4] if len(sys.argv) > 4 else "anf"
    check_every = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    n, gates = load(c_path)
    bounds = [int(x) for x in open(b_path) if x.strip()]
    m = len(gates)
    cuts = [0] + sorted(b for b in bounds if 0 < b < m) + [m]
    rng = random.Random(SEED)

    tot_anf_bits = tot_gate_bits = tot_monos = 0
    overflow = checked = failed = 0
    per_win = []
    for wi in range(len(cuts) - 1):
        lo, hi = cuts[wi], cuts[wi + 1]
        gate_bits = sum(8 + 1 + 4 + 9 * len(c) for _, _, c, _ in gates[lo:hi])
        tot_gate_bits += gate_bits
        changed, total = window_anf(gates, lo, hi, budget)
        if changed is None:
            overflow += 1
            tot_anf_bits += gate_bits   # fallback: keep as gates / split later
            continue
        wires = set()
        for p in changed.values():
            for mono in p:
                mm = mono
                while mm:
                    low = mm & -mm
                    wires.add(low.bit_length() - 1)
                    mm ^= low
        f = max(len(wires), 1)
        bits = 8 + f * 8 + len(changed) * (8 + 16) + total * f
        tot_anf_bits += bits
        tot_monos += total
        per_win.append((hi - lo, total, bits, gate_bits))
        if wi % check_every == 0:
            checked += 1
            if not check_window(gates, lo, hi, changed, n, rng):
                failed += 1
                print(f"  CHECK FAILED window {wi} [{lo},{hi})")
    nw = len(cuts) - 1
    per_win.sort(key=lambda r: r[1])
    k = len(per_win)
    print(f"{label}: {nw} windows, budget={budget} monomials; "
          f"overflow {overflow} ({100*overflow/nw:.2f}%)")
    if k:
        print(f"monomials/window: med={per_win[k//2][1]} p90={per_win[9*k//10][1]} "
              f"p99={per_win[99*k//100][1]} max={per_win[-1][1]}; total={tot_monos}")
    print(f"size: ANF {tot_anf_bits/8e6:.1f} MB vs gate-list {tot_gate_bits/8e6:.1f} MB "
          f"(x{tot_anf_bits/max(tot_gate_bits,1):.2f})")
    print(f"spot checks: {checked} windows re-evaluated, {failed} failures"
          + ("  ** FUNCTIONAL MISMATCH **" if failed else " (OK)"))

if __name__ == "__main__":
    main()
