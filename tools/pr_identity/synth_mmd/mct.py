"""Multi-control Toffoli -> {CNOT, 2-control Toffoli, NOT} using DIRTY borrowed
ancillas (Barenco Lemma 7.3 recursion, g1 g2 g1 g2; each ancilla restored).

Emits base gates as ('X',t) / ('CX',c,t) / ('CCX',c1,c2,t), all positive
controls.  Verified by full-table equality incl. dirty-ancilla restoration.
"""
import numpy as np
from math import ceil

def decompose(controls, target, spare, out):
    """Append base gates realizing target ^= AND(controls) (positive).
    `spare` = list of borrowable dirty wires (disjoint from controls+target)."""
    k = len(controls)
    if k == 0:
        out.append(('X', target)); return
    if k == 1:
        out.append(('CX', controls[0], target)); return
    if k == 2:
        out.append(('CCX', controls[0], controls[1], target)); return
    assert spare, f"need >=1 dirty ancilla for k={k}"
    a = spare[0]
    h = ceil(k / 2)
    C1, C2 = controls[:h], controls[h:]
    poolA = C2 + [target] + spare[1:]     # dirty wires usable while writing a
    poolB = C1 + spare[1:]                # dirty wires usable while writing target
    g1, g2 = [], []
    decompose(C1, a, poolA, g1)           # a ^= AND(C1)
    decompose(C2 + [a], target, poolB, g2)  # target ^= AND(C2)*a
    out += g1 + g2 + g1 + g2

def base_table(gates, n):
    st = np.arange(1 << n, dtype=np.int64)
    for gt in gates:
        if gt[0] == 'X':
            t = gt[1]; st = st ^ (1 << t)
        elif gt[0] == 'CX':
            c, t = gt[1], gt[2]
            fire = (st >> c) & 1
            st = st ^ (fire << t)
        else:
            c1, c2, t = gt[1], gt[2], gt[3]
            fire = ((st >> c1) & 1) & ((st >> c2) & 1)
            st = st ^ (fire << t)
    return st

def mct_table(controls, target, n):
    st = np.arange(1 << n, dtype=np.int64)
    fire = np.ones(1 << n, dtype=np.int64)
    for c in controls:
        fire &= (st >> c) & 1
    return st ^ (fire << target)

def selftest():
    import itertools, random
    rng = random.Random(0)
    print("=== MCT dirty-ancilla decomposition verification ===")
    allok = True
    for n in range(3, 10):
        for target in range(n):
            others = [w for w in range(n) if w != target]
            for k in range(0, min(len(others), n - 1) + 1):
                # need k-... spare: choose controls, rest are spare
                if k > len(others):
                    continue
                # try a few random control choices
                for _ in range(3):
                    controls = rng.sample(others, k)
                    spare = [w for w in others if w not in controls]
                    # decomposition needs >=1 spare only when k>=3
                    if k >= 3 and not spare:
                        continue
                    out = []
                    try:
                        decompose(controls, target, spare, out)
                    except AssertionError:
                        continue
                    got = base_table(out, n)
                    want = mct_table(controls, target, n)
                    ok = np.array_equal(got, want)
                    allok &= ok
                    if not ok:
                        print(f"  FAIL n={n} k={k} target={target} controls={controls}")
    # specifically the tight w=6 -> canvas 9 case: k=5 with 3 spare
    n = 9
    for _ in range(20):
        controls = rng.sample(range(6), 5)
        target = [w for w in range(6) if w not in controls][0]
        spare = [6, 7, 8]
        out = []
        decompose(controls, target, spare, out)
        got = base_table(out, n); want = mct_table(controls, target, n)
        allok &= np.array_equal(got, want)
    print("  #CCX for a k=5 MCT:", sum(1 for g in out if g[0]=='CCX'),
          " total base gates:", len(out))
    print("ALL MCT DECOMPS VERIFIED:", allok)

if __name__ == "__main__":
    selftest()
