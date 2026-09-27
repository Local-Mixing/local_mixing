"""Transformation-based (MMD) reversible synthesis over positive-control
multi-control-Toffoli (MCT) gates, no ancilla at the logical level.

A logical gate = (target, controls_tuple) with all-positive controls; empty
controls = NOT.  MMD returns a list L of logical gates with
    (applied left-to-right)  L(P) = identity     [each gate self-inverse]
so  word_table(reverse(L)) == P.
"""
import numpy as np

def mmd_synth(P):
    """P: permutation array of length 2^w (P[i]=image of i).  Returns list of
    logical gates [(target,(ctrl,...)),...] whose left-to-right product == P."""
    N = len(P)
    w = N.bit_length() - 1
    assert 1 << w == N
    C = P.copy()                       # current permutation, mutate to identity
    found = []                         # g_1, g_2, ... in order applied to C

    def apply_gate(target, ctrls):
        """C := g ∘ C where g flips bit `target` iff all `ctrls` bits set."""
        if ctrls:
            mask = 0
            for c in ctrls:
                mask |= (1 << c)
            fire = ((C & mask) == mask)
        else:
            fire = np.ones(N, dtype=bool)
        C[fire] ^= (1 << target)
        found.append((target, tuple(sorted(ctrls))))

    for i in range(N):
        p = int(C[i])
        if p == i:
            continue
        # p >= i guaranteed (rows <i already fixed to themselves)
        ci = i                          # target bits pattern
        # step 1: set bits that are 1 in i but 0 in p, control on 1-bits of p
        pbits = [b for b in range(w) if (p >> b) & 1]
        for b in range(w):
            if ((i >> b) & 1) and not ((p >> b) & 1):
                apply_gate(b, pbits)     # control on original 1-bits of p
        # now current value of row i is p | i
        # step 2: clear bits that are 1 in (p|i) but 0 in i, control on 1-bits of i
        q = p | i
        ibits = [b for b in range(w) if (i >> b) & 1]
        for b in range(w):
            if ((q >> b) & 1) and not ((i >> b) & 1):
                apply_gate(b, ibits)
        assert int(C[i]) == i
    assert np.array_equal(C, np.arange(N)), "MMD did not reach identity"
    # word realizing P (left-to-right) is reverse of the applied sequence
    return list(reversed(found))

def verify_logical(gates, P):
    """Check left-to-right product of logical gates equals P."""
    N = len(P); w = N.bit_length()-1
    st = np.arange(N)
    for (t, ctrls) in gates:
        if ctrls:
            mask = 0
            for c in ctrls: mask |= (1 << c)
            fire = ((st & mask) == mask)
        else:
            fire = np.ones(N, dtype=bool)
        st = st ^ (fire.astype(np.int64) << t)
    # st now = product applied to identity input array => st[i] = product(i)
    return np.array_equal(st, P)

if __name__ == "__main__":
    import itertools, random, sys
    sys.path.insert(0, ".")
    import g57
    from geodesic import gates as ggates, make_unit
    m = 6
    rng = random.Random(3)
    from collections import Counter
    ksz = Counter(); nlog = []; maxk = []
    N = 200
    for _ in range(N):
        unit = make_unit(m, 6, rng)
        P = g57.word_table(unit, m)
        L = mmd_synth(P.copy())
        assert verify_logical(L, P), "logical verify failed"
        nlog.append(len(L))
        mk = 0
        for (t, ctrls) in L:
            ksz[len(ctrls)] += 1
            mk = max(mk, len(ctrls))
        maxk.append(mk)
    print(f"=== MMD on {N} real w=6 units ===")
    print(f"  logical gates/unit: mean {np.mean(nlog):.1f}  min {min(nlog)} max {max(nlog)}")
    print(f"  control-size histogram (over all logical gates): {dict(sorted(ksz.items()))}")
    from collections import Counter as C2
    print(f"  per-unit max control size: {dict(sorted(C2(maxk).items()))}")
