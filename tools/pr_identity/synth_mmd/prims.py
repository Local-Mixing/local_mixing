"""g57 primitive templates: shortest g57 words for NOT / CNOT / 2-control
mixed Toffoli, extracted from the full w=3 g57 group (= S_8).  These become
relabelable building blocks for constructive (MMD-style) synthesis.

A "logical gate" is (target, ((ctrl_wire, pol), ...)) meaning
    target ^= AND_i (wire_i == pol_i)
i.e. a mixed-polarity multi-control Toffoli; empty controls = NOT.
"""
import itertools
import numpy as np
import g57

def gates(m): return list(itertools.permutations(range(m), 3))

def build_ball(m, radius=12):
    """BFS the g57 group on m wires -> {perm_bytes: shortest word}."""
    ID = np.arange(1 << m, dtype=np.int64)
    gts = {gg: g57.gate_table(gg, m) for gg in gates(m)}
    ball = {ID.tobytes(): ()}
    frontier = {ID.tobytes(): ()}
    for _ in range(radius):
        nf = {}
        for pb, w in frontier.items():
            P = np.frombuffer(pb, dtype=np.int64)
            for gg, tb in gts.items():
                npb = tb[P].tobytes()
                if npb not in ball:
                    ball[npb] = w + (gg,)
                    nf[npb] = w + (gg,)
        frontier = nf
        if not frontier:
            break
    return ball

def logical_table(target, ctrls, m):
    """Permutation table on m wires for target ^= AND_i(wire==pol)."""
    st = np.arange(1 << m, dtype=np.int64)
    fire = np.ones(1 << m, dtype=np.int64)
    for (w, pol) in ctrls:
        bit = (st >> w) & 1
        fire &= bit if pol == 1 else (1 - bit)
    return st ^ (fire << target)

# Cache of templates on canonical small supports.  Each template is a g57 word
# on wires 0..s-1 where wire 0 = target, wires 1.. = controls (in order), and
# any remaining wires are clean-return helpers.
_BALL3 = None
def _ball3():
    global _BALL3
    if _BALL3 is None:
        _BALL3 = build_ball(3, radius=14)
    return _BALL3

def template_not():
    """g57 word (on wires 0,1,2) = NOT(wire0), identity on 1,2."""
    b = _ball3()
    T = logical_table(0, (), 3)
    return list(b[T.tobytes()])

def template_cnot(pol=1):
    """g57 word (wires 0,1,2) = wire0 ^= (wire1==pol), id on wire2."""
    b = _ball3()
    T = logical_table(0, ((1, pol),), 3)
    return list(b[T.tobytes()])

def template_toffoli(pol1=1, pol2=1):
    """g57 word (wires 0,1,2) = wire0 ^= (wire1==pol1)&(wire2==pol2)."""
    b = _ball3()
    T = logical_table(0, ((1, pol1), (2, pol2)), 3)
    return list(b[T.tobytes()])

def selftest():
    b = _ball3()
    print(f"w=3 ball size (should be |S8|=40320): {len(b)}")
    lens = {}
    for name, w in [("NOT", template_not()),
                    ("CNOT+", template_cnot(1)), ("CNOT-", template_cnot(0)),
                    ("TOFF++", template_toffoli(1,1)), ("TOFF+-", template_toffoli(1,0)),
                    ("TOFF-+", template_toffoli(0,1)), ("TOFF--", template_toffoli(0,0))]:
        lens[name] = len(w)
    print("template lengths:", lens)
    # verify each template realizes its logical table
    ok = True
    for (name, tgt, ctrls, w) in [
        ("NOT", 0, (), template_not()),
        ("CNOT+", 0, ((1,1),), template_cnot(1)),
        ("CNOT-", 0, ((1,0),), template_cnot(0)),
        ("TOFF++", 0, ((1,1),(2,1)), template_toffoli(1,1)),
        ("TOFF+-", 0, ((1,1),(2,0)), template_toffoli(1,0)),
        ("TOFF--", 0, ((1,0),(2,0)), template_toffoli(0,0)),
    ]:
        got = g57.word_table(w, 3)
        want = logical_table(tgt, ctrls, 3)
        good = np.array_equal(got, want); ok &= good
        print(f"  verify {name}: {'OK' if good else 'FAIL'}  word={w}")
    print("ALL TEMPLATES VERIFIED:", ok)

if __name__ == "__main__":
    selftest()
