"""Constructive g57 unit synthesizer (w=6..8) via MMD + Barenco MCT + templates.

synth_unit(unit_word, w, n_anc) -> (g57_word_on_canvas, stats)
  canvas = w source wires (0..w-1) + n_anc dirty ancilla wires (w..w+n_anc-1).
  The returned g57 word equals the unit permutation on the source wires and the
  identity on the ancillas, for ALL 2^(w+n_anc) states (verified).
"""
import numpy as np
import g57
import prims
from mmd import mmd_synth
from mct import decompose

# cached templates (canonical wires 0=target,1..=controls, extras=helpers)
_T_NOT = prims.template_not()          # wires 0,1,2 : NOT(0)
_T_CX  = prims.template_cnot(1)        # wires 0,1,2 : 0 ^= 1
_T_CCX = prims.template_toffoli(1, 1)  # wires 0,1,2 : 0 ^= 1&2

def _relabel(word, mp):
    return [(mp[a], mp[x], mp[y]) for (a, x, y) in word]

def _compile_base(gt, canvas_wires, active):
    """One base gate ('X'/'CX'/'CCX') -> g57 gates.  active = wires this gate
    logically touches; helpers drawn from canvas_wires\\active (ancilla-first)."""
    helpers = [wv for wv in canvas_wires if wv not in active]
    if gt[0] == 'CCX':
        _, c1, c2, t = gt
        return _relabel(_T_CCX, {0: t, 1: c1, 2: c2})
    if gt[0] == 'CX':
        _, c, t = gt
        h = helpers[0]
        return _relabel(_T_CX, {0: t, 1: c, 2: h})
    # 'X'
    _, t = gt
    h1, h2 = helpers[0], helpers[1]
    return _relabel(_T_NOT, {0: t, 1: h1, 2: h2})

def synth_unit(unit_word, w, n_anc=3, verify=True):
    canvas = w + n_anc
    src = list(range(w))
    anc = list(range(w, canvas))
    # ancilla-first helper ordering so CX/NOT helpers avoid source wires
    canvas_wires_helperpref = anc + src
    P = g57.word_table(unit_word, w)               # target perm on source wires
    logical = mmd_synth(P.copy())                  # positive-control MCTs
    maxk = max((len(c) for (_, c) in logical), default=0)
    g57word = []
    for (target, ctrls) in logical:
        ctrls = list(ctrls)
        spare = [a for a in anc if a not in ctrls and a != target]
        base = []
        decompose(ctrls, target, spare, base)      # -> CCX/CX/X on canvas
        for bg in base:
            active = set(bg[1:])
            # helper order: ancilla-first, excluding active
            cw = [wv for wv in canvas_wires_helperpref]
            g57word += _compile_base(bg, cw, active)
    stats = {"logical": len(logical), "maxk": maxk, "g57": len(g57word),
             "n_anc": n_anc}
    if verify:
        got = g57.word_table(g57word, canvas)
        st = np.arange(1 << canvas, dtype=np.int64)
        low = (1 << w) - 1
        want = (st & ~low) | P[st & low]
        stats["ok"] = bool(np.array_equal(got, want))
    return g57word, stats

if __name__ == "__main__":
    import random, time
    from geodesic import make_unit
    for w in (4, 5, 6):
        rng = random.Random(100 + w)
        n_anc = max(0, w - 3)            # enough dirty ancilla for k=w-1
        oks = 0; g57s = []; logs = []; t0 = time.time()
        N = 20
        for _ in range(N):
            unit = make_unit(w, w, rng)
            word, stt = synth_unit(unit, w, n_anc=n_anc)
            oks += stt["ok"]; g57s.append(stt["g57"]); logs.append(stt["logical"])
        dt = (time.time() - t0) / N * 1000
        print(f"w={w} n_anc={n_anc}: verified {oks}/{N}  "
              f"mean g57 {np.mean(g57s):.0f}  mean logical {np.mean(logs):.0f}  "
              f"{dt:.0f} ms/unit")
