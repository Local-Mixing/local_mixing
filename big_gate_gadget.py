"""
big_gate_gadget.py — the SG-level gate ("the behemoth") built from gate_gadget gadgets.

One SG-level logical wire W is 2-shared at the SG level, W = r ⊕ s, and each of r, s is a
gadget-level wire (2-shared nonlinear encoding, 10 physical wires) — 20 physical wires per
logical wire.  One SG-level gate  W_t ^= W_a OR ~W_b  is SIX gadget calls (the modified
secret-share construction  r_t ^= f⊕g,  s_t ^= g,  with f expanded over the shares) and
NOTHING else — no glue gates; the internal re-share of every sub-gadget chains the writes.

    G1: r_t ^= 1 ⊕ (~r_a)·r_b     [vtype r57]      G4: r_t ^= s_a·s_b   [vtype and]
    G2: r_t ^= (~r_a)·s_b         [vtype nab]      G5: r_t ^= g         [vtype copy]
    G3: r_t ^= s_a·r_b            [vtype and]      G6: s_t ^= g         [vtype copy]

This module uses the MAX-SIZE (unconditional) configuration: every operand 2-shared (dd),
including the fresh mask g — 193+190+193+193+85+85 = 939 gates per SG-level gate.

Interface: SG-share wires live in a dict  shares[name] = [P1_block, P2_block]  (each a
list of 5 wire indices).  sg_gate() mutates the target's entries per the gadget-output
relabeling.  Each sub-gadget consumes one `extras` tuple (R5, out3, scr, scr2, chaff4) of
fresh-random / clean wires supplied by the caller (strict upfront layout).
"""
import numpy as np
import gate_gadget as G

# (tag, vtype, a-share suffix, b-share suffix, target-share suffix); None = no a operand.
SG_RECIPE = [
    ("G1", "r57", "r{a}", "r{b}", "r{t}"),
    ("G2", "nab", "r{a}", "s{b}", "r{t}"),
    ("G3", "and", "s{a}", "r{b}", "r{t}"),
    ("G4", "and", "s{a}", "s{b}", "r{t}"),
    ("G5", "copy", None, "{g}", "r{t}"),
    ("G6", "copy", None, "{g}", "s{t}"),
]


def share_val(circ, shares, nm, init=False):
    src = circ.init if init else circ.s
    P1, P2 = shares[nm]
    return (G.E([src[w] for w in P1]) ^ G.E([src[w] for w in P2])).astype(np.uint8)


def W_val(circ, shares, k):
    return (share_val(circ, shares, f"r{k}") ^ share_val(circ, shares, f"s{k}")).astype(np.uint8)


def sg_gate(circ, shares, a, b, t, g_name, extras6, checks=None, tag=""):
    """Emit one SG-level gate  W_t ^= W_a OR ~W_b  (six max-size gadgets).

    shares  : dict name -> [P1_block(list of 5), P2_block(list of 5)]; must contain
              r{a}, s{a}, r{b}, s{b}, r{t}, s{t}, and g_name.  Target entries mutated.
    extras6 : six tuples (R5, out3, scr, scr2, chaff4) of fresh wires (R, chaff random;
              out/scr/scr2 clean).
    checks  : optional dict; per-sub-gadget c_in/c_out values are recorded into it.
    Returns the expected written-wire set of this SG-gate.
    """
    expw = set()
    for gi, (gtag, vt, a_pat, b_pat, t_pat) in enumerate(SG_RECIPE):
        fmt = dict(a=a, b=b, t=t, g=g_name)
        b_nm = b_pat.format(**fmt)
        t_nm = t_pat.format(**fmt)
        a_nm = a_pat.format(**fmt) if a_pat else None

        av = share_val(circ, shares, a_nm) if a_nm else None
        bv = share_val(circ, shares, b_nm)
        V = {"r57": lambda: (1 ^ bv ^ (av & bv)),
             "nab": lambda: ((1 ^ av) & bv),
             "and": lambda: (av & bv),
             "copy": lambda: bv}[vt]().astype(np.uint8)

        R, out, scr, scr2, chaff = extras6[gi]
        expw.update(out); expw.add(scr)
        if vt != "copy":
            expw.add(scr2)
        c_in = share_val(circ, shares, t_nm)
        c_out = (c_in ^ V).astype(np.uint8)
        if checks is not None:
            checks[f"{tag}{gtag}:c_in"] = c_in
            checks[f"{tag}{gtag}:c_out"] = c_out

        if hasattr(circ, "mark"):
            circ.mark(f"{tag}{gtag} ({vt}): {t_nm} ^= {b_nm}" + (f" * {a_nm}" if a_nm else ""))
        a_blocks = () if a_nm is None else (tuple(shares[a_nm][0]), tuple(shares[a_nm][1]))
        b_blocks = (tuple(shares[b_nm][0]), tuple(shares[b_nm][1]))
        P1t, P2t = shares[t_nm]
        G.gadget_gate(circ, a_blocks, b_blocks, tuple(P1t), tuple(P2t),
                      R, out, scr, scr2, chaff, vtype=vt)
        shares[t_nm] = [list(R), [P2t[0], P2t[1]] + list(out)]     # the zero-gate relabel
        assert np.array_equal(share_val(circ, shares, t_nm), c_out), f"{tag}{gtag} incorrect"
    return expw


if __name__ == "__main__":
    # smoke: one SG-level gate on fresh shares
    rng = np.random.default_rng(0); n = 8000
    p = 0
    def block():
        global p
        b = list(range(p, p + 5)); p += 5
        return b
    shares = {nm: [block(), block()] for nm in
              ("r0", "s0", "r1", "s1", "r2", "s2", "gA")}
    extras = []
    for _ in range(6):
        R = tuple(block()); out = tuple(range(p, p + 3)); scr = p + 3; scr2 = p + 4
        p += 5; chaff = tuple(range(p, p + 4)); p += 4
        extras.append((R, out, scr, scr2, chaff))
    wires = [rng.integers(0, 2, n).astype(np.uint8) for _ in range(p)]
    for (R, out, scr, scr2, chaff) in extras:
        for w in out + (scr, scr2):
            wires[w] = np.zeros(n, np.uint8)
    circ = G.Circuit(wires)
    Wa, Wb, Wt = W_val(circ, shares, 0), W_val(circ, shares, 1), W_val(circ, shares, 2)
    f = (1 ^ Wb ^ (Wa & Wb)).astype(np.uint8)
    sg_gate(circ, shares, 0, 1, 2, "gA", extras)
    ok = np.array_equal(W_val(circ, shares, 2), (Wt ^ f).astype(np.uint8))
    print(f"big gate (SG of gadgets): gates={len(circ.flips)} correct={ok}")
