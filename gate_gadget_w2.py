"""
gate_gadget_w2.py — the nonlinear193 gadget restricted to WEIGHT-2 GATES (fan-in <= 2).

The canonical gadget emits gates of fan-in up to 4 (e.g. `o ^= a2_2 & a2_4 & R0 & scr`).
This module decomposes every fan-in->=3 gate into a chain of fan-in-2 Toffolis using a small
pool of decomposition ancillas, via MASK-FIRST accumulation:

    to emit  t ^= c_1 & ... & c_k  (k>=3), fold the MASK controls (scr=B, scr2, R, chaff)
    into the accumulator FIRST and the OPERAND-share controls (a, b, s1) LAST:

        anc0 = mask_a & mask_b            (mask-only product — no operand)
        anc1 = anc0   & operand_1         (operand, but already carrying every mask)
        t   ^= anc1   & operand_2
        (then uncompute anc1, anc0)

Why mask-first is the security-preserving choice: a naive decomposition would expose a
BARE operand-share product (e.g. `a2_2 & a2_4`) as an ancilla trace coordinate; XORed with
the operand's input wires (which are already in the trace) that reconstructs a share's
decode E(a2), reviving the EXACT-LINEAR attack.  Mask-first guarantees every exposed
intermediate still carries the full mask (>= B), so no bare operand monomial ever appears
and the exact/weight-<=2 immunity is preserved.  A decomposition can only ADD trace
coordinates (the target's own flip is bit-identical to the original single gate), so the
only question is whether the added masked intermediates leak — which the harness checks.

To keep the gate count down, a reversible COMMON-SUBEXPRESSION cache shares mask-prefix
products across gates: `R0&scr`, `R1&scr` (level 1) and `scr2&a_i`, `R0&scr&a_i`, ... (level
2) are each built once onto a persistent wire, reused by every gate needing them, and
uncomputed when their underlying mask (scr / scr2) is rebuilt at a group boundary or by
flush().  This is still SUBSET-safe: every cached prefix is a value the plain mask-first
decomposition already computed as an intermediate, so no new trace coordinate appears — it
only reduces redundant rebuilds.  Result (r57): naive 505 -> 291 gates (canonical fan-in-4:
193); the SG: naive 2331 -> 1415 with exact per-sub-gadget mask classification
(set_masks; the naive union-of-all-masks classification costs 1455).  Verified
fan-in<=2, exact/w1/w2 clean.

Correctness is exact and ancillas are restored, so this is a drop-in replacement usable
inside big_gate_gadget / the SG as well.
"""
import numpy as np
import gate_gadget_v2 as G


class Weight2Circuit(G.Circuit):
    """Circuit that auto-decomposes any fan-in>=3 gate into fan-in-<=2 gates (mask-first),
    with a persistent MASK-PREFIX cache that removes redundant rebuilds.

    core_mask : wires holding the primary masks (scr = B, scr2) — folded FIRST.
    aux_mask  : public random mask wires (R block, chaff) — folded next.
    Operand-share wires (everything else) are folded LAST.
    decomp    : >=2 clean fallback ancillas (restored after every decomposition).
    persist   : clean wires that hold cached 2-mask products (e.g. R0&scr) across the whole
                gadget; each is restored to 0 by flush() (call once per gadget_gate).
    temp      : >=1 clean wire for per-gate operand accumulation (restored each gate).

    SAFETY: caching a 2-mask product P = m0&m1 exposes only a MASK-ONLY coordinate (already
    exposed by the uncached decomposition as its first ancilla), and every downstream
    intermediate P&op equals an intermediate the uncached path already produced.  So the
    cache emits a SUBSET of the uncached trace coordinates — it cannot introduce any leak.
    """
    def __init__(self, wire_values, decomp_ancillas, core_mask, aux_mask,
                 persist=(), temp=()):
        super().__init__(wire_values)
        self.decomp = list(decomp_ancillas)
        self.core_mask = set(core_mask)
        self.aux_mask = set(aux_mask)
        self.maxfanin = 0
        self._persist_pool = list(persist)
        self._temp = list(temp)
        self._cache = {}                       # cache key -> (wire, build-controls, maskset)

    def _is_mask(self, w):
        return w in self.core_mask or w in self.aux_mask

    def set_masks(self, core_mask, aux_mask):
        """Re-point the mask classification (e.g. to the CURRENT gadget's masks in a
        chain, where a previous gadget's R/chaff wires have been relabeled into operand
        share blocks).  Must be called with an empty cache (after flush()); mask wires
        are never operand controls and are never written while cached entries depend on
        them, so per-gadget classification is the exact, safe discipline."""
        assert not self._cache, "set_masks with live cache entries"
        self.core_mask = set(core_mask)
        self.aux_mask = set(aux_mask)

    def _get_prod(self, key, factors, maskset):
        """Return a persistent wire holding product(factors) (2 controls), building &
        caching it on first request.  `key` is the cache key; `maskset` is the set of mask
        wires the product depends on (for invalidation).  Returns None if pool exhausted."""
        ent = self._cache.get(key)
        if ent is not None:
            return ent[0]
        if not self._persist_pool:
            return None
        wire = self._persist_pool.pop()
        super().gate(wire, list(factors), 0)                   # wire = factors[0] & factors[1]
        self._cache[key] = (wire, list(factors), set(maskset))
        return wire

    def _invalidate(self, w):
        """A mask wire w is about to change: uncompute every cache entry depending on it,
        in REVERSE build order (so a level-2 product is undone before the level-1 product
        it was built from)."""
        for key in [k for k, (_, _, ms) in reversed(list(self._cache.items())) if w in ms]:
            wire, factors, _ = self._cache.pop(key)
            super().gate(wire, list(factors), 0)               # re-emit product -> zero
            self._persist_pool.append(wire)

    def gate(self, target, controls, comp=0):
        if (target in self.core_mask or target in self.aux_mask) and self._cache:
            self._invalidate(target)
        self.maxfanin = max(self.maxfanin, len(controls))
        if len(controls) <= 2:
            return super().gate(target, controls, comp)
        core = [c for c in controls if c[0] in self.core_mask]
        aux = [c for c in controls if c[0] in self.aux_mask]
        op = [c for c in controls if not self._is_mask(c[0])]
        mask = core + aux
        mset = frozenset(w for (w, _) in mask)

        # ---- level 1: the mask-product P1 ----
        if len(mask) == 1:
            P1 = mask[0]                                        # single mask wire, no cache
        elif len(mask) == 2 and self._persist_pool:
            w1 = self._get_prod(("m", frozenset(mask)), mask, mset)
            P1 = (w1, 1) if w1 is not None else None
        else:
            P1 = None                                          # >=3 masks (rare) -> fallback

        if P1 is not None:
            if len(op) == 0:
                super().gate(target, [P1], comp)
                return
            if len(op) == 1:
                super().gate(target, [P1, op[0]], comp)
                return
            # ---- level 2: cache P2 = P1 & op[0], then hit target with op[1] ----
            if len(op) == 2:
                w2 = self._get_prod(("mo", mset, op[0]), [P1, op[0]], mset)
                if w2 is not None:
                    super().gate(target, [(w2, 1), op[1]], comp)
                    return
                if self._temp:                                 # pool full: transient temp
                    t = self._temp[0]
                    super().gate(t, [P1, op[0]], 0)
                    super().gate(target, [(t, 1), op[1]], comp)
                    super().gate(t, [P1, op[0]], 0)
                    return

        # ---- fallback: mask-first decomposition with fallback ancillas ----
        order = mask + op
        assert order[0][0] in self.core_mask or order[0][0] in self.aux_mask
        k = len(order); need = k - 2
        assert need <= len(self.decomp), f"need {need} decomp ancillas, have {len(self.decomp)}"
        anc = self.decomp
        super().gate(anc[0], [order[0], order[1]], 0)
        for i in range(1, need):
            super().gate(anc[i], [(anc[i - 1], 1), order[i + 1]], 0)
        super().gate(target, [(anc[need - 1], 1), order[k - 1]], comp)
        for i in range(need - 1, 0, -1):
            super().gate(anc[i], [(anc[i - 1], 1), order[i + 1]], 0)
        super().gate(anc[0], [order[0], order[1]], 0)

    def flush(self):
        """Uncompute all cached products (reverse build order); call once per gadget_gate."""
        for key in reversed(list(self._cache.keys())):
            wire, factors, _ = self._cache.pop(key)
            super().gate(wire, list(factors), 0)
            self._persist_pool.append(wire)


# ------------------------------------------------------------------ single-gadget runner
def run_gate(n=20000, seed=0, vtype="r57", a_single=False, b_single=False, dirty=False):
    """Weight-2 version of G.run_gate: identical logical gate, fan-in <= 2 everywhere."""
    na = 0 if vtype == "copy" else (1 if a_single else 2)
    nb = 1 if b_single or vtype == "copy" else 2
    ngroups = 2
    S1B = tuple(range(0, 5)); S2B = tuple(range(5, 10)); RB = tuple(range(10, 15))
    p = 15
    a_blocks = tuple(tuple(range(p + 5 * i, p + 5 * i + 5)) for i in range(na)); p += 5 * na
    b_blocks = tuple(tuple(range(p + 5 * i, p + 5 * i + 5)) for i in range(nb)); p += 5 * nb
    OUTB = tuple(range(p, p + 3)); SCR = p + 3; p += 4
    if dirty:
        scr2 = tuple(range(p, p + ngroups)); p += ngroups
    else:
        scr2 = p; p += 1
    chaff = (p, p + 1, p + 2, p + 3); p += 4
    DEC = (p, p + 1); p += 2                                # fallback decomposition ancillas
    PERSIST = tuple(range(p, p + 24)); p += 24                     # cached mask-prefix wires
    TEMP = (p,); p += 1                                     # per-gate operand accumulator
    NW = p

    rng = np.random.default_rng(seed)
    wires = [rng.integers(0, 2, n).astype(np.uint8) for _ in range(NW)]
    for w in OUTB + (SCR,) + (tuple(scr2) if dirty else (scr2,)) + DEC + PERSIST + TEMP:
        wires[w] = np.zeros(n, np.uint8)

    core_mask = {SCR} | (set(scr2) if dirty else {scr2})
    aux_mask = set(RB) | set(chaff)
    circ = Weight2Circuit(wires, DEC, core_mask, aux_mask, persist=PERSIST, temp=TEMP)

    def dec(blocks):
        v = np.zeros(n, np.uint8)
        for blk in blocks:
            v = (v ^ G.E([circ.init[w] for w in blk])).astype(np.uint8)
        return v
    a = dec(a_blocks) if a_blocks else np.zeros(n, np.uint8)
    b = dec(b_blocks)
    c_in = (G.E([circ.init[w] for w in S1B]) ^ G.E([circ.init[w] for w in S2B])).astype(np.uint8)
    rho = G.E([circ.init[w] for w in RB])
    u = (G.E([circ.init[w] for w in S1B]) ^ rho).astype(np.uint8)
    V = {"r57": (1 ^ b ^ (a & b)), "nab": ((1 ^ a) & b), "and": (a & b),
         "copy": b}[vtype].astype(np.uint8)
    c_out = (c_in ^ V).astype(np.uint8)

    G.gadget_gate(circ, a_blocks, b_blocks, S1B, S2B, RB, OUTB, SCR, scr2, chaff, vtype=vtype)
    circ.flush()                                           # restore cached mask-prefix wires

    share1p = [circ.s[w] for w in RB]
    share2p = [circ.s[S2B[0]], circ.s[S2B[1]], circ.s[OUTB[0]], circ.s[OUTB[1]], circ.s[OUTB[2]]]
    c_out_actual = (G.E(share1p) ^ G.E(share2p)).astype(np.uint8)
    scr_ok = bool(np.array_equal(circ.s[SCR], np.zeros(n, np.uint8)))
    if not dirty:
        scr_ok &= bool(np.array_equal(circ.s[scr2], np.zeros(n, np.uint8)))
    dec_ok = all(bool(np.array_equal(circ.s[w], np.zeros(n, np.uint8)))
                 for w in DEC + PERSIST + TEMP)
    expw = set(OUTB) | {SCR} | set(DEC) | set(PERSIST) | set(TEMP)
    if vtype != "copy":
        expw |= set(scr2) if dirty else {scr2}
    info = dict(a=a, b=b, c_in=c_in, gate_ab=V, c_out=c_out, rho=rho, u=u,
                correct=bool(np.array_equal(c_out_actual, c_out)),
                scratch_restored=scr_ok and dec_ok, maxfanin=circ.maxfanin,
                n_gates=len(circ.flips), NW=NW, written_expected=expw)
    return circ, info


# ------------------------------------------------------------------ two-gadget chain builder
def build_chain(n, seed):
    """Two weight-2 nonlinear gadgets chained (output of gate 1 = input of gate 2),
    in the gadget_tester submission format."""
    rng = np.random.default_rng(seed)
    p = 0
    P1, P2 = {}, {}
    for k in range(4):
        P1[k] = list(range(p, p + 5)); P2[k] = list(range(p + 5, p + 10)); p += 10
    extras = []
    for _ in range(2):
        R = tuple(range(p, p + 5)); out = tuple(range(p + 5, p + 8))
        scr = p + 8; scr2 = p + 9; p += 10
        chaff = tuple(range(p, p + 4)); p += 4
        extras.append((R, out, scr, scr2, chaff))
    DEC = (p, p + 1); p += 2
    PERSIST = tuple(range(p, p + 24)); p += 24
    TEMP = (p,); p += 1
    NW = p

    wires = [rng.integers(0, 2, n).astype(np.uint8) for _ in range(NW)]
    for (R, out, scr, scr2, chaff) in extras:
        for w in out + (scr, scr2):
            wires[w] = np.zeros(n, np.uint8)
    for w in DEC + PERSIST + TEMP:
        wires[w] = np.zeros(n, np.uint8)

    circ = Weight2Circuit(wires, DEC, set(), set(), persist=PERSIST, temp=TEMP)

    def val(k):
        return (G.E([circ.s[w] for w in P1[k]]) ^ G.E([circ.s[w] for w in P2[k]])).astype(np.uint8)

    targets, expw = {}, set()
    for gi, (tw, aw, bw) in enumerate(((2, 0, 1), (3, 2, 0))):
        av, bv, ci = val(aw), val(bw), val(tw)
        V = (1 ^ bv ^ (av & bv)).astype(np.uint8)
        co = (ci ^ V).astype(np.uint8)
        lbl = "g2:a(=g1.out)" if gi == 1 else f"g{gi+1}:a"
        targets.update({lbl: av, f"g{gi+1}:b": bv, f"g{gi+1}:gate": V,
                        f"g{gi+1}:c_in": ci, f"g{gi+1}:c_out": co})
        R, out, scr, scr2, chaff = extras[gi]
        expw |= set(out) | {scr, scr2}
        # exact per-gadget mask classification: gadget 1's R/out wires have been
        # relabeled into operand share blocks and must not classify as masks
        circ.set_masks({scr, scr2}, set(R) | set(chaff))
        G.gadget_gate(circ, (tuple(P1[aw]), tuple(P2[aw])), (tuple(P1[bw]), tuple(P2[bw])),
                      tuple(P1[tw]), tuple(P2[tw]), R, out, scr, scr2, chaff, vtype="r57")
        circ.flush()                                       # restore cached prefixes per gadget
        P1[tw] = list(R); P2[tw] = [P2[tw][0], P2[tw][1]] + list(out)
        assert np.array_equal(val(tw), co)
    expw |= set(DEC) | set(PERSIST) | set(TEMP)
    return dict(name="nonlinear193-w2 — two weight-2 gadgets chained",
                circ=circ, targets=targets, written=expw)


if __name__ == "__main__":
    print("weight-2 gadget — correctness / fan-in / gate-count")
    for vt in ("r57", "nab", "and", "copy"):
        circ, info = run_gate(12000, seed=1, vtype=vt,
                              a_single=(vt == "copy"), b_single=(vt == "copy"))
        cg, ig = G.run_gate(12000, seed=1, vtype=vt,
                            a_single=(vt == "copy"), b_single=(vt == "copy"))
        print(f"  {vt:4s}: w2 gates={info['n_gates']:3d} (canonical {ig['n_gates']:3d}) "
              f"maxfanin={info['maxfanin']} correct={info['correct']} "
              f"restored={info['scratch_restored']}")
