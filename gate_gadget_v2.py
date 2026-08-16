"""
gate_gadget.py
==============

CANONICAL gadgetized reversible gate — the folded, self-contained, optimized design.

One gadget realizes one gate of the original circuit,

        c_out = c_in XOR V(a, b),

over nonlinear encodings, such that the full per-gate trace (every input wire, plus every
gate's flip bit and the target's new value) admits

  * NO exact affine leak (GF(2) Gaussian elimination) of a, b, V(a,b), c_in, or c_out;
  * NO weight-1 correlation: every single trace coordinate has zero covariance with them;
  * NO weight-2 correlation: every pair of coordinates under EVERY 2-bit Boolean function
    (all 16, incl. the asymmetric andnot family) has zero covariance with them.

Verified empirically to sampling precision (~0.005 at n=400k for weight-1, ~0.02 at n=48k
for weight-2), single-gadget AND composed: 2-gate chains (output feeding an input), 3-gate
strict chains, and a full 6-gadget SG spliced with no glue (see README).

--------------------------------------------------------------------------------
REPRESENTATION
--------------------------------------------------------------------------------
Decode (5 wires, nonlinear, 2-resilient — minimal: 2 uniform pads + balanced maj):

        E(x0,x1,x2,x3,x4) = x0 ^ x1 ^ maj(x2,x3,x4)

  * value ("target") wire :  ALWAYS 2-shared,  c = E(S1) ^ E(S2).
  * control wire          :  2-shared, i.e. read from both blocks (a = E(A1)^E(A2)) —
                             UNLESS the wire is read-only for its whole life in the
                             surrounding circuit, in which case it may stay a single
                             E-encoding (its encoding is never used as a flip carrier,
                             which is the only thing that exposes a single encoding).

--------------------------------------------------------------------------------
THE GADGET (one carrier flip + one relabeling)
--------------------------------------------------------------------------------
Borrow a block R of 5 fresh random wires.  Flip S2's majority, permutation-form, by

        W  =  u ^ V,     u = E(S1) ^ E(R)        (the re-share, folded in)
                         V = the gate's cleartext output over the control encodings

landing the permuted majority on 3 fresh out-wires.  Then, with ZERO gates:

        new share 1 := R                     new share 2 := (S2_0, S2_1, out0, out1, out2)

Correctness: E(share2') = E(S2)^u^V, E(share1') = E(R), so the value gains exactly V.
Because u is fresh-uniform and independent, the flip key W is uniform and independent of
V — the gate output is never a flip amount anywhere, killing the V·B weight-2 witnesses.
Because each gadget has ONE carrier (S2's majority) and share-1 generations (S1, then R)
are only ever read as masked control monomials, composition adds one never-exposed
majority unknown per gate and the exact attacker's system never closes.

--------------------------------------------------------------------------------
EMISSION (the ~190 gates, and why each discipline rule exists)
--------------------------------------------------------------------------------
V's a·b term is computed by a MASKED TOFFOLI CASCADE per output bit i (not the flat
10x10 monomial expansion):  scratch2 <- (b ^ R0 ^ R1)·B_i(s)  then a's monomials multiply
against scratch2, with two correction runs (a-mons ∧ R0 ∧ B_i, a-mons ∧ R1 ∧ B_i)
canceling the mask spill.  TWO masks because one is strippable by a weight-2 mirror pair;
R0, R1 double as masks because u contains them linearly, canceling two emissions outright.

Order discipline (all costs ~0 gates; each rule closes a measured leak class):
  * scratch2 build/unbuild: pads of b at the extremes, masks strictly interior, the
    inter-mask segment balanced (for a single-encoded b: the full maj-triple), unbuild in
    EXACT REVERSE — every contiguous window either keeps a uniform mask or leaves a fresh
    pad of b in its complement.
  * main run: single-role chunks (cascade / corr-R0 / corr-R1) of <=5 gates; every chunk
    boundary carries a PROVABLY BALANCED gate (linear-u monomials and chaff pairs on 2
    extra borrowed wires, arranged so every boundary interval keeps one uncanceled) —
    maj-pair u-monomials are NOT balanced and do not count.
  * shared-B groups (B0 == B1): rounds 0,1 share one scratch/scratch2 build and emit
    their runs in the IDENTICAL order, collapsing every cross-run pair into the
    single-run window family.

Permutation constraints (searched over all valid maj-preserving/flipping pairs):
  every B_i non-constant;  NO XOR-subset of the distinct B_i constant (else b lands in
  the exact span);  E[B_i] balanced (kills a second-order ~0.05 residual);  B0 == B1
  (enables sharing);  minimal ANF overhead.  The shipped pair costs 13.

--------------------------------------------------------------------------------
GATE TYPES (compile-time ANF change only) and gate counts (clean / dirty)
--------------------------------------------------------------------------------
  vtype='r57'  : V = 1 ^ b ^ ab   (c ^= a OR NOT b)    193/169 dd,  128/114 ss
  vtype='nab'  : V = b ^ ab       (c ^= (NOT a) AND b)
  vtype='and'  : V = ab           (c ^= a AND b; u emitted fully — no s2emit)
  vtype='copy' : V = op           (c ^= op; 1-input: no cascade at all, ~70 gates)
  (dd = both controls 2-shared; ss = both single-encoded; dirty=True skips scratch2
   uncompute, leaving masked garbage — trace-equivalent, saves ~11%.)

Gate model: one generalized-Toffoli per gate: target ^= comp ^ AND(literals); comp=0
everywhere here; constant terms use comp=0 with the mask/scratch factor as control.
"""
import numpy as np
import random as _random
from collections import Counter


# ---------------------------------------------------------------- primitives
def maj(a, b, c):
    """Majority of three bit-vectors: ab ^ bc ^ ac."""
    return (a & b) ^ (b & c) ^ (a & c)


def E(wire_list):
    """Decode E(x)=x0^x1^maj(x2,x3,x4) from a list of 5 bit-vectors."""
    return (wire_list[0] ^ wire_list[1] ^ maj(wire_list[2], wire_list[3], wire_list[4])).astype(np.uint8)


def _bit(v, i):
    return (v >> i) & 1


def _anf3(truth):
    """ANF monomial masks of a 3-variable boolean function (8-entry truth table)."""
    a = list(truth)
    for b in range(3):
        for v in range(8):
            if (v >> b) & 1:
                a[v] ^= a[v ^ (1 << b)]
    return [v for v in range(8) if a[v]]


def _decompose_perm(U0, U1):
    """A_mons[i] = ANF(bit_i(U0(s))) — the V-independent part;
       B_mons[i] = ANF(bit_i(U1(s)) ^ bit_i(U0(s))) — the flip-key multiplier."""
    A_mons, B_mons = [], []
    for i in range(3):
        A_mons.append(_anf3([_bit(U0[v], i) for v in range(8)]))
        B_mons.append(_anf3([_bit(U1[v], i) ^ _bit(U0[v], i) for v in range(8)]))
    return A_mons, B_mons


# The shipped permutation pair (U0 majority-preserving, U1 majority-flipping):
# B0 == B1 (shared-B), all B_i non-constant, no XOR-subset of {B0,B2} constant,
# E[B_i] = 1/2 (balanced), ANF overhead 13 — from the constrained exhaustive search.
SB_U0 = (0, 1, 2, 3, 4, 5, 6, 7)
SB_U1 = (7, 5, 6, 0, 3, 1, 2, 4)

# legacy constants (the ORIGINAL 119-gate gadget's permutation; kept for the development
# modules gate_gadget_v2/v3/v4 and history — not used by the canonical gadget)
CLEAN_U0 = (0, 2, 1, 5, 4, 6, 7, 3)
CLEAN_U1 = (3, 6, 7, 4, 5, 1, 2, 0)


# ---------------------------------------------------------------- circuit recorder
class Circuit:
    """Reversible-circuit simulator recording the full per-gate trace (what the threat
    model's adversary sees: every gate's flip bit and the target's new value)."""

    def __init__(self, wire_values):
        self.s = [w.copy() for w in wire_values]
        self.init = [w.copy() for w in wire_values]
        self.n = len(wire_values[0])
        self.flips = []
        self.newvals = []
        self.gate_log = []
        self.marks = []          # [(gate_index, section_label)] — spec/SVG export only

    def gate(self, target, controls, comp=0):
        prod = np.ones(self.n, np.uint8)
        for (w, pol) in controls:
            prod = prod & (self.s[w] if pol == 1 else (1 ^ self.s[w]))
        flip = (comp ^ prod).astype(np.uint8)
        self.s[target] = (self.s[target] ^ flip).astype(np.uint8)
        self.flips.append(flip.copy())
        self.newvals.append(self.s[target].copy())
        self.gate_log.append((target, comp, list(controls)))

    def mark(self, label):
        """Section boundary marker for the readable wiring export (no gates emitted)."""
        self.marks.append((len(self.gate_log), label))

    def _emit_conjunctions(self, target, monomials, var_wires, extra_controls=()):
        """Legacy helper (used by the development-history modules)."""
        extra = list(extra_controls)
        for mask in monomials:
            controls = [(var_wires[k], 1) for k in range(len(var_wires)) if (mask >> k) & 1] + extra
            if not controls:
                self.gate(target, [], comp=0)
            else:
                self.gate(target, controls, comp=0)


# ---------------------------------------------------------------- monomial algebra
def _E_monos(off):
    """Monomial masks of E for a block at wire-offset `off` in the gadget's wire list."""
    return [1 << (off + 0), 1 << (off + 1),
            (1 << (off + 2)) | (1 << (off + 3)),
            (1 << (off + 3)) | (1 << (off + 4)),
            (1 << (off + 2)) | (1 << (off + 4))]


def _xor_monos(*lists):
    c = Counter()
    for L in lists:
        c.update(L)
    return sorted(m for m, k in c.items() if k % 2 == 1)


def _lits(mask, wires):
    return [(wires[k], 1) for k in range(len(wires)) if (mask >> k) & 1]


# ---------------------------------------------------------------- the gadget
def gadget_gate(circ, a_blocks, b_blocks, S1, S2, R, out3, scratch, scratch2, chaff,
                U0=SB_U0, U1=SB_U1, sharedB=True, dirty=False, vtype="r57"):
    """Emit ONE folded homomorphic gate onto `circ`.

    a_blocks, b_blocks : tuple of 1 block (single-encoded, read-only-for-life wire) or
                         2 blocks (2-shared wire); each block is 5 wire indices.
                         For vtype='copy' the single operand is b_blocks; a_blocks=().
    S1, S2             : the target's two share blocks (5 wires each; read-only).
    R                  : 5 FRESH random borrow wires (read-only).  R[0], R[1] double as
                         the cascade masks.
    out3               : 3 clean ancillas -> the permuted majority (new share-2 majority).
    scratch            : 1 clean ancilla (holds B_i(s); restored).
    scratch2           : clean mode: 1 clean ancilla (restored); dirty mode: tuple with
                         one wire per build-group, left holding the masked garbage.
    chaff              : 4 fresh random borrow wires.  chaff[0:2] = balanced boundary
                         gates (all types).  chaff[2:4] = DEDICATED cascade masks for
                         vtype='and' — 'and' emits u's R0,R1-linear gates (no s2emit to
                         cancel them), and reusing R0,R1 as masks would let those bare
                         mask·B coordinates echo against the build's mask·B2 across
                         shared-B groups (a measured 0.118 weight-2 leak).  r57/nab keep
                         R0,R1 as masks, where the cancellation removes the bare gates.

    After the call:  share1' = R,  share2' = (S2[0], S2[1], *out3),
                     value' = E(share1') ^ E(share2') = c_in ^ V(a, b).
    """
    A_mons, B_mons = _decompose_perm(U0, U1)
    if sharedB:
        assert B_mons[0] == B_mons[1], "sharedB requires B0 == B1 as functions"
    s = (S2[2], S2[3], S2[4])
    a_w = [w for blk in a_blocks for w in blk]
    b_w = [w for blk in b_blocks for w in blk]
    wiresN = list(S1) + list(R) + a_w + b_w
    a_off = 10; b_off = 10 + len(a_w)
    a_mons = _xor_monos(*[_E_monos(a_off + 5 * i) for i in range(len(a_blocks))])
    b_mons = _xor_monos(*[_E_monos(b_off + 5 * i) for i in range(len(b_blocks))])
    u_all = _xor_monos(_E_monos(0), _E_monos(5))
    u_run = [m for m in u_all if m not in (1 << 5, 1 << 6)]       # R0,R1 linears cancel

    def pads_rest(mons, rr):
        p = [m for m in mons if bin(m).count("1") == 1]
        r = [m for m in mons if bin(m).count("1") > 1]
        rr.shuffle(p); rr.shuffle(r)
        return p, r

    def spread(mons, rr):
        """Pads at spread positions: every chunk-sized window misses at least one pad."""
        p, r = pads_rest(mons, rr)
        pos = (0, 3, 6, 9) if len(mons) == 10 else (0, 4)
        order = [None] * len(mons)
        for k, q in enumerate(pos):
            order[q] = p[k]
        ri = iter(r)
        for j in range(len(mons)):
            if order[j] is None:
                order[j] = next(ri)
        return order

    groups = [((0, 1), B_mons[0]), ((2,), B_mons[2])] if sharedB \
        else [((i,), B_mons[i]) for i in range(3)]
    s2_list = list(scratch2) if isinstance(scratch2, (tuple, list)) else [scratch2] * len(groups)

    for gidx, (idxs, Bm) in enumerate(groups):
        rr = _random.Random(4000 + gidx)
        s2 = s2_list[gidx]

        circ.mark(f"build B (group {gidx})")
        for m in Bm:                                              # scratch = B(s)
            circ.gate(scratch, [(s[k], 1) for k in range(3) if (m >> k) & 1], comp=0)

        two_in = vtype in ("r57", "nab", "and")

        if not two_in:
            # ------------- 'copy': V = op — no cascade, no scratch2 -------------
            op_g = [_lits(m, wiresN) + [(scratch, 1)] for m in spread(b_mons, rr)]
            h2 = 3 if len(b_mons) == 5 else 5
            u_g = [_lits(m, wiresN) + [(scratch, 1)] for m in spread(u_all, rr)]
            chunks = [op_g[0:h2], op_g[h2:], u_g[0:5], u_g[5:]]
            rr.shuffle(chunks)
            cA = [(chaff[0], 1), (scratch, 1)]; cB = [(chaff[1], 1), (scratch, 1)]
            run = []
            for ci, ch in enumerate(chunks):
                run.extend(ch)
                if ci < len(chunks) - 1:
                    run.append([cA, cB, cA][ci])                  # balanced boundaries
            run.append(cB)                                        # even chaff counts
            for i in idxs:
                circ.mark(f"A{i} + run -> o{i} (copy)")
                for m in A_mons[i]:
                    circ.gate(out3[i], [(s[k], 1) for k in range(3) if (m >> k) & 1], comp=0)
                for ctl in run:
                    circ.gate(out3[i], ctl, comp=0)
            circ.mark(f"unbuild B (group {gidx})")
            for m in Bm:
                circ.gate(scratch, [(s[k], 1) for k in range(3) if (m >> k) & 1], comp=0)
            continue

        # ---- build scratch2 = (b ^ mask0 ^ mask1)·B — window-hardened order ----
        circ.mark(f"build scr2 = (b^m0^m1)*B (group {gidx})")
        masks = (R[0], R[1]) if vtype in ("r57", "nab") else (chaff[2], chaff[3])
        bp, bnp = pads_rest(b_mons, rr)
        if len(b_mons) == 10:
            seq = ([bp[0]] + bnp[0:2] + ["M0"] + [bnp[2], bp[1], bnp[3]] + ["M1"]
                   + bnp[4:6] + [bp[2], bp[3]])
        else:
            # 2 pads only: inter-mask segment = the full maj triple (balanced as a set)
            seq = [bp[0], "M0", bnp[0], bnp[1], bnp[2], "M1", bp[1]]
        build = []
        for m in seq:
            if m == "M0":
                build.append([(masks[0], 1), (scratch, 1)])
            elif m == "M1":
                build.append([(masks[1], 1), (scratch, 1)])
            else:
                build.append(_lits(m, wiresN) + [(scratch, 1)])
        for ctl in build:
            circ.gate(s2, ctl, comp=0)

        # ---- main run: single-role chunks, balanced boundaries, ONE order per group ----
        casc = [_lits(m, wiresN) + [(s2, 1)] for m in spread(a_mons, rr)]
        corr = [[_lits(m, wiresN) + [(mk, 1), (scratch, 1)] for m in spread(a_mons, rr)]
                for mk in masks]
        h = (len(a_mons) + 1) // 2 if len(a_mons) == 5 else 5
        chunks = [casc[0:h], casc[h:]]
        for cr in corr:
            chunks += [cr[0:h], cr[h:]]
        rr.shuffle(chunks)
        if vtype == "r57":
            chunks[0] = [[(scratch, 1)]] + chunks[0]              # const 1·B
        if vtype in ("r57", "nab"):
            chunks[1] = [[(s2, 1)]] + chunks[1]                   # s2-emit
            u_pool = u_run
        else:                                                     # 'and': no s2emit
            u_pool = u_all
        lin_masks = (1, 2) if vtype in ("r57", "nab") else (1, 2, 1 << 5, 1 << 6)
        u_lin = [_lits(m, wiresN) + [(scratch, 1)] for m in u_pool if m in lin_masks]
        u_pay = [_lits(m, wiresN) + [(scratch, 1)] for m in u_pool if m not in lin_masks]
        cA = [(chaff[0], 1), (scratch, 1)]; cB = [(chaff[1], 1), (scratch, 1)]
        n_b = len(chunks) - 1
        boundary = ([cA, cB, u_lin[0], cA, cB] if n_b == 5
                    else [cA, u_lin[0], cA] if n_b == 3
                    else [cA, u_lin[0], cA] + [cB, cB] * ((n_b - 3) // 2))[:n_b]
        payload = u_pay + u_lin[1:]
        rr.shuffle(payload)
        for k, g in enumerate(payload):
            ch = chunks[k % len(chunks)]
            ch.insert(1 + rr.randrange(max(1, len(ch) - 1)), g)
        run = []
        for ci, ch in enumerate(chunks):
            run.extend(ch)
            if ci < len(chunks) - 1:
                run.append(boundary[ci])

        for i in idxs:                            # IDENTICAL order for every group target
            circ.mark(f"A{i} + run -> o{i} (group {gidx})")
            for m in A_mons[i]:
                circ.gate(out3[i], [(s[k], 1) for k in range(3) if (m >> k) & 1], comp=0)
            for ctl in run:
                circ.gate(out3[i], ctl, comp=0)

        if not dirty:
            circ.mark(f"unbuild scr2, reverse (group {gidx})")
            for ctl in reversed(build):           # unbuild in EXACT REVERSE
                circ.gate(s2, ctl, comp=0)
        circ.mark(f"unbuild B (group {gidx})")
        for m in Bm:
            circ.gate(scratch, [(s[k], 1) for k in range(3) if (m >> k) & 1], comp=0)


# ---------------------------------------------------------------- runner / self-test
def run_gate(n=20000, seed=0, vtype="r57", a_single=False, b_single=False, dirty=False,
             U0=SB_U0, U1=SB_U1, sharedB=True):
    """Random encoded inputs -> one gadget -> (circ, info).  info carries the cleartext
    values (a, b, V, c_in, c_out, rho, u), correctness flags, and audit metadata."""
    na = 0 if vtype == "copy" else (1 if a_single else 2)
    nb = 1 if b_single or vtype == "copy" else 2
    ngroups = 2 if sharedB else 3
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
    NW = p

    rng = np.random.default_rng(seed)
    wires = [rng.integers(0, 2, n).astype(np.uint8) for _ in range(NW)]
    for w in OUTB + (SCR,) + (tuple(scr2) if dirty else (scr2,)):
        wires[w] = np.zeros(n, np.uint8)
    circ = Circuit(wires)

    def dec(blocks):
        v = np.zeros(n, np.uint8)
        for blk in blocks:
            v = (v ^ E([circ.init[w] for w in blk])).astype(np.uint8)
        return v
    a = dec(a_blocks) if a_blocks else np.zeros(n, np.uint8)
    b = dec(b_blocks)
    c_in = (E([circ.init[w] for w in S1B]) ^ E([circ.init[w] for w in S2B])).astype(np.uint8)
    rho = E([circ.init[w] for w in RB])
    u = (E([circ.init[w] for w in S1B]) ^ rho).astype(np.uint8)
    V = {"r57": (1 ^ b ^ (a & b)), "nab": ((1 ^ a) & b), "and": (a & b), "copy": b}[vtype].astype(np.uint8)
    c_out = (c_in ^ V).astype(np.uint8)

    gadget_gate(circ, a_blocks, b_blocks, S1B, S2B, RB, OUTB, SCR, scr2, chaff,
                U0=U0, U1=U1, sharedB=sharedB, dirty=dirty, vtype=vtype)

    share1p = [circ.s[w] for w in RB]
    share2p = [circ.s[S2B[0]], circ.s[S2B[1]], circ.s[OUTB[0]], circ.s[OUTB[1]], circ.s[OUTB[2]]]
    c_out_actual = (E(share1p) ^ E(share2p)).astype(np.uint8)
    scr_ok = bool(np.array_equal(circ.s[SCR], np.zeros(n, np.uint8)))
    if not dirty:
        scr_ok &= bool(np.array_equal(circ.s[scr2], np.zeros(n, np.uint8)))
    expw = set(OUTB) | {SCR}
    if vtype != "copy":
        expw |= set(scr2) if dirty else {scr2}
    info = dict(a=a, b=b, c_in=c_in, gate_ab=V, c_out=c_out, rho=rho, u=u,
                correct=bool(np.array_equal(c_out_actual, c_out)),
                scratch_restored=scr_ok, n_gates=len(circ.flips), NW=NW,
                written_expected=expw)
    return circ, info


if __name__ == "__main__":
    print("canonical gadget — correctness / gate-count table")
    for vt in ("r57", "nab", "and", "copy"):
        for (asg, bsg, tag) in ((False, False, "dd"), (True, True, "ss")):
            if vt == "copy" and (asg, bsg) != (True, True):
                continue
            for dr in (False, True):
                circ, info = run_gate(12000, seed=1, vtype=vt, a_single=asg, b_single=bsg, dirty=dr)
                print(f"  {vt:4s} {tag} {'dirty' if dr else 'clean':5s}: "
                      f"gates={info['n_gates']:3d} correct={info['correct']} "
                      f"scr_ok={info['scratch_restored']}")
