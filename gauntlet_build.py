"""
gauntlet_build.py — build chained-gate obfuscated circuits with the Python
gadgets (gg = canonical 193-gate folded gadget, big = 939-gate behemoth) and
export them in the uniform gauntlet format so ONE Rust tracer/auditor handles
every gadget:

    <prefix>.mpmct1      the gadgetized circuit (r57 gates)
    <prefix>.init.bin    init columns, u64-LE bit-packed: nw columns x S samples.
                         Encoding blocks uniform random SUBJECT TO each logical
                         wire's decode equalling the chain input x_v; chaff
                         random; out/scratch zero; the final n columns are
                         untouched INPUT HOLDER wires carrying x verbatim (the
                         Rust tracer reads x off them; x never appears on any
                         gate-carrying wire).
    <prefix>.buildmeta   key<TAB>value: nw, ng, samples, builder_checked=true,
                         x_holders, and decode[v] = final decode-block wires
                         for logical v (gg: 10 wires = E(P1)^E(P2); big: 20 =
                         r-pair ++ s-pair, W = (E^E)_r ^ (E^E)_s).  Lets the
                         Rust tracer run a genuine end-to-end behavioral check
                         on the FINAL (possibly mixed) circuit.

Layouts (n logical wires, k chain gates):
  gg : P1[i]=10i..10i+4, P2[i]=10i+5..10i+9; per gate R5+out3+chaff4 = 12k;
       2 global scratch.   nw = 10n + 12k + 2, ng = 193k.
  big: shares r{i},s{i} = [5-block,5-block] each (20 wires/logical); per gate a
       fresh mask share gA (10) + six extras tuples (R5,out3,scr,scr2,chaff4 =
       14 each).  nw = 20n + 94k, ng = 939k.
"""
import argparse
import sys

import numpy as np

import gate_gadget_v2 as G
sys.modules["gate_gadget"] = G   # big_gate_gadget imports `gate_gadget`; give it v2
import big_gate_gadget as BIG


def parse_chain(path):
    gates = []
    with open(path) as fh:
        toks = fh.read().split()
    assert toks[0] == "mpmct1"
    n, ng = int(toks[1]), int(toks[2])
    idx = 3
    for _ in range(ng):
        t, comp, k = int(toks[idx]), int(toks[idx + 1]), int(toks[idx + 2])
        assert comp == 1 and k == 2, "chain must be r57 gates"
        w0, p0, w1, p1 = (int(toks[idx + 3]), int(toks[idx + 4]),
                          int(toks[idx + 5]), int(toks[idx + 6]))
        idx += 7
        a, b = (w0, w1) if p0 == 0 else (w1, w0)   # a = neg pin, b = pos pin
        gates.append((t, a, b))
    return n, gates


# ----------------------------------------------------------------------------
# POOL INTERFACE (borrow wires).  The gadgets never mint their own randomness:
# every mask / chaff / re-share value enters on a *borrow wire* (gg: R0..R4 +
# chaff0..3 per gate; big: gA share (10) + six x (R5 + chaff4) per gate).
# Contract: each borrowed wire is consumed by exactly one gadget gate in one
# keyed context (no cross-gadget reuse), and its values must be
# (pseudo)independent of that gate's operand values.
#
#   --pool ideal        : borrow columns are fresh uniform random (isolated
#                         test; models a perfect avalanche pool).
#   --pool band         : borrow columns are BAND VALUES generated exactly the
#                         way gadgetize_xgates_single / the NC gadget fills its
#                         band (value-level port of emit_band_fill_nl[_blind]):
#                         per band wire  value = pivot  ^  (1..min(7,n-1)
#                         linear data wires)  ^  2 g57 products whose sources
#                         are data wires or earlier band wires (the cascade),
#                         with random literal polarities.  Band values are
#                         deterministic functions of the input x (re-keyed per
#                         input pair), marginally balanced by the pivot trick.
#   --blind-layers L    : pre-processing before the band fill (their NC_BLIND
#                         butterfly): copy the inputs to a blind pool and churn
#                         it through L layers of U0 (the fixed class-preserving
#                         fixed-point-free 5-bit permutation, 35 r57 gates)
#                         over shifted overlapping 5-windows, offsets
#                         (layer*2) % n.  The band fill then reads the blinded
#                         high-degree images of the inputs.  L=0 == off.
#
# In band mode EVERY traced value is a deterministic function of x, so the
# audit's NULL baseline is input-keyed too (gauntlet_gen reads `pool` from
# buildmeta and builds NULL as a random function of x).

# The verified 35-gate U0 (ported from gadgets.rs NC_U0): entries
# (target, [(wire, polarity); 2], lits_len); gate: c[t] ^= AND of literals
# (literal = c[w] if polarity else ~c[w]).
NC_U0 = [
    (2, [(3, 1), (0, 0)], 1), (4, [(2, 1), (3, 0)], 2), (0, [(2, 1), (3, 0)], 2),
    (0, [(2, 0), (3, 0)], 2), (2, [(3, 1), (0, 0)], 1), (1, [(0, 0), (2, 1)], 2),
    (3, [(0, 0), (4, 0)], 2), (3, [(1, 1), (2, 0)], 2), (0, [(2, 1), (3, 1)], 2),
    (3, [(1, 1), (2, 1)], 2), (2, [(3, 0), (4, 1)], 2), (2, [(1, 0), (3, 1)], 2),
    (1, [(2, 0), (0, 0)], 1), (4, [(2, 1), (0, 0)], 1), (1, [(4, 0), (0, 0)], 1),
    (2, [(3, 0), (4, 0)], 2), (2, [(0, 0), (4, 0)], 2), (2, [(3, 0), (4, 0)], 2),
    (3, [(2, 0), (0, 0)], 1), (4, [(3, 0), (0, 0)], 1), (3, [(2, 1), (0, 0)], 1),
    (2, [(1, 1), (4, 0)], 2), (2, [(0, 1), (3, 1)], 2), (2, [(0, 0), (1, 0)], 2),
    (0, [(1, 1), (0, 0)], 1), (4, [(0, 0), (0, 0)], 1), (4, [(1, 1), (3, 0)], 2),
    (2, [(0, 1), (0, 0)], 1), (3, [(0, 0), (2, 0)], 2), (2, [(1, 1), (4, 0)], 2),
    (1, [(0, 0), (2, 0)], 2), (4, [(0, 1), (2, 1)], 2), (2, [(0, 1), (4, 0)], 2),
    (1, [(0, 0), (2, 1)], 2), (3, [(1, 1), (2, 1)], 2),
]


def apply_u0(vals, idx):
    """One U0 application on the 5 columns vals[idx[0..4]] (in place)."""
    c = [vals[i] for i in idx]
    for (t, lits, ln) in NC_U0:
        acc = None
        for (w, pol) in lits[:ln]:
            lit = c[w] if pol else (1 - c[w])
            acc = lit if acc is None else (acc & lit)
        c[t] = (c[t] ^ acc).astype(np.uint8)
    for i, w in enumerate(idx):
        vals[w] = c[i]


def band_pool(X, count, blind_layers, rng, n_key=120):
    """Value-level port of their band fill (emit_band_fill_nl[_blind], fill_nl=2).

    X: the n data columns the chain reads.  The pool is keyed by the FULL
    pipeline input: X plus n_key fresh random "neighbor input" columns (the
    production sandwich runs at n=128; our chain reads 8 of them), so every
    band column is a deterministic function of the full input -- "re-keyed
    per input pair" -- and the context space stays huge (no 2^8 collapse),
    while mask<->operand statistics stay honest.
    blind_layers > 0 first churns the full input through the U0 butterfly and
    the fill reads the blinded images (their NC_BLIND path).  Cascade fidelity:
    per-wire transitive data supports are tracked, and a band wire's product
    sources draw 50/50 from earlier band wires whose support excludes this
    wire's pivot (their `eligible_band` rule) and non-pivot data wires.
    """
    n = len(X) + n_key
    src = [c.copy() for c in X] + [
        rng.integers(0, 2, len(X[0])).astype(np.uint8) for _ in range(n_key)]
    if blind_layers > 0:
        for layer in range(blind_layers):
            off = (layer * 2) % n
            i = 0
            while i + 5 <= n:
                apply_u0(src, [(off + i + kk) % n for kk in range(5)])
                i += 5
    band, supports = [], []
    for _ in range(count):
        pivot = int(rng.integers(0, n))
        support = {pivot}
        col = src[pivot].copy()
        lin_max = min(7, n - 1)
        lin_w = min(1 + int(rng.integers(0, lin_max)), n - 1)
        pool = [w for w in range(n) if w != pivot]
        for _ in range(lin_w):
            w = pool.pop(int(rng.integers(0, len(pool))))
            support.add(w)
            col ^= src[w]
        eligible = [i for i in range(len(band)) if pivot not in supports[i]]
        drawn = []
        for _ in range(2):  # fill_nl = 2 (their call sites)
            srcs = []
            for _s in range(2):
                wid = None
                for _try in range(64):
                    if eligible and rng.integers(0, 2):
                        cand = ("b", eligible[int(rng.integers(0, len(eligible)))])
                    else:
                        w = int(rng.integers(0, n))
                        if w == pivot:
                            continue
                        cand = ("s", w)
                    if cand not in drawn and cand not in srcs:
                        wid = cand
                        break
                if wid is None:  # give up on distinctness; any non-pivot input
                    wid = ("s", (pivot + 1) % n)
                srcs.append(wid)
            drawn += srcs
            lits = []
            for wid in srcs:
                base = band[wid[1]] if wid[0] == "b" else src[wid[1]]
                pol = bool(rng.integers(0, 2))
                lits.append(base if pol else (1 - base))
                if wid[0] == "b":
                    support |= supports[wid[1]]
                else:
                    support.add(wid[1])
            col ^= (lits[0] & lits[1]).astype(np.uint8)  # residual const ignored (their bookkeeping too)
        band.append(col.astype(np.uint8))
        supports.append(support)
    return band


def simulate_chain(n, gates, X):
    s = [c.copy() for c in X]
    out = []
    for (t, a, b) in gates:
        av, bv, cold = s[a], s[b], s[t]
        f = (1 ^ bv ^ (av & bv)).astype(np.uint8)
        cnew = (cold ^ f).astype(np.uint8)
        s[t] = cnew
        out += [av, bv, cold, f, cnew]
    return out, s


def solve_block(wires, blocks, target):
    """Adjust wires so XOR of E over each 5-block equals `target` per sample.
    E = x0^x1^maj(x2,x3,x4); x0 is linear in E, so flip x0 of block 0 to fix."""
    cur = np.zeros_like(target)
    for blk in blocks:
        cur ^= G.E([wires[w] for w in blk])
    wires[blocks[0][0]] = (wires[blocks[0][0]] ^ cur ^ target).astype(np.uint8)


def dump_cols(path, columns):
    arr = np.array(columns, dtype=np.uint8)
    np.packbits(arr, axis=1, bitorder="little").tofile(path)


def E_of(src, blk):
    return G.E([src[w] for w in blk])


# ----------------------------------------------------------------------------
def build_gg(n, gates, samples, rng, pool="ideal", blind_layers=0, n_key=120):
    k = len(gates)
    P1 = {i: list(range(10 * i, 10 * i + 5)) for i in range(n)}
    P2 = {i: list(range(10 * i + 5, 10 * i + 10)) for i in range(n)}
    p = 10 * n
    extras = []
    for _ in range(k):
        R = list(range(p, p + 5)); out3 = list(range(p + 5, p + 8))
        chaff = list(range(p + 8, p + 12)); p += 12
        extras.append((R, out3, chaff))
    scr, scr2 = p, p + 1
    nw = p + 2

    wires = [rng.integers(0, 2, samples).astype(np.uint8) for _ in range(nw)]
    for (R, out3, chaff) in extras:
        for w in out3:
            wires[w] = np.zeros(samples, np.uint8)
    wires[scr] = np.zeros(samples, np.uint8)
    wires[scr2] = np.zeros(samples, np.uint8)

    X = [rng.integers(0, 2, samples).astype(np.uint8) for _ in range(n)]

    # strict = chaff (never leaves its context); transitory = R (becomes
    # the output share block, consumed downstream as *data*, not as mask)
    borrows = {gi: (list(chaff), list(R)) for gi, (R, out3, chaff) in enumerate(extras)}
    null_band = None
    if pool == "band":
        need = 10 * n + sum(len(s) + len(t) for s, t in borrows.values())
        # +1: one extra band column, exported as the audit's NULL baseline
        # (same construction class as every pool-fed value)
        bp = band_pool(X, need + 1, blind_layers, np.random.default_rng(args_seed_pool()),
                       n_key=n_key)
        null_band = bp.pop()
        it = iter(bp)
        for i in range(n):                       # entry share encoding, keyed
            for w in P1[i] + P2[i]:
                wires[w] = next(it)
        for gi in range(k):                      # gadget borrows (chaff, R)
            for w in borrows[gi][0] + borrows[gi][1]:
                wires[w] = next(it)

    for i in range(n):
        solve_block(wires, [P1[i], P2[i]], X[i])

    # input holder columns: x verbatim, appended to init.bin ONLY (never
    # circuit wires, so the plaintext input stays out of the traced circuit)
    xh = list(range(nw, nw + n))
    holder_cols = [X[i].copy() for i in range(n)]

    circ = G.Circuit(wires)

    def val(i):
        return (E_of(circ.s, P1[i]) ^ E_of(circ.s, P2[i])).astype(np.uint8)

    for gi, (t, a, b) in enumerate(gates):
        bv, av, ci = val(b), val(a), val(t)
        co = (ci ^ (1 ^ bv ^ (av & bv))).astype(np.uint8)
        R, out3, chaff = extras[gi]
        G.gadget_gate(circ, (tuple(P1[a]), tuple(P2[a])), (tuple(P1[b]), tuple(P2[b])),
                      tuple(P1[t]), tuple(P2[t]), R, out3, scr, scr2, chaff, vtype="r57")
        P1[t] = list(R)
        P2[t] = [P2[t][0], P2[t][1]] + list(out3)
        assert np.array_equal(val(t), co), f"gg gate {gi} incorrect"

    _, final = simulate_chain(n, gates, X)
    assert all(np.array_equal(val(t), final[t]) for t in range(n)), "gg e2e decode failed"

    check_borrow_isolation(circ, borrows, k, 193)
    decode = {v: list(P1[v]) + list(P2[v]) for v in range(n)}   # final (relabeled) blocks
    if null_band is not None:
        holder_cols.append(null_band)
    return circ, wires, nw, decode, xh, holder_cols, borrows


def check_borrow_isolation(circ, borrows, k, period):
    """STRICT borrow wires (chaff, and for big the gA mask share) may be read
    only by gates of their own context.  Transitory borrows (the R blocks) are
    exempt: the gadget relabels R as the new share block, so downstream gates
    read it as *data* (operand encoding), not as mask -- that mask<->operand
    statistical independence question is exactly what the band-mode battery
    measures, so we do not forbid the data flow, we audit it."""
    owner = {}
    for gi, (strict, _trans) in borrows.items():
        for w in strict:
            owner[w] = gi
    for gi_gate, (t, comp, ctrls) in enumerate(circ.gate_log):
        ctx = gi_gate // period
        for (w, _pol) in ctrls:
            if w in owner:
                assert owner[w] == ctx, \
                    f"strict borrow wire {w} of gate {owner[w]} consumed in context {ctx}"


def args_seed_pool():
    """Separate stream for pool tables so --pool never perturbs ideal draws."""
    return args_seed_pool.value


def build_big(n, gates, samples, rng, pool="ideal", blind_layers=0, n_key=120):
    k = len(gates)
    p = 0

    def block5():
        nonlocal p
        b = list(range(p, p + 5)); p += 5
        return b

    shares = {}
    for i in range(n):
        shares[f"r{i}"] = [block5(), block5()]
        shares[f"s{i}"] = [block5(), block5()]
    per_gate = []
    for gi in range(k):
        gA = f"gA{gi}"
        shares[gA] = [block5(), block5()]
        ex = []
        for _ in range(6):
            R = tuple(block5()); out = tuple(range(p, p + 3)); scr = p + 3; scr2 = p + 4
            p += 5; chaff = tuple(range(p, p + 4)); p += 4
            ex.append((R, out, scr, scr2, chaff))
        per_gate.append((gA, ex))
    nw = p

    wires = [rng.integers(0, 2, samples).astype(np.uint8) for _ in range(nw)]
    for (gA, ex) in per_gate:
        for (R, out, scr, scr2, chaff) in ex:
            for w in out:
                wires[w] = np.zeros(samples, np.uint8)
            wires[scr] = np.zeros(samples, np.uint8)
            wires[scr2] = np.zeros(samples, np.uint8)

    def Wv(src, i):
        return (E_of(src, shares[f"r{i}"][0]) ^ E_of(src, shares[f"r{i}"][1])
                ^ E_of(src, shares[f"s{i}"][0]) ^ E_of(src, shares[f"s{i}"][1])).astype(np.uint8)

    X = [rng.integers(0, 2, samples).astype(np.uint8) for _ in range(n)]

    # strict = gA mask share + all chaff; transitory = the six R blocks
    borrows = {}
    for gi, (gA, ex) in enumerate(per_gate):
        strict = [w for blk in shares[gA] for w in blk]
        trans = []
        for (R, out, scr, scr2, chaff) in ex:
            strict += list(chaff)
            trans += list(R)
        borrows[gi] = (strict, trans)
    null_band = None
    if pool == "band":
        entry = [w for i in range(n) for blk in shares[f"r{i}"] + shares[f"s{i}"] for w in blk]
        need = len(entry) + sum(len(s) + len(t) for s, t in borrows.values())
        bp = band_pool(X, need + 1, blind_layers, np.random.default_rng(args_seed_pool()),
                       n_key=n_key)
        null_band = bp.pop()
        it = iter(bp)
        for w in entry:                          # entry shares, keyed
            wires[w] = next(it)
        for gi in range(k):                      # strict then transitory borrows
            for w in borrows[gi][0] + borrows[gi][1]:
                wires[w] = next(it)

    for i in range(n):
        rho = (E_of(wires, shares[f"r{i}"][0]) ^ E_of(wires, shares[f"r{i}"][1])).astype(np.uint8)
        solve_block(wires, shares[f"s{i}"], (X[i] ^ rho).astype(np.uint8))
        assert np.array_equal(Wv(wires, i), X[i]), f"big init decode failed wire {i}"
        # mask share gA stays uniform random (its W-value is the fresh mask)

    # input holder columns (appended to init.bin only; NOT circuit wires)
    xh = list(range(nw, nw + n))
    holder_cols = [X[i].copy() for i in range(n)]

    circ = G.Circuit(wires)
    for gi, (t, a, b) in enumerate(gates):
        gA, ex = per_gate[gi]
        BIG.sg_gate(circ, shares, a, b, t, gA, ex, tag=f"g{gi}:")   # asserts internally

    _, final = simulate_chain(n, gates, X)
    for t in range(n):
        assert np.array_equal(Wv(circ.s, t), final[t]), f"big e2e decode failed wire {t}"

    check_borrow_isolation(circ, borrows, k, 939)
    decode = {v: (list(shares[f"r{v}"][0]) + list(shares[f"r{v}"][1])
                  + list(shares[f"s{v}"][0]) + list(shares[f"s{v}"][1])) for v in range(n)}
    if null_band is not None:
        holder_cols.append(null_band)
    return circ, wires, nw, decode, xh, holder_cols, borrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gadget", required=True, choices=["gg", "big"])
    ap.add_argument("--c-in", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, required=True,
                    help="total sample count (orchestrator pre-computes the layout)")
    ap.add_argument("--pool", default="ideal", choices=["ideal", "band"],
                    help="borrow-wire feed: fresh rng (isolated test) or their "
                         "band-fill construction on the input (pipeline-faithful)")
    ap.add_argument("--blind-layers", type=int, default=0,
                    help="U0 butterfly churn layers before the band fill "
                         "(their NC_BLIND pre-processing; 0 = off)")
    ap.add_argument("--pool-keys", type=int, default=120,
                    help="neighbor input columns keying the band (production "
                         "sandwich runs at n=128; the chain reads 8 of them)")
    args = ap.parse_args()
    args_seed_pool.value = args.seed + 999999

    cn, gates = parse_chain(args.c_in)
    assert cn == args.n
    n, k = args.n, len(gates)
    samples = (args.samples + 63) // 64 * 64
    rng = np.random.default_rng(args.seed)

    if args.gadget == "gg":
        circ, wires, nw, decode, xh, holders, borrows = build_gg(
            n, gates, samples, rng, args.pool, args.blind_layers, args.pool_keys)
    else:
        circ, wires, nw, decode, xh, holders, borrows = build_big(
            n, gates, samples, rng, args.pool, args.blind_layers, args.pool_keys)

    ng = len(circ.flips)
    expect = 193 * k if args.gadget == "gg" else 939 * k
    assert ng == expect, f"{ng} gates != {expect}"

    with open(f"{args.out_prefix}.mpmct1", "w") as fh:
        fh.write(f"mpmct1 {nw} {ng}\n")
        for (t, comp, ctrls) in circ.gate_log:
            fh.write(f"{t} {comp} {len(ctrls)} " + " ".join(f"{w} {pol}" for w, pol in ctrls) + "\n")

    dump_cols(f"{args.out_prefix}.init.bin", wires + holders)

    with open(f"{args.out_prefix}.buildmeta", "w") as fh:
        fh.write(f"gadget\t{args.gadget}\nk\t{k}\nn\t{n}\nn_wires\t{nw}\nn_gates\t{ng}\n")
        fh.write(f"samples\t{samples}\nbuilder_checked\ttrue\n")
        fh.write(f"pool\t{args.pool}\n")
        fh.write(f"blind_layers\t{args.blind_layers}\n")
        fh.write(f"pool_keys\t{args.pool_keys}\n")
        fh.write(f"null_holder\t{str(args.pool == 'band').lower()}\n")
        fh.write("x_holders\t" + ",".join(map(str, xh)) + "\n")
        fh.write(f"init_cols\t{nw + len(holders)}\n")  # nw circuit wires + n holders (trailing)
        for v in range(n):
            fh.write(f"decode[{v}]\t" + ",".join(map(str, decode[v])) + "\n")
        for gi, (strict, trans) in borrows.items():
            fh.write(f"borrow_strict[g{gi}]\t" + ",".join(map(str, strict)) + "\n")
            fh.write(f"borrow_trans[g{gi}]\t" + ",".join(map(str, trans)) + "\n")

    print(f"[build:{args.gadget}] k={k} nw={nw} gates={ng} samples={samples} "
          f"features={nw + 2 * ng} pool={args.pool} blind={args.blind_layers} builder_checked=true")


if __name__ == "__main__":
    main()
