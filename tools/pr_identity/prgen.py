#!/usr/bin/env python3
"""prgen -- pseudorandom g57 identity generator (v1) + audit.

Usage:
  python3 prgen.py selftest
  python3 prgen.py gen  --n 64 --atoms 12 --launder 4000 [--out word.txt]
  python3 prgen.py audit --n 64 --in word.txt

The generator composes minimal-identity atoms (curated CSV, minted class-O, and
BFS-MITM samples) into one long identity, then launders it with local DB-style
rewrites.  `audit` reports the fingerprint dashboard against a random baseline.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import g57
import audit as audit_mod
from atoms import (load_curated, load_identity_library, mint_class_o_library,
                   mint_doubled_cycle, Resynth)
from sampler import MITMSampler
from generator import Generator

CURATED = os.path.join(HERE, "atoms_curated.csv")
LIBRARY = os.path.join(HERE, "identities_m1m11_le12_dedup.csv")
if not os.path.exists(LIBRARY):
    LIBRARY = os.path.join(HERE, "identities_m1m11_le12.csv")
# Repeat-free multi-target atoms (constructed): fix BOTH the write-collision and
# repeated-gate fingerprints the DB library couldn't. The short (<=16 gate)
# non-factoring ones are the prize.
REPEATFREE = os.path.join(HERE, "repeatfree_atoms.csv")


def build_pool(rng, with_mitm=True, per_len_cap=4000):
    pool = load_curated(CURATED)
    pool += mint_class_o_library(rng, sizes=(3, 4, 5, 6), per_size=3, span=8)
    # The DB-extracted locally-geodesic identity library (330k atoms, 6-12
    # gates). Length-balanced subsample keeps the pool manageable while giving
    # the generator the multi-target atoms the curated set lacked.
    if os.path.exists(LIBRARY):
        pool += load_identity_library(LIBRARY, max_len=12, per_len_cap=per_len_cap, rng=rng)
    if with_mitm:
        # support=4/radius=4 keeps the interim BFS oracle cheap (165k nodes,
        # ~0.2s); yields random minimal identities of length 6-8.  Longer random
        # minimal identities come from the parent DB (complete <=6 gates) once
        # located; curated + class-O already cover lengths up to 15 / arbitrary.
        smp = MITMSampler(support=4, radius=4, seed=rng.randrange(1 << 30))
        for length in (6, 7, 8):
            for _ in range(4):
                w = smp.sample(length)
                if w:
                    pool.append({"name": f"mitm{length}", "word": w, "len": len(w),
                                 "support": g57.support_size(w), "source": "mitm"})
    return pool


def cmd_gen(args):
    rng = random.Random(args.seed)
    pool = build_pool(rng, with_mitm=not args.no_mitm)
    print(f"atom pool: {len(pool)} atoms "
          f"({sum(1 for a in pool if a['source']=='curated')} curated, "
          f"{sum(1 for a in pool if a['source']=='minted-O')} class-O, "
          f"{sum(1 for a in pool if a['source']=='db-library')} db-library, "
          f"{sum(1 for a in pool if a['source']=='mitm')} MITM)")
    gen = Generator(args.n, pool, resynth=Resynth(max_wires=4, radius=4),
                    seed=rng.randrange(1 << 30))
    res = gen.generate(args.atoms, args.launder, share_frac=args.share,
                       interleave=not args.contiguous)
    print(f"raw length {len(res['raw'])}, laundered length {len(res['word'])}, "
          f"rewrites applied {res['rewrites']}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(g57.fmt_word(res["word"]) + "\n")
        print(f"wrote {args.out}")
    if not args.no_audit:
        audit_mod.print_report(audit_mod.full_report(res["raw"], args.n, "raw composition"))
        audit_mod.print_report(audit_mod.full_report(res["word"], args.n, "laundered"))


def cmd_audit(args):
    with open(args.infile) as f:
        toks = f.read().strip().replace("\n", ";").split(";")
    word = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if "," in t:
            a, x, y = (int(z) for z in t.split(","))
        else:
            a, x, y = int(t[0]), int(t[1]), int(t[2])
        word.append((a, x, y))
    assert g57.is_identity_random(word, args.n, trials=8192), "input is not an identity!"
    audit_mod.print_report(audit_mod.full_report(word, args.n, os.path.basename(args.infile)))


def cmd_chain(args):
    import chain as chain_mod
    rng = random.Random(args.seed)
    n, k = args.n, args.k
    C = (chain_mod.local_circuit(n, k, rng) if args.local
         else chain_mod.random_circuit(n, k, rng))
    rs = Resynth(max_wires=4, radius=4)
    pool = g57.make_state_pool(n, trials=256, seed=7)
    arms = (["naive", "randleg", "brick", "brick2", "mono"] if args.arm == "all"
            else [args.arm])

    def run(label, res):
        D = res["D"]
        ok = chain_mod.words_equivalent(D, C, n, pool=pool)
        extra = {kk: v for kk, v in res.items() if kk not in ("D", "arm")}
        print(f"[{label}] len {len(D)}  equivalent-to-C: {'OK' if ok else 'FAIL'}  {extra}")
        assert ok, f"{label}: delivered circuit is not equivalent to C"
        rep = chain_mod.cut_exposure(D, C, n, affine_cuts=args.affine_cuts)
        chain_mod.print_exposure(rep, label)
        if args.dashboard:
            audit_mod.print_report(audit_mod.full_report(D, n, label))
        if args.emit:
            path = f"{args.emit}_{label}.mpmct1"
            with open(path, "w") as f:
                f.write(chain_mod.to_mpmct1(D, n))
            # also emit the source C intermediates so the post-fmix re-audit
            # can be run without recompiling (audit needs C, not D).
            with open(f"{args.emit}_C.mpmct1", "w") as f:
                f.write(chain_mod.to_mpmct1(C, n))
            print(f"  emitted {path} (+ {args.emit}_C.mpmct1)")
        return rep

    print(f"source C: {k} {'wire-local' if args.local else 'random'} g57 gates "
          f"on n={n} wires  (seed {args.seed})")
    # calibration controls: C delivered as-is, and a random word of similar length
    if not args.no_controls:
        rep = chain_mod.cut_exposure(C, C, n, affine_cuts=args.affine_cuts)
        chain_mod.print_exposure(rep, "control: C itself")
        rw = audit_mod.random_word(n, max(64, 2 * k), random.Random(args.seed + 1))
        rep = chain_mod.cut_exposure(rw, C, n, affine_cuts=args.affine_cuts)
        chain_mod.print_exposure(rep, "control: random word")
    for arm in arms:
        if arm == "naive":
            run("naive", chain_mod.compile_naive(C, n, rng, rs=rs,
                                                 launder_steps=args.launder))
        elif arm == "randleg":
            run("randleg", chain_mod.compile_randleg(
                C, n, rng, ell=args.ell, stagger=args.stagger,
                launder_steps=args.launder, rs=rs))
        elif arm == "brick":
            run("brick", chain_mod.compile_brick(
                C, n, rng, w=args.w, mask_len=args.mask_len,
                launder_steps=args.launder, rs=rs,
                gate_interleave=not args.brick_contig,
                unit_resynth=args.unit_resynth))
        elif arm == "brick2":
            run("brick2", chain_mod.compile_brick2(
                C, n, rng, w=args.w, mask_len=args.mask_len, acc=args.acc,
                launder_steps=args.launder, rs=rs,
                lookahead=args.lookahead))
        elif arm == "mono":
            run("mono", chain_mod.compile_mono(
                C, n, rng, w=args.w, mask_len=args.mask_len, acc=args.acc,
                synth_radius=args.synth_radius, refresh=args.refresh,
                launder_steps=args.launder))


def cmd_selftest(args):
    ok = True
    rng = random.Random(1)

    # 1. curated atoms all verify
    pool = load_curated(CURATED)
    print(f"[1] curated atoms load & verify: {len(pool)} atoms  "
          f"({'OK' if len(pool) == 26 else 'FAIL'})")
    ok &= len(pool) == 26

    # 2. class-O minting is sound at several sizes
    good = True
    for k in (3, 4, 5, 6, 7):
        w = mint_doubled_cycle(k, rng)
        good &= g57.is_identity_exact(w, max(g57.wires_of(w)) + 1)
    print(f"[2] class-O minting sound (k=3..7): {'OK' if good else 'FAIL'}")
    ok &= good

    # 3. MITM sampler yields minimal identities
    smp = MITMSampler(support=4, radius=4, seed=7)
    lens_ok, min_ok, got = True, True, 0
    for length in (6, 7, 8):
        w = smp.sample(length)
        if w is None:
            continue
        got += 1
        lens_ok &= (len(w) == length and g57.is_identity_exact(w, 4))
        min_ok &= MITMSampler._minimal(w, 4)
    print(f"[3] MITM sampler (len 6/7/8): {got}/3 sampled, identities={'OK' if lens_ok else 'FAIL'}, "
          f"minimal={'OK' if min_ok else 'FAIL'}")
    ok &= (got >= 2) and lens_ok and min_ok

    # 4. resynth preserves function, changes syntax
    rs = Resynth(max_wires=4, radius=4)
    w = pool[1]["word"][:4]                       # a 4-gate window on <=4 wires
    alts = rs.alternatives(w)
    m = max(g57.wires_of(w)) + 1
    tgt = g57.word_table(w, m).tobytes()
    func_ok = all(g57.word_table(a, m).tobytes() == tgt for a in alts)
    syntax_diff = all(list(a) != list(w) for a in alts)
    print(f"[4] resynth: {len(alts)} alternatives for a 4-gate window, "
          f"function-equal={'OK' if (alts and func_ok) else 'FAIL'}, "
          f"syntactically distinct={'OK' if syntax_diff else 'FAIL'}")
    ok &= bool(alts) and func_ok and syntax_diff

    # 5. end-to-end generate at n=48 stays identity, length-preserving launder
    gpool = build_pool(rng, with_mitm=True)
    gen = Generator(48, gpool, resynth=rs, seed=3)
    res = gen.generate(n_atoms=12, launder_steps=2000, share_frac=0.3)
    e2e = g57.is_identity_random(res["word"], 48, trials=16384)
    len_ok = len(res["word"]) == len(res["raw"])
    print(f"[5] end-to-end n=48: {len(res['raw'])} gates, {res['rewrites']} rewrites, "
          f"length-preserved={'OK' if len_ok else 'FAIL'}, identity={'OK' if e2e else 'FAIL'}")
    ok &= e2e and len_ok

    # 6. interleave kills the contiguous fingerprints (vs random baseline)
    rep = audit_mod.full_report(res["word"], 48, "laundered")
    idw0 = rep["id_windows"]["count"] == 0
    fact0 = rep["factorization"]["word"]["fraction_recovered"] < 0.05
    print(f"[6] interleaved identity: id-windows={rep['id_windows']['count']} "
          f"({'OK' if idw0 else 'FAIL'}), contiguous-factorization="
          f"{rep['factorization']['word']['fraction_recovered']*100:.0f}% "
          f"({'OK' if fact0 else 'FAIL'})")
    ok &= idw0 and fact0

    # 7. repeat-free atom pool + injective placement -> zero repeated gates
    from collections import Counter
    for a in gpool:
        a["_rep"] = sum(c - 1 for c in Counter(a["word"]).values() if c > 1)
    rf = [a for a in gpool if a["_rep"] == 0]
    gen2 = Generator(48, rf, resynth=rs, seed=5)
    w2 = gen2.compose(gen2.seed(14, share_frac=0.3))
    reps = audit_mod.repeated_gates(w2)
    id2 = g57.is_identity_random(w2, 48, trials=16384)
    print(f"[7] repeat-free pool ({len(rf)} atoms): repeated gates={reps} "
          f"({'OK' if reps == 0 else 'FAIL'}), identity={'OK' if id2 else 'FAIL'}")
    ok &= (reps == 0) and id2

    print()
    print("SELFTEST:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="pseudorandom g57 identity generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=64)
    g.add_argument("--atoms", type=int, default=12)
    g.add_argument("--launder", type=int, default=4000)
    g.add_argument("--share", type=float, default=0.15)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--out", type=str, default=None)
    g.add_argument("--contiguous", action="store_true")
    g.add_argument("--no-mitm", action="store_true")
    g.add_argument("--no-audit", action="store_true")
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("audit")
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--in", dest="infile", required=True)
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("selftest")
    s.set_defaults(func=cmd_selftest)

    c = sub.add_parser("chain", help="leg-chain compiler + cut-exposure audit")
    c.add_argument("--n", type=int, default=32)
    c.add_argument("--k", type=int, default=24)
    c.add_argument("--arm",
                   choices=["naive", "randleg", "brick", "brick2", "mono", "all"],
                   default="all")
    c.add_argument("--acc", type=int, default=3, help="brick2/mono: accretion gates/era")
    c.add_argument("--synth-radius", type=int, default=5,
                   help="mono: forward ball radius for unit synthesis")
    c.add_argument("--refresh", type=int, default=0,
                   help="mono: scheduled layer-refresh window (0=at g-visit only)")
    c.add_argument("--ell", type=int, default=48, help="randleg: leg length")
    c.add_argument("--stagger", type=int, default=30, help="randleg: swap passes x len")
    c.add_argument("--w", type=int, default=4, help="brick: block width")
    c.add_argument("--mask-len", type=int, default=6,
                   help="gates per masking piece (>= w); sets unit size ~2*mask_len+1 and degree wall ~min(mask_len+1, w-1)")
    c.add_argument("--launder", type=int, default=0, help="window-launder steps")
    c.add_argument("--affine-cuts", type=int, default=120)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--dashboard", action="store_true")
    c.add_argument("--no-controls", action="store_true")
    c.add_argument("--brick-contig", action="store_true",
                   help="brick: contiguous units instead of gate interleave")
    c.add_argument("--unit-resynth", action="store_true",
                   help="brick: monolithic per-unit resynthesis (kills the "
                        "systematic bare stretch inside <=4-wire units)")
    c.add_argument("--lookahead", action="store_true",
                   help="brick2: register-allocation-style seam routing")
    c.add_argument("--local", action="store_true",
                   help="draw C wire-locally (sliced-sandwich flavor)")
    c.add_argument("--emit", type=str, default=None,
                   help="write each arm's D (and C) as mpmct1 for fmix, prefix")
    c.set_defaults(func=cmd_chain)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
