"""n=64 arm comparison driver (big-int audit).  Wire-local source C, fixed per
(n,k,seed).  Arms: brick2, mono-ball, mono-mmd.  Big-int soundness check (D
computes C on wires 0..n-1 and restores ancillas) + cut_exposure_big.
"""
import argparse
import json
import random
import sys
import time

import chain
import audit_big
from atoms import Resynth


def apply_word_state(word, s):
    """Apply a mixed g57/general word to a Python-int state s (bit=wire)."""
    for g in word:
        if isinstance(g[2], (list, tuple)):
            target, comp, ctrls = g
            fires = 1
            for (w, pol) in ctrls:
                bit = (s >> w) & 1
                fires &= bit if pol == 1 else (1 - bit)
            if comp:
                fires ^= 1
            s ^= (fires << target)
        else:
            a, x, y = g
            if ((s >> x) & 1) | (1 - ((s >> y) & 1)):
                s ^= (1 << a)
    return s


def soundness(D, C, n, n_total, trials=256, seed=12345):
    """True iff, for random n-bit inputs (ancilla bits 0), D reproduces C on
    wires 0..n-1 AND restores every ancilla wire to 0."""
    rng = random.Random(seed)
    low = (1 << n) - 1
    for _ in range(trials):
        x = rng.getrandbits(n)                 # ancilla bits (>= n) are 0
        outD = apply_word_state(D, x)
        outC = apply_word_state(C, x)          # C is g57, width n
        if (outD & low) != (outC & low):
            return False, "function mismatch"
        if (outD >> n) != 0:
            return False, "ancilla not restored"
    return True, "ok"


def build_C(n, k, seed):
    return chain.local_circuit(n, k, random.Random(seed))


def run_arm(arm, C, n, seed, cut_subsample=None, ball_radius=4):
    # n=64 is divisible by neither 5 nor 6, and _route_partition drops wires on
    # ragged blocks -- so we compile at the next multiple of w (padding wires,
    # never touched by C, held at 0 and restored) while the AUDITED SOURCE stays
    # width n=64.  compile width cw; delivered n_total = cw (+ mmd ancillas).
    rng = random.Random(seed * 1000 + 7)
    t0 = time.time()
    if arm == "brick2":
        w = 6
        cw = ((n + w - 1) // w) * w
        res = chain.compile_brick2(C, cw, rng, w=w, lam=6, acc=3, lookahead=True)
    elif arm == "mono-ball":
        w = 5
        cw = ((n + w - 1) // w) * w
        res = chain.compile_mono(C, cw, rng, w=w, lam=5, acc=0, refresh=6,
                                 backend="ball", synth_radius=ball_radius)
    elif arm == "mono-mmd":
        w = 6
        cw = ((n + w - 1) // w) * w
        res = chain.compile_mono(C, cw, rng, w=w, lam=6, acc=0, refresh=6,
                                 backend="mmd")
    else:
        raise ValueError(arm)
    tc = time.time() - t0
    D = res["D"]
    # brick2 has no n_total key -> its delivered width is the compile width cw;
    # mono returns n_total = cw + ancillas.
    n_total = res.get("n_total", cw)
    res["n_total"] = n_total
    out = {"arm": arm, "seed": seed, "len": len(D), "compile_w": cw,
           "n_total": n_total,
           "coblocked_frac": res.get("coblocked_frac"),
           "mono_units": res.get("mono_units"), "raw_units": res.get("raw_units"),
           "refreshed": res.get("refreshed"), "mono_fail": res.get("mono_fail"),
           "compile_s": round(tc, 1)}
    print(f"[{arm} seed{seed}] compiled: {json.dumps(out)}", flush=True)
    return res, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["brick2", "mono-ball", "mono-mmd"])
    ap.add_argument("--cut-subsample", type=int, default=None)
    ap.add_argument("--audit", action="store_true",
                    help="run soundness + cut_exposure_big (else just compile)")
    ap.add_argument("--emit", default=None, help="write D as mpmct1 to this path")
    args = ap.parse_args()

    C = build_C(args.n, args.k, args.seed)
    res, meta = run_arm(args.arm, C, args.n, args.seed,
                        cut_subsample=args.cut_subsample)
    D = res["D"]
    n_total = res.get("n_total", args.n)

    if args.emit:
        with open(args.emit, "w") as f:
            f.write(chain.to_mpmct1(D, n_total))
        with open(args.emit.replace(".mpmct1", "_C.mpmct1"), "w") as f:
            f.write(chain.to_mpmct1(C, args.n))
        print(f"  emitted {args.emit} (+ _C.mpmct1)", flush=True)

    if args.audit:
        t0 = time.time()
        ok, why = soundness(D, C, args.n, n_total)
        print(f"[{args.arm} seed{args.seed}] soundness: {'OK' if ok else 'FAIL'} "
              f"({why})  ({time.time()-t0:.1f}s)", flush=True)
        t0 = time.time()
        rep = audit_big.cut_exposure_big(D, C, args.n, n_total=n_total,
                                         cut_subsample=args.cut_subsample)
        rep["audit_s"] = round(time.time() - t0, 1)
        audit_big.print_exposure_big(rep, f"{args.arm} seed{args.seed}")
        cov, slots = rep["source_coverage"]
        summary = {"arm": args.arm, "seed": args.seed, "len": rep["len"],
                   "n_total": rep["n_total"], "coblocked_frac": meta["coblocked_frac"],
                   "mono_units": meta["mono_units"], "raw_units": meta["raw_units"],
                   "refreshed": meta["refreshed"], "mono_fail": meta["mono_fail"],
                   "exposure_fraction": rep["exposure_fraction"],
                   "exposed_pairs": rep["exposed_pairs"],
                   "coverage": [cov, slots],
                   "coverage_pct": round(cov / max(1, slots) * 100, 1),
                   "coverage_scope": rep["coverage_scope"],
                   "full_state_hits": rep["full_state_hits"],
                   "affine": rep["affine"], "audit_s": rep["audit_s"]}
        print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
