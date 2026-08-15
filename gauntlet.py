"""
gauntlet.py — the unified obfuscation-testing pipeline (superset of the old
gadget gauntlet and the heatmap/mixing pipeline).

One uniform machinery:

  BUILD (Python, optional):  chain --(gg|big)-->  G.mpmct1 + init.bin + buildmeta
  GADGETIZE (Rust):          chain --(none|ss|semi|nc|ncx)--> G      [in-process]
  MIX (Rust, optional):      G --(local-mixing walk, their Mixer)--> G'   [--mix N]
  TRACE (Rust):              G or G' --(bit-sliced sim)--> trace.bin/targets.bin/meta
  AUDIT (Rust):              bundle --> a1 | xrows | xtrace | w1 | w2 | w3
                                      (+ <prefix>.hits.jsonl witnesses)
  MAP (Python):              hits.jsonl --> one heatmap PNG per attack
                             (y = target wire of C, x = trace value of G)

Attack taxonomy (two exact-linear kinds, as specified):
  a1     direct wire match (control; must fire on --gadget none)
  xrows  exact-linear, ROW-BOUNDED: Gaussian against a single prefix state G_j
         (the old heatmap/hmap_affine concept), CV-verified
  xtrace exact-linear, GLOBAL: Gaussian over the full trace of G
         (span(init + flips) == span(full trace), CV-verified)
  w1/w2  weight-1/2 correlations (w2 over xor/and/or/and-not), NULL-referenced
  w3     weight-3 correlations (5 ops; strided scan) -- side note, not headline

The gadget ladder:  none < ss < semi < nc/ncx < gg < big
  none  bare chain (positive control)
  ss    paired secret-share (gadgetize_xgates)
  semi  band product-share V = C ^ M(B) ^ kappa with Gray fold
        (gadgetize_xgates_single; "their semi-nonlinear gadget")
  nc    nonlinear carrier, as shipped (gadgetize_xgates_nc, band 2n)
  ncx   nc with NC_EXPAND=1 (monomial-expanded emission: the doc's fix for
        the whole-trace exact attack)
  gg    canonical folded gadget, 193 gates/gate (gate_gadget_v2)
  big   the behemoth, 939 gates/gate (big_gate_gadget: SG of six gg gadgets)

Mixing: --mix N runs their local-mixing walk in-process after gadgetization
(same engine as fmix; twists on at 0.08).  With mixing on this reproduces the
old gen_*_gadget -> fmix pipeline, with gadgetization generalized to every
gadget above.

Usage: python3 gauntlet.py {all|gen|audit|maps|report} [--ks 1,2,16]
       [--mix both|on|off] [--arms ...] [--outdir reports/mx]
"""
import argparse
import json
import os
import re
import subprocess
import sys

GEN = "./target/release/gauntlet_gen"
AUDIT = "./target/release/gauntlet_audit"
PY = ["nix-shell", "shell-gauntlet.nix", "--run"]  # matplotlib env for maps

N = 8
MIX_MOVES = 20000
# correlation tail by k (sigma ~= 1/sqrt): headline cells get extra power
CORR = {1: 16384, 2: 8192, 16: 4096}
CORR_OVERRIDE = {("nonlinear193", 16): 16384}
# arms: name -> (kind, aux policies, extra)
#   kind rust:  gauntlet_gen --gadget <name>
#   kind file:  gauntlet_build.py --gadget <name> then gauntlet_gen --gadget file
# Gadget arms, named  <descriptor><marginal gates per source gate>:
#   none            -- no gadgetization (1 gate/gate; control)
#   secretshare14   -- the paper's paired secret-share gadgetizer (w = s xor r).
#                      Per source gate the cost is a RANDOM variable by design:
#                      one SG from a 7-variant menu of sizes {5,5,4,6,6,6,4}
#                      (mean 36/7 = 5.14) plus rg_freq=2 refresh gadgets drawn
#                      from RG1/RG2/RG3 of sizes {6,6,2} (mean 14/3 = 4.67):
#                      exact expectation 36/7 + 2*14/3 = 14.48 gates/gate.
#                      Plus ~220 gates fixed infrastructure (Z-bookends +
#                      7-CNOT per-value share setups)
#   bandproduct92   -- the paper's product-share band gadgetizer (V = C xor
#                      M(B) xor kappa, Gray fold, production_single); per-gate
#                      cost again randomized (fold menu + mask gathers + band
#                      rolls + ledger top-ups): 4-seed measured mean 92.4
#   nonlinear193    -- the user's folded 193-gate gadget (gate_gadget_v2.py)
#   nonlinear939    -- the user's 939-gate six-fold gadget (big_gate_gadget.py)
#   *_band0/_band16 -- borrow-wire pool-fed variants (band fill keyed by the
#                      full pipeline input; suffix = U0 blind churn layers)
ARMS = {
    "none":          ("rust", ["zero", "random"], {"rust_gadget": "none"}),
    "secretshare14": ("rust", ["zero", "random"], {"rust_gadget": "ss"}),
    "bandproduct92": ("rust", ["zero", "random"], {"rust_gadget": "semi"}),
    "nonlinear193":  ("file", ["builder"], {"gadget": "gg"}),
    "nonlinear939":  ("file", ["builder"], {"gadget": "big"}),
    "nonlinear193_band0":  ("file", ["builder"], {"gadget": "gg", "pool": "band", "blind": 0}),
    "nonlinear193_band16": ("file", ["builder"], {"gadget": "gg", "pool": "band", "blind": 16}),
    "nonlinear939_band0":  ("file", ["builder"], {"gadget": "big", "pool": "band", "blind": 0}),
    "nonlinear939_band16": ("file", ["builder"], {"gadget": "big", "pool": "band", "blind": 16}),
}
# analytic gate/wire counts for Python-built gadgets (deterministic)
def file_layout(gadget, k):
    if gadget == "gg":
        return 10 * N + 12 * k + 2, 193 * k
    return 20 * N + 94 * k, 939 * k


def run(cmd, log, env_extra=None, timeout=None, allow_fail=False):
    env = dict(os.environ)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    with open(log, "a") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if r.returncode != 0 and not allow_fail:
        raise RuntimeError(f"failed: {' '.join(cmd)} (see {log})")


def last_line(log, prefix):
    out = None
    with open(log) as f:
        for line in f:
            if line.startswith(prefix):
                out = line.strip()
    return out


def gen_cell(outdir, arm, k, aux, mix_on):
    """Run build+gadgetize+mix+trace for one cell; return the bundle prefix."""
    kind, _, env = ARMS[arm]
    prefix = os.path.join(outdir, "bundle")
    b_gadget = env.get("gadget", arm)          # builder gadget for file mode
    b_extra = []
    if env.get("pool") == "band":
        b_extra = ["--pool", "band", "--blind-layers", str(env.get("blind", 0))]
    chain = f"reports/gauntlet/chain_k{k}.mpmct1"
    if not os.path.exists(chain):
        os.makedirs("reports/gauntlet", exist_ok=True)
        # deterministic serial chain (same generator as the old pipeline)
        with open(chain, "w") as fh:
            fh.write(f"mpmct1 {N} {k}\n")
            for i in range(k):
                fh.write(f"{i % N} 1 2 {(i + 3) % N} 0 {(i + 5) % N} 1\n")
    corr = CORR_OVERRIDE.get((arm, k), CORR[k])
    seed, gseed = 100 + k, 7000 + k

    if kind == "file":
        # pass 1: build with nominal samples to learn the circuit size
        run(["python3", "gauntlet_build.py", "--gadget", b_gadget, "--c-in", chain,
             "--out-prefix", prefix, "--n", str(N), "--seed", str(seed),
             "--samples", "64"] + b_extra, f"{outdir}/build.log")
        m = re.search(r"nw=(\d+) gates=(\d+)", last_line(f"{outdir}/build.log", "[build"))
        nw, ng = int(m.group(1)), int(m.group(2))
        ng_final = ng
        if mix_on:
            # pass 2: probe the post-mix size (the run panics afterwards on the
            # init sample-count mismatch -- expected; the [mix] line is already out)
            run([GEN, "--gadget", "file", "--g-in", f"{prefix}.mpmct1",
                 "--init-in", f"{prefix}.init.bin", "--c-in", chain,
                 "--out-prefix", "/tmp/mxprobe", "--n", str(N), "--seed", str(seed),
                 "--gadget-seed", str(gseed), "--corr-samples", "64", "--aux", "zero",
                 "--mix", str(MIX_MOVES)], f"{outdir}/probe.log", allow_fail=True)
            m = re.search(r"gates=(\d+)", last_line(f"{outdir}/probe.log", "[mix"))
            assert m, f"probe did not report mixed size (see {outdir}/probe.log)"
            ng_final = int(m.group(1))
        nfeat = nw + 2 * ng_final
        samples = (nfeat + 256 + 63) // 64 * 64 + 2048 + corr
        # pass 3: rebuild at the right sample count (gates deterministic)
        run(["python3", "gauntlet_build.py", "--gadget", b_gadget, "--c-in", chain,
             "--out-prefix", prefix, "--n", str(N), "--seed", str(seed),
             "--samples", str(samples)] + b_extra, f"{outdir}/build.log")
        cmd = [GEN, "--gadget", "file", "--g-in", f"{prefix}.mpmct1",
               "--init-in", f"{prefix}.init.bin", "--c-in", chain,
               "--out-prefix", prefix, "--n", str(N), "--seed", str(seed),
               "--gadget-seed", str(gseed), "--corr-samples", str(corr), "--aux", "zero"]
        if mix_on:
            cmd += ["--mix", str(MIX_MOVES)]
        run(cmd, f"{outdir}/gen.log", env_extra=env)
    else:
        cmd = [GEN, "--gadget", env.get("rust_gadget", arm), "--c-in", chain,
               "--out-prefix", prefix, "--n", str(N), "--seed", str(seed),
               "--gadget-seed", str(gseed), "--corr-samples", str(corr),
               "--aux", aux]
        if mix_on:
            cmd += ["--mix", str(MIX_MOVES)]
        run(cmd, f"{outdir}/gen.log", env_extra=env)
    return prefix


def audit_cell(outdir, prefix, arm, k):
    run([AUDIT, "--prefix", prefix], f"{outdir}/audit.log")
    return last_line(f"{outdir}/audit.log", "RESULT")


def maps_cell(outdir, prefix):
    run(PY + [f"python3 gauntlet_heatmap.py --prefix {prefix} --outdir {outdir}/heatmaps"],
        f"{outdir}/maps.log")


def cell_name(arm, aux, mix_on):
    a = f"_{aux}" if aux != "builder" else ""
    return f"{arm}{a}_{'mix' if mix_on else 'nomix'}"


def iter_cells(arms, ks, mix_modes):
    for k in ks:
        for arm in arms:
            for aux in ARMS[arm][1]:
                for mix_on in mix_modes:
                    yield k, arm, aux, mix_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "gen", "audit", "maps", "report"])
    ap.add_argument("--ks", default="1,2,16")
    ap.add_argument("--mix", default="both", choices=["both", "on", "off"])
    ap.add_argument("--arms", default=",".join(ARMS.keys()))
    ap.add_argument("--outdir", default="reports/mx")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    arms = args.arms.split(",")
    mix_modes = {"both": [False, True], "on": [True], "off": [False]}[args.mix]
    os.makedirs(args.outdir, exist_ok=True)

    index_path = os.path.join(args.outdir, "index.json")

    if args.cmd == "report":
        results = {}
        if os.path.exists(index_path):
            results = {tuple(d["cell"]): d for d in json.load(open(index_path))}
        build_report(args.outdir, results, ks, arms, mix_modes)
        return

    cells = [(k, arm, aux, mix_on) for k, arm, aux, mix_on in iter_cells(arms, ks, mix_modes)]

    def do_cell(k, arm, aux, mix_on):
        name = cell_name(arm, aux, mix_on)
        cdir = os.path.join(args.outdir, f"k{k}", name)
        os.makedirs(cdir, exist_ok=True)
        prefix = os.path.join(cdir, "bundle")
        if args.cmd in ("all", "gen") and not os.path.exists(f"{prefix}.meta"):
            print(f"[cell] gen {name} k={k}", flush=True)
            gen_cell(cdir, arm, k, aux, mix_on)
        res = None
        if args.cmd in ("all", "audit") and not os.path.exists(f"{prefix}.hits.jsonl"):
            print(f"[cell] audit {name} k={k}", flush=True)
            res = audit_cell(cdir, prefix, arm, k)
        if args.cmd in ("all", "maps") and os.path.exists(f"{prefix}.hits.jsonl") \
                and not os.path.exists(os.path.join(cdir, "heatmaps", "w2.png")):
            print(f"[cell] maps {name} k={k}", flush=True)
            maps_cell(cdir, prefix)
        if res is not None:
            # results collection happens in the parent for jobs==1; for
            # parallel runs audit.log holds the RESULT line (report re-reads)
            return (k, name, res)
        return None

    if args.jobs <= 1:
        results = {}
        if os.path.exists(index_path):
            results = {tuple(d["cell"]): d for d in json.load(open(index_path))}
        for k, arm, aux, mix_on in cells:
            r = do_cell(k, arm, aux, mix_on)
            if r:
                results[(r[0], r[1])] = {"cell": [r[0], r[1]], "result": r[2]}
                json.dump(list(results.values()), open(index_path, "w"), indent=1)
    else:
        import concurrent.futures as cf
        os.environ["RAYON_NUM_THREADS"] = str(max(1, 32 // args.jobs))
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(do_cell, *c) for c in cells]
            for f in cf.as_completed(futs):
                f.result()  # surface exceptions
        # collect results from audit logs
        results = {}
        for k, arm, aux, mix_on in cells:
            name = cell_name(arm, aux, mix_on)
            cdir = os.path.join(args.outdir, f"k{k}", name)
            res = last_line(f"{cdir}/audit.log", "RESULT") if os.path.exists(f"{cdir}/audit.log") else None
            if res:
                results[(k, name)] = {"cell": [k, name], "result": res}
        json.dump(list(results.values()), open(index_path, "w"), indent=1)

    if args.cmd == "all":
        results = {tuple(d["cell"]): d for d in json.load(open(index_path))}
        build_report(args.outdir, results, ks, arms, mix_modes)


def build_report(outdir, results, ks, arms, mix_modes):
    def parse(res):
        if not res:
            return None
        d = {}
        for tok in res.split()[1:]:
            if "=" in tok:
                k2, v = tok.split("=", 1)
                d[k2] = v
        return d
    L = []
    L.append("# Unified gauntlet report\n")
    L.append("Ladder: none < secretshare14 < bandproduct92 < nonlinear193 < nonlinear939 "
             "(name = descriptor + marginal gates per source gate; _bandN = pool-fed "
             "variant with N U0 blind layers).  "
             "Attacks: a1 (direct), xrows (row-bounded exact), xtrace (global exact), "
             "w1/w2/w3 (correlations).  mix = their local-mixing walk, "
             f"{MIX_MOVES} moves, twists 0.08.\n")
    for k in ks:
        L.append(f"\n## k = {k}\n")
        L.append("| cell | a1_nt | xrows_nt | xtrace_nt | w1 | w2 | w3 | null(w1,w2,w3) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm in arms:
            for aux in ARMS[arm][1]:
                for mix_on in mix_modes:
                    name = cell_name(arm, aux, mix_on)
                    # read the RESULT straight from the audit log (robust to
                    # arm renames; index.json is only a jobs==1 cache)
                    log = os.path.join(outdir, f"k{k}", name, "audit.log")
                    res = last_line(log, "RESULT") if os.path.exists(log) else None
                    d = parse(res) or parse(results.get((k, name), {}).get("result"))
                    if not d:
                        continue
                    hm = f"k{k}/{name}/heatmaps"
                    L.append(f"| [{name}]({hm}) | {d.get('a1_nt','?')} | {d.get('xrows_nt','?')} "
                             f"| {d.get('xtrace_nt','?')} | {d.get('w1flag','?')} | {d.get('w2flag','?')} "
                             f"| {d.get('w3flag','?')} | {d.get('null1','?')},{d.get('null2','?')},{d.get('null3','?')} |")
    with open(os.path.join(outdir, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] wrote {outdir}/REPORT.md")


if __name__ == "__main__":
    main()
