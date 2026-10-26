#!/usr/bin/env python3
"""Recording mock of the TDP stage binaries (gen_sandwich_gadget,
circuit_mixer, tdp_packing, tdp_unpacking, tdp_adaptive, hmap_affine), selected
by argv[0]'s name.

The driver launches stage binaries under `env -i`, so the mock cannot rely on
an inherited variable: it appends one JSON line {name, args, env} to
`<run dir>.calls` (the run directory is derived from the artifact paths in
argv) and writes small placeholder artifacts. Seeds handed over as files
(gen's `@PATH` positionals, the mixer's `--seed-file PATH`, both /dev/fd/N
from the driver) are read as the real binaries read them: `args` records the
resolved argv (`--seed-file P` as `--seed V`, as val_chain.sh passes it),
`argv` the raw one (what ps would show). Optional behaviour is read from
`<run dir>.mock` (key=value lines): gen_seed_override=K makes the gen banner
report a different seed; packing_exit=K makes `tdp_packing run` exit K;
packing_status=S writes that status; unpacking_exit=K / unpacking_status=S
likewise for `tdp_unpacking run --refine` (the unpacking sub-step of the
delivery stage: it writes the refined deliverable.tdp4, bounds.final,
unpacked.mpmct1, windows.tsv, rounds.tsv and unpacking.report);
adaptive_exit=K / adaptive_status=S likewise for `tdp_adaptive run` (stage 6 in
its adaptive mode: it writes final.mpmct1, adaptive.report and adaptive.log).
"""
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
argv = sys.argv[1:]
private = set()


def read_seed(path):
    with open(path) as f:
        return f.read().strip()


args = []
i = 0
while i < len(argv):
    a = argv[i]
    if name in ("circuit_mixer", "tdp_unpacking", "tdp_adaptive") and a == "--seed-file":
        args += ["--seed", read_seed(argv[i + 1])]
        private.add("--seed")
        i += 2
        continue
    if name == "gen_sandwich_gadget" and 7 <= i <= 9 and a.startswith("@"):
        args.append(read_seed(a[1:]))
        private.add(i)
    else:
        args.append(a)
    i += 1


def run_dir():
    if name == "gen_sandwich_gadget":
        return Path(args[0]).parent
    if name == "circuit_mixer":
        return Path(args[args.index("--output") + 1]).parent
    if name == "tdp_packing":
        if args[0] == "run":
            return Path(args[args.index("--out-dir") + 1]).parent
        return Path(args[1]).parent
    if name in ("tdp_unpacking", "tdp_adaptive"):
        return Path(args[args.index("--out-dir") + 1]).parent
    if name == "hmap_affine":
        return Path(args[args.index("--g") + 1]).parent
    raise SystemExit(f"mock: unknown binary {name}")


rd = run_dir()
with open(str(rd) + ".calls", "a") as f:
    f.write(json.dumps({"name": name, "args": args, "argv": argv, "env": dict(os.environ)}) + "\n")
opts = {}
cfg = Path(str(rd) + ".mock")
if cfg.exists():
    for line in cfg.read_text().splitlines():
        k, _, v = line.partition("=")
        opts[k] = v

if name == "gen_sandwich_gadget":
    out = args[0]
    n = int(args[1])
    seed, gseed, sseed = (("(file)" if k in private else args[k]) for k in (7, 8, 9))
    if "gen_seed_override" in opts:
        seed = opts["gen_seed_override"]
    mode = args[10] if len(args) > 10 else "embedded-masking"
    construction = os.environ.get("TDP_SOURCE_CONSTRUCTION", "random")
    print(f"[gen] n={n} |C|={args[2]} |D|={args[3]} s={args[4]} rg_freq={args[5]} slice_gates={args[6]} "
          f"seed={seed} gadget_seed={gseed} sandwich_seed={sseed} gadgetization_mode={mode} "
          f"sandwich_variant=classic source_construction={construction}")
    g = int(opts.get("gen_gates", "1"))
    Path(out).write_text(f"mpmct1 {4 * n} {g}\n" + "0 0 0\n" * g)
    Path(out + ".sandwich.mpmct1").write_text(f"mpmct1 {2 * n} 1\n0 0 0\n")
    Path(out + ".source_c.g57").write_text("c\n")
elif name == "circuit_mixer":
    out = args[args.index("--output") + 1]
    if "--resume" in args:
        # a resumed run (stage 5 after stage 4) reads no --input: the state must exist
        state = Path(args[args.index("--resume") + 1])
        assert state.is_file() and state.read_text().startswith("circuit-mixer-state"), state
    else:
        assert Path(args[args.index("--input") + 1]).is_file(), args
    Path(out).write_text("mpmct1 16 2\n0 0 0\n0 0 0\n")
    if "--state-out" in args:
        Path(args[args.index("--state-out") + 1]).write_text("circuit-mixer-state 6\nmoves 7\n")
    if "--split" in args:
        reason = opts.get("split_stop", "split stage complete (stopped at the stage boundary)")
    elif "--target-size" in args:
        reason = opts.get("cross_stop", "moves budget spent")
    else:
        reason = opts.get("mixer_stop", "moves budget spent")
    print(f"[circuit_mixer] chain done in 0.0s: {reason} (1 -> 2 gates, 0 -> 0 g57 fossils)")
elif name == "tdp_packing":
    if args[0] == "mult-census":
        print(f" {args[2]}: born=1 distinct=1")
    elif args[0] == "run":
        out = Path(args[args.index("--out-dir") + 1])
        code = int(opts.get("packing_exit", "0"))
        if code:
            print(f"mock tdp_packing exit {code}", file=sys.stderr)
            sys.exit(code)
        out.mkdir(mode=0o700)
        status = opts.get("packing_status", "ok")
        if "--allow-floor-miss" in args and status == "ok" and "packing_status" not in opts:
            status = "ok"
        (out / "deliverable.bin").write_bytes(b"body")
        (out / "deliverable.tdp4").write_bytes(b"LMTDPANF" + bytes(40) + b"body")
        (out / "packing.report").write_text(f"packing_report=2\nstatus={status}\n")
        g = args[args.index("--window-gates") + 1] if "--window-gates" in args else "512"
        (out / f"rep{g}.mpmct1").write_text("mpmct1 16 2\n0 0 0\n0 0 0\n")
        (out / f"rep{g}.bounds").write_text("1\n")
elif name == "tdp_unpacking":
    assert args[0] == "run", args
    for flag in ("--deliverable", "--packed", "--bounds", "--source-c"):
        assert Path(args[args.index(flag) + 1]).is_file(), (flag, args)
    # the driver runs the refinement loop (docs/DELIVERY_STAGE.md) with an
    # explicit round bound; the verbatim fallback switch is gone with it
    assert "--refine" in args and "--max-rounds" in args and "--no-fallback" not in args, args
    rounds = int(args[args.index("--max-rounds") + 1])
    out = Path(args[args.index("--out-dir") + 1])
    code = int(opts.get("unpacking_exit", "0"))
    if code:
        print(f"mock tdp_unpacking exit {code}", file=sys.stderr)
        sys.exit(code)
    out.mkdir(mode=0o700)
    status = opts.get("unpacking_status", "ok")
    (out / "deliverable.tdp4").write_bytes(b"LMTDPANF" + bytes(40) + b"refined")
    (out / "bounds.final").write_text("1\n")
    (out / "unpacked.mpmct1").write_text("mpmct1 16 1\n0 0 0\n")
    (out / "windows.tsv").write_text("idx\tlo\thi\tround\n0\t0\t1\t0\n")
    (out / "rounds.tsv").write_text("round\tpending\tconverged\tfailed\tsplit\n0\t1\t1\t0\t0\n")
    (out / "unpacking.report").write_text(
        f"unpacking_report=2\nstatus={status}\nrounds=1\nwindows_initial=1\nwindows_final=1\n"
        f"splits=0\nunclean_cuts=0\nverbatim={'1' if status == 'verbatim_used' else '0'}\n"
        f"param.max_rounds={rounds}\n")
elif name == "tdp_adaptive":
    assert args[0] == "run", args
    for flag in ("--input", "--source-c", "--sandwich"):
        assert Path(args[args.index(flag) + 1]).is_file(), (flag, args)
    assert "--n" in args and "--seed" in args, args
    out = Path(args[args.index("--out-dir") + 1])
    code = int(opts.get("adaptive_exit", "0"))
    if code:
        print(f"mock tdp_adaptive exit {code}", file=sys.stderr)
        sys.exit(code)
    # the driver creates the directory (it holds attempt.recipe); rounds resume
    out.mkdir(mode=0o700, exist_ok=True)
    status = opts.get("adaptive_status", "ok")
    rounds = args[args.index("--max-rounds") + 1] if "--max-rounds" in args else "0"
    (out / "round1").mkdir(exist_ok=True)
    (out / "round1" / "result.report").write_text("round=1\ninput_gates=1\noutput_gates=1\n")
    (out / "final.mpmct1").write_text("mpmct1 16 1\n0 0 0\n")
    (out / "adaptive.log").write_text("mock\n")
    print(f"tdp: adaptive: mock run, status={status}")
    (out / "adaptive.report").write_text(
        f"adaptive_report=1\nstatus={status}\nrounds=1\nstop_reason=two consecutive rounds without size decrease\n"
        f"input_gates=2\nfinal_gates=1\nparam.max_rounds={rounds}\n")
elif name == "hmap_affine":
    print("[span] mock")
