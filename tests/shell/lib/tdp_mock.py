#!/usr/bin/env python3
"""Recording mock of the TDP stage binaries (gen_sandwich_gadget,
circuit_mixer, tdp_packing, tdp_unpacking, hmap_affine), selected by argv[0]'s name.

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
packing_status=S writes that status.
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
    if name in ("circuit_mixer", "tdp_unpacking") and a == "--seed-file":
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
    if name == "tdp_unpacking":
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
    print(f"[gen] n={n} |C|={args[2]} |D|={args[3]} s={args[4]} rg_freq={args[5]} slice_gates={args[6]} "
          f"seed={seed} gadget_seed={gseed} sandwich_seed={sseed} gadgetization_mode=embedded-masking "
          f"sandwich_variant=classic")
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
        (out / "rep512.mpmct1").write_text("mpmct1 16 2\n0 0 0\n0 0 0\n")
        (out / "rep512.bounds").write_text("1\n")
elif name == "tdp_unpacking":
    assert args[0] == "run", args
    for flag in ("--deliverable", "--packed", "--bounds", "--source-c"):
        assert Path(args[args.index(flag) + 1]).is_file(), (flag, args)
    out = Path(args[args.index("--out-dir") + 1])
    code = int(opts.get("unpacking_exit", "0"))
    if code:
        print(f"mock tdp_unpacking exit {code}", file=sys.stderr)
        sys.exit(code)
    out.mkdir(mode=0o700)
    status = opts.get("unpacking_status", "ok")
    (out / "unpacked.mpmct1").write_text("mpmct1 16 1\n0 0 0\n")
    (out / "unpacking.report").write_text(f"unpacking_report=1\nstatus={status}\n")
elif name == "hmap_affine":
    print("[span] mock")
