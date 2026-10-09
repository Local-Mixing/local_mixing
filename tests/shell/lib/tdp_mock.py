#!/usr/bin/env python3
"""Recording mock of the TDP stage binaries (gen_sandwich_gadget,
circuit_mixer, tdp_packing, hmap_affine), selected by argv[0]'s name.

The driver launches stage binaries under `env -i`, so the mock cannot rely on
an inherited variable: it appends one JSON line {name, args, env} to
`<run dir>.calls` (the run directory is derived from the artifact paths in
argv) and writes small placeholder artifacts. Optional behaviour is read from
`<run dir>.mock` (key=value lines): gen_seed_override=K makes the gen banner
report a different seed; packing_exit=K makes `tdp_packing run` exit K;
packing_status=S writes that status.
"""
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]


def run_dir():
    if name == "gen_sandwich_gadget":
        return Path(args[0]).parent
    if name == "circuit_mixer":
        return Path(args[args.index("--output") + 1]).parent
    if name == "tdp_packing":
        if args[0] == "run":
            return Path(args[args.index("--out-dir") + 1]).parent
        return Path(args[1]).parent
    if name == "hmap_affine":
        return Path(args[args.index("--g") + 1]).parent
    raise SystemExit(f"mock: unknown binary {name}")


rd = run_dir()
with open(str(rd) + ".calls", "a") as f:
    f.write(json.dumps({"name": name, "args": args, "env": dict(os.environ)}) + "\n")
opts = {}
cfg = Path(str(rd) + ".mock")
if cfg.exists():
    for line in cfg.read_text().splitlines():
        k, _, v = line.partition("=")
        opts[k] = v

if name == "gen_sandwich_gadget":
    out = args[0]
    n = int(args[1])
    seed, gseed, sseed = args[7], args[8], args[9]
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
    Path(out).write_text("mpmct1 16 2\n0 0 0\n0 0 0\n")
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
elif name == "hmap_affine":
    print("[span] mock")
