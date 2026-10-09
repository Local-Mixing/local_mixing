#!/usr/bin/env python3
"""Recording stub for the stage-4 Python/bash oracle scripts that
val_chain.sh calls (window_pack3.py, repair.sh, anf_dump_eval.py,
mult_census.py), used by tests/shell/tdp_defaults.bash. Appends
{"name", "args"} to `<run dir>.calls` and writes the files the next step
reads, so the reference chain runs to its end on mock binaries."""
import json
import sys
from pathlib import Path

me = Path(sys.argv[0]).name
args = sys.argv[1:]
if me == "repair.sh":
    rd = Path(sys.argv[0]).parent / args[0]
else:
    rd = Path(args[0]).parent
with open(str(rd) + ".calls", "a") as f:
    f.write(json.dumps({"name": me, "args": args}) + "\n")
if me == "window_pack3.py":
    Path(args[7]).write_text("mpmct1 16 1\n0 0 0\n")
    Path(args[7] + ".bounds").write_text("\n")
elif me == "repair.sh":
    (rd / f"rep{args[1]}.mpmct1").write_text("mpmct1 16 1\n0 0 0\n")
    (rd / f"rep{args[1]}.bounds").write_text("\n")
elif me == "anf_dump_eval.py":
    Path(args[2]).write_bytes(b"body")
