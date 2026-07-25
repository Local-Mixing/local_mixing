#!/usr/bin/env python3
"""Read hmap_stat plates: per-row best-predictor agreement vs the null floor.

The plate holds, per (C-prefix, G-prefix) cell, the best agreement of a 1- or
2-wire predictor of C's state bit from G's wires.

BOTH AXES HAVE PORT ARTEFACTS and both must be trimmed:
  * row 0 and the last row are C's input and output states;
  * the first and last COLUMNS are G before the encoding is ramped in and
    after it is stripped — there G's low wires literally hold x (resp. C(x)),
    so any row whose C-state is still close to x reads high there for reasons
    that have nothing to do with the masks.
Cell (0,0) is 1.0 by construction in every build, which makes an untrimmed
"peak" meaningless. `--trim` (default 0.1) drops that fraction of rows and of
columns from each end before taking the per-row max over columns.

Usage: stat_readout.py [--trim F] stem1 [stem2 ...]
"""
import sys, json
import numpy as np


def main(argv):
    trim = 0.1
    if len(argv) > 1 and argv[0] == "--trim":
        trim = float(argv[1])
        argv = argv[2:]
    print(f"{'stem':>18} {'floor':>6} {'median':>7} {'mean':>6} {'p90':>6} {'max':>6}   "
          f"(interior only, trim={trim})")
    for stem in argv:
        m = json.load(open(f"{stem}.meta.json"))
        A = np.frombuffer(open(f"{stem}.bin", "rb").read(), dtype="<f4")
        A = A.reshape(m["rows"], m["cols"]).astype(float)
        r0 = max(1, int(m["rows"] * trim))
        c0 = max(1, int(m["cols"] * trim))
        inner = A[r0:m["rows"] - r0, c0:m["cols"] - c0]
        rm = inner.max(axis=1)
        print(f"{stem.split('/')[-1]:>18} {m['floor']:6.3f} {np.median(rm):7.3f} "
              f"{rm.mean():6.3f} {np.percentile(rm, 90):6.3f} {rm.max():6.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
