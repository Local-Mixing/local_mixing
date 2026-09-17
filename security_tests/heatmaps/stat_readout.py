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
import argparse
import json
from pathlib import Path

import numpy as np


def interior_scores(stem: Path, trim: float) -> tuple[float, np.ndarray]:
    with Path(f"{stem}.meta.json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    rows, cols = int(metadata["rows"]), int(metadata["cols"])
    if rows <= 0 or cols <= 0:
        raise ValueError("plate dimensions must be positive")
    values = np.fromfile(f"{stem}.bin", dtype="<f4")
    if values.size != rows * cols:
        raise ValueError(f"expected {rows * cols} float32 cells, found {values.size}")
    row_margin = max(1, int(rows * trim))
    col_margin = max(1, int(cols * trim))
    if 2 * row_margin >= rows or 2 * col_margin >= cols:
        raise ValueError(
            "no interior cells remain after trimming the endpoints; "
            "use a denser plate or a smaller trim (at least three rows and columns are required)"
        )
    plate = values.reshape(rows, cols).astype(float)
    interior = plate[row_margin:rows - row_margin, col_margin:cols - col_margin]
    return float(metadata["floor"]), interior.max(axis=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", type=Path, nargs="+", help="hmap_stat output stems, each with .bin and .meta.json files")
    parser.add_argument("--trim", type=float, default=0.1,
                        help="fraction removed from each end of each axis, in [0,0.5); always excludes the endpoint row/column")
    args = parser.parse_args(argv)
    trim = args.trim
    if not 0 <= trim < 0.5:
        parser.error("trim must be finite and in [0,0.5)")
    print(f"{'stem':>18} {'floor':>6} {'median':>7} {'mean':>6} {'p90':>6} {'max':>6}   "
          f"(interior only, trim={trim})")
    for stem in args.stems:
        try:
            floor, scores = interior_scores(stem, trim)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"{stem}: {error}")
        print(f"{stem.name:>18} {floor:6.3f} {np.median(scores):7.3f} "
              f"{scores.mean():6.3f} {np.percentile(scores, 90):6.3f} {scores.max():6.3f}")


if __name__ == "__main__":
    main()
