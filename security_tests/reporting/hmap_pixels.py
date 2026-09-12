#!/usr/bin/env python3
"""Render an hmap_affine plate at TRUE 1:1 — one image pixel per matrix entry.

The ridge plotter draws plates inside a figure, so matplotlib resamples them
to the axes size and neighbouring cells bleed together; single-column
features (which is what the interesting ones are) get smeared into their
neighbours. This writes the array straight to a PNG instead: no axes, no
interpolation, no resampling. `--scale k` replicates each cell into a k x k
block of identical pixels, which magnifies without inventing anything.

Usage:
  hmap_pixels.py <stem> --out plate.png [--scale 6]
                 [--rows A:B] [--cols A:B]      # crop, in cell indices
                 [--gates A:B]                  # crop columns by G-gate index
                 [--vmin 0.25 --vmax 0.5] [--cmap RdYlBu]
                 [--annotate]                   # print the crop's stats

Prints the value range and, with --annotate, per-column means for the crop
so the numbers behind the pixels are on record next to the picture.
"""
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image


def load(stem):
    m = json.load(open(f"{stem}.meta.json"))
    H = np.fromfile(f"{stem}.bin", dtype=np.float32).reshape(m["rows"], m["cols"])
    return H, np.array(m["i_idx"]), np.array(m["j_idx"])


def parse_range(s, hi):
    a, _, b = s.partition(":")
    return (int(a) if a else 0), (int(b) if b else hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=1,
                    help="integer pixel replication (default 1 = true 1:1)")
    ap.add_argument("--scale-x", type=int, default=0,
                    help="per-axis replication; overrides --scale for columns")
    ap.add_argument("--scale-y", type=int, default=0,
                    help="per-axis replication; overrides --scale for rows. "
                         "Anisotropic plates (very wide, few rows) need this to "
                         "be legible without resampling.")
    ap.add_argument("--rows", default="")
    ap.add_argument("--cols", default="")
    ap.add_argument("--gates", default="", help="column crop by G-gate index, A:B")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--cmap", default="RdYlBu")
    ap.add_argument("--annotate", action="store_true")
    a = ap.parse_args()

    H, ii, jj = load(a.stem)
    r0, r1 = parse_range(a.rows, H.shape[0]) if a.rows else (0, H.shape[0])
    if a.gates:
        g0, g1 = parse_range(a.gates, int(jj[-1]))
        keep = np.where((jj >= g0) & (jj <= g1))[0]
        c0, c1 = int(keep[0]), int(keep[-1]) + 1
    else:
        c0, c1 = parse_range(a.cols, H.shape[1]) if a.cols else (0, H.shape[1])
    S = H[r0:r1, c0:c1]
    if S.size == 0:
        sys.exit("empty crop")

    vmin = a.vmin if a.vmin is not None else float(S.min())
    vmax = a.vmax if a.vmax is not None else float(S.max())
    norm = np.clip((S - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = (matplotlib.colormaps[a.cmap](norm)[:, :, :3] * 255).astype(np.uint8)
    sy = a.scale_y or a.scale
    sx = a.scale_x or a.scale
    if sy > 1:
        rgb = np.repeat(rgb, sy, axis=0)
    if sx > 1:
        rgb = np.repeat(rgb, sx, axis=1)
    Image.fromarray(rgb).save(a.out)

    print(f"{a.stem}: crop rows {r0}:{r1} cols {c0}:{c1} "
          f"(G gates {jj[c0]}..{jj[c1-1]}), {S.shape[0]}x{S.shape[1]} cells "
          f"-> {rgb.shape[1]}x{rgb.shape[0]} px")
    print(f"  H range in crop: {S.min():.4f} .. {S.max():.4f}   "
          f"colour scale {vmin:.4f} .. {vmax:.4f}  (blue = hidden, red = leaking)")
    if a.annotate:
        cmn = S.mean(axis=0)
        print("  per-column mean H:")
        for k, v in enumerate(cmn):
            print(f"    col {c0+k:>5}  G gate {jj[c0+k]:>8}  meanH {v:.4f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
