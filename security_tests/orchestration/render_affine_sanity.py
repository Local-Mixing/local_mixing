#!/usr/bin/env python3
"""Render fixed-scale, port-aware affine heatmaps for red-team circuits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def load(stem: Path) -> tuple[np.ndarray, dict]:
    with Path(f"{stem}.meta.json").open("r", encoding="utf-8") as source:
        meta = json.load(source)
    rows, cols = int(meta["rows"]), int(meta["cols"])
    values = np.fromfile(Path(f"{stem}.bin"), dtype="<f4")
    if values.size != rows * cols:
        raise ValueError(f"wrong number of cells for {stem}")
    return values.reshape(rows, cols), meta


def interior_stats(matrix: np.ndarray, meta: dict, n: int) -> tuple[float, float, int]:
    i_idx = np.asarray(meta["i_idx"], dtype=float)
    j_idx = np.asarray(meta["j_idx"], dtype=float)
    rows = np.flatnonzero((i_idx / i_idx[-1] >= 0.2) & (i_idx / i_idx[-1] <= 0.8))
    cols = np.flatnonzero((j_idx / j_idx[-1] >= 0.2) & (j_idx / j_idx[-1] <= 0.8))
    core = matrix[np.ix_(rows, cols)]
    prominence = np.median(core, axis=1) - np.min(core, axis=1)
    return float(np.mean(core)), float(np.max(prominence)), int(np.count_nonzero(prominence >= 0.5 / n))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="+", type=Path)
    parser.add_argument("--titles", required=True, help="semicolon-separated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--vmin", type=float, default=0.45)
    parser.add_argument("--vmax", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    titles = args.titles.split(";")
    if len(titles) != len(args.stems):
        parser.error("title count must match stem count")

    loaded = [load(stem) for stem in args.stems]
    fig, axes = plt.subplots(
        1,
        len(loaded),
        figsize=(6.1 * len(loaded), 3.8),
        squeeze=False,
        constrained_layout=True,
        sharey=True,
    )
    image = None
    for panel, (axis, (matrix, meta), title) in enumerate(zip(axes[0], loaded, titles)):
        image = axis.imshow(
            matrix,
            origin="upper",
            extent=[0, 1, 1, 0],
            aspect="auto",
            cmap="RdYlBu",
            vmin=args.vmin,
            vmax=args.vmax,
            interpolation="nearest",
        )
        mean_h, max_prominence, rows_one_bit = interior_stats(matrix, meta, args.n)
        axis.add_patch(
            Rectangle(
                (0.2, 0.2),
                0.6,
                0.6,
                fill=False,
                edgecolor="black",
                linewidth=1.1,
                linestyle="--",
            )
        )
        axis.set_title(
            f"{title}\n20–80% core: H={mean_h:.3f} · "
            f"max ridge={max_prominence:.3f} · ≥1-bit rows={rows_one_bit}",
            fontsize=10,
        )
        axis.set_xlabel("fraction of final circuit")
        if panel == 0:
            axis.set_ylabel("fraction of source circuit")
    assert image is not None
    fig.colorbar(image, ax=axes, label="degree-1 reconstruction error H (0.5 = hidden)", shrink=0.85)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
