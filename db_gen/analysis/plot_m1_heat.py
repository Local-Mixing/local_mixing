#!/usr/bin/env python3
"""2D heatmap of the m1 key's candidates: (gates x wires), color = count (log)."""

import sys
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    rows = {}
    in_table = False
    for line in open(path):
        if line.startswith("=== 2-D histogram"):
            in_table = True
            continue
        if in_table:
            if line.startswith("==="):
                break
            m = re.match(r"\s+(\d+)\s+(.*)", line)
            if not m or line.lstrip().startswith("g\\w") or line.lstrip().startswith("tot"):
                continue
            g = int(m.group(1))
            cells = m.group(2).split()
            rows[g] = [0 if c == "." else int(c) for c in cells[:-1]]
    return rows


def main():
    path, out = sys.argv[1], sys.argv[2]
    rows = parse(path)
    gates = list(range(min(rows), max(rows) + 1))
    n_w = max(len(v) for v in rows.values())
    grid = np.zeros((len(gates), n_w))
    for i, g in enumerate(gates):
        for w, count in enumerate(rows.get(g, [])):
            grid[i, w] = count

    masked = np.ma.masked_equal(grid, 0)
    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#e8e8ee")
    mesh = ax.pcolormesh(
        np.arange(n_w + 1) - 0.5,
        np.arange(gates[0], gates[-1] + 2) - 0.5,
        masked,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=grid.max()),
        cmap=cmap,
        edgecolors="white",
        linewidth=0.6,
    )
    for i, g in enumerate(gates):
        for w in range(n_w):
            c = grid[i, w]
            if c > 0:
                ax.text(
                    w,
                    g,
                    f"{int(c):,}" if c < 1000 else f"{c/1000:.1f}k",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if c > 30 else "black",
                )
    ax.set_xlabel("distinct wires")
    ax.set_ylabel("gates")
    ax.set_xticks(range(n_w))
    ax.set_yticks(gates)
    ax.invert_yaxis()
    ax.set_title(
        "m1 key 66ca88… in frozen_curated_v2 — 235,008 candidates by (gates, wires)\n"
        "12–16-gate rows (50,384 candidates) are the new cross-glued material"
    )
    fig.colorbar(mesh, ax=ax, label="count (log scale)", shrink=0.85)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
