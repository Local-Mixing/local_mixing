#!/usr/bin/env python3
"""3D histogram of the m1 key's candidates: count over (gates, wires).

Parses the 2-D table from curated_key_histogram output and renders 3D bars,
log-scaled so the 5-order-of-magnitude count range stays readable.
"""

import sys
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


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
            counts = [0 if c == "." else int(c) for c in cells[:-1]]  # drop row total
            rows[g] = counts
    return rows


def main():
    path = sys.argv[1]
    out = sys.argv[2]
    rows = parse(path)
    gates = sorted(rows)
    n_w = max(len(v) for v in rows.values())

    xs, ys, zs, cs = [], [], [], []
    for g in gates:
        for w, count in enumerate(rows[g]):
            if count > 0:
                xs.append(g)
                ys.append(w)
                zs.append(np.log10(count))
                cs.append(count)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=max(cs))
    colors = cm.viridis(norm(cs))
    ax.bar3d(
        np.array(xs) - 0.4,
        np.array(ys) - 0.4,
        np.zeros(len(xs)),
        0.8,
        0.8,
        zs,
        color=colors,
        shade=True,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("gates", labelpad=12)
    ax.set_ylabel("distinct wires", labelpad=12)
    ax.set_zlabel("log10(count)", labelpad=8)
    ax.set_xticks(gates)
    ax.set_yticks(range(0, n_w + 1, 2))
    ax.set_title(
        "m1 key 66ca88… in frozen_curated_v2 — 235,008 candidates\n"
        "count by (gates, wires); 12–16-gate band: 50,384 candidates",
        pad=20,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, label="count")
    ax.view_init(elev=28, azim=-58)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
