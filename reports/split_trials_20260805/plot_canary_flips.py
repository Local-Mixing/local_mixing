#!/usr/bin/env python3
"""Canary flips vs original circuit position: mean +/- std per decile.

Parses the per-canary dump lines ("[fmix] canary wire=W orig=P now=N flips=F",
orig in permille) from split-stage logs and draws one panel per run: the mean
flip count per canary in each original-position decile, with a +/-1 std band
and the per-canary points behind it. Small multiples, shared x, one hue per
run (identity), no legend needed (panel titles name the single series).
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Default: the three p_join=1 phase-A runs. "arms" mode: the three
# bracket-draw arms of the span comparison, all on nR20_mixed, colors
# matching the span figure so the arm identity carries across figures.
MODES = {
    "phaseA": (
        [
            ("nR20_mixed  (179k g, 99.3% g57)", "nR20.log", "#4269d0"),
            ("g57A phaseA  (1.66M g, 83.5% comp)", "g57A.log", "#3ca951"),
            ("cg1_phaseA5  (65k g, 73.5% comp)", "pA5.log", "#efb118"),
        ],
        "Split-stage NOT-twist coverage: canary flips by original wire-segment position",
        "canary_flips_by_position.png",
    ),
    "arms": (
        [
            ("original cascade (other-half first)", "nR20_legacy.log", "#9498a0"),
            ("directional max-of-2, own direction", "nR20_k2.log", "#efb118"),
            ("directional max-of-2, side ∝ remaining length (shipped)", "nR20_k2p.log", "#4269d0"),
        ],
        "Canary flips by original position, per bracket-draw design (nR20_mixed, p_join = 1, seed 76)",
        "canary_flips_by_arm_nR20.png",
    ),
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "phaseA"
RUNS, TITLE, OUT = MODES[MODE]
LINE = re.compile(r"canary wire=(\d+) orig=(\d+) now=(-?\d+) flips=(\d+)")

def load(path):
    pts = []
    for m in (LINE.search(l) for l in open(path)):
        if m:
            pts.append((int(m.group(2)) / 10.0, int(m.group(4))))  # (% , flips)
    return np.array(pts)

runs = [(title, load(f"reports/split_trials_20260805/{log}"), color) for title, log, color in RUNS]
runs = [(t, p, c) for t, p, c in runs if len(p)]

fig, axes = plt.subplots(len(runs), 1, figsize=(7.2, 2.3 * len(runs)), sharex=True)
if len(runs) == 1:
    axes = [axes]
edges = np.arange(0, 101, 10)
mids = edges[:-1] + 5

for ax, (title, pts, color) in zip(axes, runs):
    pos, flips = pts[:, 0], pts[:, 1]
    mean, std, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = flips[(pos >= lo) & (pos < hi if hi < 100 else pos <= hi)]
        mean.append(sel.mean() if len(sel) else np.nan)
        std.append(sel.std(ddof=1) if len(sel) > 1 else 0.0)
        n.append(len(sel))
    mean, std = np.array(mean), np.array(std)
    ax.scatter(pos, flips, s=7, color=color, alpha=0.22, linewidths=0, zorder=1)
    ax.fill_between(mids, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(mids, mean, color=color, linewidth=2, marker="o", markersize=5, zorder=3)
    ax.set_title(f"{title} — {len(pts)} canaries, mean {flips.mean():.1f} flips", fontsize=9.5, loc="left")
    ax.set_ylabel("flips per canary", fontsize=8.5)
    ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel("original position in circuit (%)", fontsize=9)
axes[-1].set_xticks(edges)
fig.suptitle(TITLE + "\n(line = decile mean, band = ±1 std, dots = individual canaries)",
             fontsize=10, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = f"reports/split_trials_20260805/{OUT}"
fig.savefig(out, dpi=160)
print("wrote", out)
