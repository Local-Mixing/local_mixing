#!/usr/bin/env python3
"""Twist span distributions: original cascade vs directional max-of-k draw.

Parses per-move "[fmix] split ... size=S ... span=N" lines (span >= 0 = a
twist landed that move; frac = span / size at that move) and overlays the two
designs' span-fraction histograms, means annotated, xmid fraction in the
panel titles.
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR = "reports/split_trials_20260805"
ARMS = [
    ("original cascade (other-half first)", f"{DIR}/nR20_legacy.log", "#9498a0"),
    ("directional max-of-2, own direction", f"{DIR}/nR20_k2.log", "#efb118"),
    ("directional max-of-2, side ∝ remaining length", f"{DIR}/nR20_k2p.log", "#4269d0"),
]
LINE = re.compile(r"split mv=\d+ size=(\d+) .* xmid=(\d+) .* span=(-?\d+)")

def load(path):
    fracs = []
    last_xmid = last_joins = 0
    for l in open(path):
        m = LINE.search(l)
        if m:
            size, xmid, span = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if span >= 0:
                fracs.append(span / size)
            last_xmid = xmid
        jm = re.search(r"joins=(\d+)", l)
        if jm:
            last_joins = int(jm.group(1))
    return np.array(fracs), last_xmid, last_joins

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bins = np.linspace(0, 1, 21)
for title, path, color in ARMS:
    fr, xmid, joins = load(path)
    pct = 100 * xmid / joins if joins else 0
    ax.hist(fr, bins=bins, weights=np.full(len(fr), 100.0 / len(fr)),
            histtype="stepfilled", alpha=0.35, color=color, linewidth=0)
    ax.hist(fr, bins=bins, weights=np.full(len(fr), 100.0 / len(fr)),
            histtype="step", color=color, linewidth=2,
            label=f"{title} — mean {fr.mean():.2f}, xmid {pct:.0f}%")
    ax.axvline(fr.mean(), color=color, linestyle=":", linewidth=1.4)

ax.set_xlabel("twist span as fraction of circuit size (at the move)", fontsize=9.5)
ax.set_ylabel("% of twists", fontsize=9.5)
ax.set_title("nR20_mixed, p_join = 1, seed 76: span distribution by bracket-draw design\n"
             "(dotted = mean; xmid = fraction of twists whose brackets straddle the midpoint)",
             fontsize=10, loc="left")
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)
fig.tight_layout()
out = f"{DIR}/span_compare_nR20.png"
fig.savefig(out, dpi=160)
print("wrote", out)
