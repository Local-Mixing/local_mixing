#!/usr/bin/env python3
"""X-panel progression snapshot: size trajectories from (partial) arm logs.

One panel per damping c, size vs part-2 moves, colored by temperature,
line style by base b. Reads whatever xp_*.log files are present.
"""
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "reports/split_trials_20260805/xpanel_partial"
RESUMED_AT = 5358
TCOL = {"100": "#4269d0", "25": "#3ca951", "8": "#d1495b"}
BSTYLE = {"1.2": ":", "1.5": "-", "1.8": "--"}
NAME = re.compile(r"xp_r([\d.]+)_b([\d.]+)_c(\d)_t(\d+)\.log")
MV = re.compile(r"mv=(\d+) size=(\d+) target=(\d+)")

arms = {}
for path in sorted(glob.glob(f"{DIR}/xp_*.log")):
    m = NAME.search(path)
    if not m:
        continue
    r, b, c, t = m.groups()
    pts, tgt = [], None
    for line in open(path):
        mm = MV.search(line)
        if mm:
            pts.append((int(mm.group(1)) - RESUMED_AT, int(mm.group(2))))
            tgt = int(mm.group(3))
    if pts:
        arms[(r, b, c, t)] = (pts, tgt)

rs = sorted({k[0] for k in arms}, key=float)
cs = sorted({k[2] for k in arms})
fig, axes = plt.subplots(1, len(cs), figsize=(4.0 * len(cs), 3.8), sharey=True)
if len(cs) == 1:
    axes = [axes]
for ax, c in zip(axes, cs):
    tgts = set()
    for (r, b, cc, t), (pts, tgt) in arms.items():
        if cc != c:
            continue
        x = [p[0] / 1e6 for p in pts]
        y = [p[1] / 1e3 for p in pts]
        ax.plot(x, y, color=TCOL[t], linestyle=BSTYLE[b], linewidth=1.4, alpha=0.9)
        tgts.add(tgt)
    for tgt in tgts:
        ax.axhline(tgt / 1e3, color="#000000", alpha=0.25, linewidth=0.8)
    ax.set_title(f"damping c = {c}", fontsize=10)
    ax.set_xlabel("part-2 moves (M)", fontsize=9)
    ax.grid(color="#000000", alpha=0.06, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("size (k gates)", fontsize=9)
handles = [plt.Line2D([], [], color=TCOL[t], linewidth=2, label=f"temp = target/{t}") for t in ("100", "25", "8")]
handles += [plt.Line2D([], [], color="#555", linestyle=BSTYLE[b], linewidth=1.4, label=f"b = {b}") for b in ("1.2", "1.5", "1.8")]
axes[-1].legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle(
    f"X-panel progression snapshot — {len(arms)} arms with data (thin gray = targets); "
    "arrival slope, overshoot and breathing band by knob",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = f"{DIR}/xpanel_progress_snapshot.png"
fig.savefig(out, dpi=160)
print("wrote", out, "-", len(arms), "arms")
