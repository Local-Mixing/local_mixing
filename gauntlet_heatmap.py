"""
gauntlet_heatmap.py — per-(arm, attack) heatmaps from a gauntlet audit bundle.

Grid semantics (as specified for the merged pipeline):
  * y-axis: which wire/target of C (g{i}:a, g{i}:b, g{i}:cold, g{i}:f,
    g{i}:cnew, NULL)
  * x-axis: which value in the trace of G is involved in the attack
    (init wires 0..nw, then per gate g: flip at nw+2g, newval at nw+2g+1)
  * a size-k hit (exact relation or correlation witness) highlights exactly
    its k involved cells, with intensity = strength (1.0 exact, |cov| corr).

One PNG per attack found in <prefix>.hits.jsonl, plus a combined overview.
Style helpers reused from heatmap.py (their plotter idiom).
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ATTACK_ORDER = ["a1", "xrows", "xtrace", "w1", "w2", "w3"]
ATTACK_TITLES = {
    "a1": "direct wire match",
    "xrows": "exact-linear, row-bounded (single state G_j)",
    "xtrace": "exact-linear, global (full trace of G)",
    "w1": "weight-1 correlation",
    "w2": "weight-2 correlation (xor/and/or/anot)",
    "w3": "weight-3 correlation (5 ops)",
}


def load_meta(prefix):
    meta, names = {}, {}
    for line in open(f"{prefix}.meta"):
        p = line.rstrip("\n").split("\t")
        if p[0].startswith("target["):
            names[int(p[0][7:-1])] = p[1]
        elif len(p) == 2:
            meta[p[0]] = p[1]
    return meta, [names[i] for i in sorted(names)]


def render(prefix, attack, hits, meta, names, outdir):
    nw = int(meta["n_wires"]); nf = int(meta["n_features"]); nt = len(names)
    k = int(meta.get("k", "0"))
    # strength grid; NaN background
    grid = np.full((nt, nf), np.nan)
    name2idx = {nm: i for i, nm in enumerate(names)}
    for h in hits:
        ti = name2idx[h["target"]]
        for f in h["features"]:
            f = int(f)
            if f < nf:
                grid[ti, f] = max(grid[ti, f] if not np.isnan(grid[ti, f]) else 0.0,
                                  float(h["strength"]))
    cmap = LinearSegmentedColormap.from_list("hit", ["#ffffcc", "#fd8d3c", "#bd0026"])
    cmap.set_bad("#f7f7f7")
    # flat, wide layout: 81 rows over ~6in (not 14) so cells read as a
    # raster, not tall stripes
    fig_w = min(26, max(12, nf / 1200))
    fig_h = max(2.2, min(6.5, nt * 0.085 + 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(np.ma.masked_invalid(grid), interpolation="nearest", cmap=cmap,
              aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(nt))
    ax.set_yticklabels(names, fontsize=4.5 if nt > 20 else 7)
    ax.tick_params(axis="y", length=0, pad=1)
    # init | gate boundary
    ax.axvline(nw - 0.5, color="#08519c", lw=1.2, ls="--")
    ytxt = nt * 0.55
    ax.text(nw + nf * 0.004, ytxt, " gate flips/newvals ->", fontsize=6,
            color="#08519c", va="center")
    ax.text(nw - nf * 0.004, ytxt, "init wires ", fontsize=6, color="#08519c",
            va="center", ha="right")
    # gadget-period ticks (x = gate index along trace); file-mode bundles
    # name the real builder in builder_gadget (newer gens) or, for older
    # bundles, in the sibling .buildmeta file
    builder = meta.get("builder_gadget", "")
    if not builder and meta.get("gadget") == "file":
        bm = f"{prefix}.buildmeta"
        if os.path.exists(bm):
            for line in open(bm):
                if line.startswith("gadget\t"):
                    builder = line.split("\t", 1)[1].strip()
                    break
    period = {"gg": 193, "big": 939}.get(builder or meta.get("gadget", ""), 0)
    if period and k > 1:
        for gi in range(1, k):
            ax.axvline(nw + 2 * period * gi - 0.5, color="#999999", lw=0.4, alpha=0.6)
    ax.set_xlabel("trace value index (init wires, then per-gate flip/newval)", fontsize=8)
    ax.set_ylabel("target (source wire value)", fontsize=8)
    gadget = meta.get("gadget", "?")
    mixed = meta.get("mixed", "false") == "true"
    ax.set_title(f"{prefix.split('/')[-1]} — {gadget}{' +mix' if mixed else ''} — "
                 f"{attack}: {ATTACK_TITLES.get(attack, attack)} "
                 f"({len(hits)} witness{'es' if len(hits) != 1 else ''})",
                 fontsize=9)
    plt.colorbar(ax.images[0], label="strength (1 = exact, else |cov|)", shrink=0.6)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{attack}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or f"{args.prefix}.heatmaps"
    meta, names = load_meta(args.prefix)
    hits = []
    hp = f"{args.prefix}.hits.jsonl"
    if os.path.exists(hp):
        for line in open(hp):
            line = line.strip()
            if line:
                hits.append(json.loads(line))
    # Always render the full attack set: an empty grid annotated "0 hits" is
    # the visual certification, so every cell has the same six PNGs.
    made = []
    for attack in ATTACK_ORDER:
        ah = [h for h in hits if h["attack"] == attack]
        made.append(render(args.prefix, attack, ah, meta, names, outdir))
    print(f"[heatmap] {args.prefix}: {len(made)} maps -> {outdir}")


if __name__ == "__main__":
    main()
