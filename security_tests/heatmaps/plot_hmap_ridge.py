#!/usr/bin/env python3
"""Ridge analysis + rendering for hmap_affine plates (the CANONICAL heatmap
reader going forward).

Input: one or more hmap_affine outputs, each a `<stem>.bin` (row-major f32,
rows = C prefixes, cols = G prefixes) plus `<stem>.meta.json`. H(i,j) is the
GF(2) reconstruction error of C's prefix-i state from G's prefix-j wires:
0 = affine-recoverable (leak), 0.5 = hidden.

A "diagonal" is a low-H valley (a ridge in reconstructability R = 0.5 - H)
whose location advances monotonically with i -- i.e. C's computational
progress is still legible in the mixed circuit. Do NOT read these maps by the
mean (it saturates near 0.5); read them by the RIDGE:

  depth     mean over rows of  R(ridge cell) - median_j R(i,j)  -- how far the
            ridge stands above its row background. This is the number MIXING
            moves; the valley may fade mid-plate and stay a ridge, so depth is
            a per-row prominence, not a global contrast.
  rho       Spearman corr of row index i vs argmax_j R(i,j) (the ridge
            location). Shape-agnostic monotonicity: rho ~ 1 == clean diagonal,
            tolerant of a fading depth. This is the "is it a diagonal" number.
  perm z    z-score of rho against a null that shuffles the per-row ridge
            columns (nperm draws). Makes "discernible above chance" quantitative.
  contrast  (mean H off-band - mean H on-band) / pooled sigma, band = cells
            within band-frac of the fitted ridge. Confirms the low-H is
            concentrated on the ridge, not smeared.

The rendered ridge (line) is a DEPTH-WEIGHTED smoothing of the per-row
argmins (dots), so shallow rows are bridged by their deep neighbours instead
of scattering -- this is what makes the trace follow a fading valley.

Usage:
  plot_hmap_ridge.py --out plates.png stem1 [stem2 ...]
                     [--titles "pre-mix;run 1;run 2"] [--nperm 3000]
                     [--band-frac 0.06] [--smooth-win 13] [--dpi 145]
Prints the score table to stdout and writes the figure to --out.
"""
import sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(stem):
    m = json.load(open(f"{stem}.meta.json"))
    r, c = m["rows"], m["cols"]
    H = np.frombuffer(open(f"{stem}.bin", "rb").read(), dtype="<f4").reshape(r, c).astype(float)
    ii = np.array(m["i_idx"], float)
    jj = np.array(m["j_idx"], float)
    return H, ii, jj


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def wsmooth(a, w, win):
    r = len(a); out = np.empty(r); h = win // 2
    for k in range(r):
        lo, hi = max(0, k - h), min(r, k + h + 1)
        ww = np.clip(w[lo:hi], 0, None)
        if ww.sum() <= 0:
            ww = np.ones(hi - lo)
        out[k] = np.average(a[lo:hi], weights=ww)
    return out


def ridge_stats(H, nperm=3000, band_frac=0.06, smooth_win=13, seed=0):
    r, c = H.shape
    R = 0.5 - H
    a = np.argmax(R, axis=1)                       # per-row ridge column
    rowmed = np.median(R, axis=1)
    depth = R[np.arange(r), a] - rowmed            # prominence above row bg
    i = np.arange(r)
    rho = spearman(i, a)
    rng = np.random.default_rng(seed)
    null = np.array([spearman(i, rng.permutation(a)) for _ in range(nperm)])
    z = (rho - null.mean()) / (null.std() + 1e-12)
    p = (np.sum(null >= rho) + 1) / (nperm + 1)
    jhat = wsmooth(a.astype(float), depth, smooth_win)
    bw = max(2.0, c * band_frac)
    band = np.abs(np.arange(c)[None, :] - jhat[:, None]) <= bw
    on, off = H[band], H[~band]
    contrast = (off.mean() - on.mean()) / np.sqrt(0.5 * (on.var() + off.var()) + 1e-12)
    return dict(rho=rho, z=z, p=p, depth=float(depth.mean()), depth_med=float(np.median(depth)),
                contrast=float(contrast), a=a, depth_arr=depth, jhat=jhat,
                meanH=float(H.mean()), stdH=float(H.std()))


def main():
    ap = argparse.ArgumentParser(description="hmap_affine ridge analysis + rendering")
    ap.add_argument("stems", nargs="+", help="hmap_affine output stems (each <stem>.bin + .meta.json)")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--titles", default="", help="';'-separated panel titles (default: stem basenames)")
    ap.add_argument("--nperm", type=int, default=3000)
    ap.add_argument("--band-frac", type=float, default=0.06)
    ap.add_argument("--smooth-win", type=int, default=13)
    ap.add_argument("--dpi", type=int, default=145)
    ap.add_argument("--no-ridge", action="store_true",
                    help="render the raw H field only: no per-row dots, no traced ridge "
                         "(the measured statistics are still computed and printed)")
    a = ap.parse_args()

    titles = a.titles.split(";") if a.titles else [s.rsplit("/", 1)[-1] for s in a.stems]
    if len(titles) != len(a.stems):
        ap.error("--titles count must match number of stems")

    res = {s: ridge_stats(load(s)[0], a.nperm, a.band_frac, a.smooth_win) for s in a.stems}
    hdr = f"{'map':22} {'meanH':>7} {'stdH':>6} {'depth':>7} {'depthMed':>8} {'rho':>6} {'perm_z':>7} {'p':>8} {'contrast':>8}"
    print(hdr)
    for s in a.stems:
        d = res[s]
        print(f"{s.rsplit('/',1)[-1]:22} {d['meanH']:7.4f} {d['stdH']:6.3f} {d['depth']:7.4f} "
              f"{d['depth_med']:8.4f} {d['rho']:6.3f} {d['z']:7.2f} {d['p']:8.4f} {d['contrast']:8.3f}")

    allH = np.concatenate([load(s)[0].ravel() for s in a.stems])
    vmin = float(np.percentile(allH, 1)); vmax = 0.5

    n = len(a.stems)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.6), constrained_layout=True, squeeze=False)
    axes = axes[0]
    im = None
    for ax, s, title in zip(axes, a.stems, titles):
        H, ii, jj = load(s); d = res[s]; r, c = H.shape
        im = ax.imshow(H, origin="upper", extent=[0, 1, 1, 0], aspect="auto",
                       cmap="RdYlBu", vmin=vmin, vmax=vmax, interpolation="nearest")
        xf = jj / jj[-1]; yf = ii / ii[-1]
        if not a.no_ridge:
            dep = d["depth_arr"]; dn = (dep - dep.min()) / (np.ptp(dep) + 1e-12)
            ax.scatter(xf[d["a"]], yf, s=6 + 70 * dn, c="k", alpha=0.30, linewidths=0, zorder=3)
            jh = np.clip(d["jhat"].round().astype(int), 0, c - 1)
            ax.plot(xf[jh], yf, "-", color="black", lw=2.2, zorder=4)
            ax.plot(xf[jh], yf, "-", color="white", lw=0.9, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("G prefix  (fraction of mixed circuit)")
        ax.set_ylabel("C prefix  (fraction of original)")
        txt = (f"depth {d['depth']:.3f}   rho {d['rho']:.2f}\n"
               f"perm z {d['z']:.1f}   contrast {d['contrast']:.2f}sigma")
        ax.text(0.03, 0.045, txt, transform=ax.transAxes, fontsize=9.5, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))
    fig.colorbar(im, ax=axes, location="right", shrink=0.8, label="H  (0 = leak,  0.5 = hidden)")
    fig.suptitle(
        "hmap: raw H field (no ridge overlay)" if a.no_ridge
        else "hmap ridge:  dots = per-row argmin (size=depth),  line = depth-weighted ridge",
        fontsize=11)
    fig.savefig(a.out, dpi=a.dpi)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
