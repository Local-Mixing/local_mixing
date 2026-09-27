#!/usr/bin/env python3
"""Progression dashboard for an fmix run log.

Parses every report line into time series and renders the multi-axis picture
needed to tune a two-phase run: size lock, g57 fraction, per-interval DB hit
rates, growth/shrink balance, twist coverage, and the p_db controller signal.

Usage: fmix_progression.py <run.log> <out.png> [title]
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.titleweight": "600",
    "legend.fontsize": 12, "figure.titlesize": 19,
})


def series(path):
    rows = []
    pat = {
        "mv": r"mv=(\d+)", "size": r" size=(\d+)", "target": r"target=(\d+)",
        # NB two `comp=` on the line: the head gate census and the DB
        # hit/miss pair below. Anchor both, and never grep unanchored.
        "comp": r" comp=(\d+) g57=", "g57": r" g57=(\d+) shaped=",
        "shaped": r" shaped=(\d+) ", "polf": r" polf=([0-9.]+)",
        "pdb": r"pdb=([0-9.]+)",
        "comp_h": r"db pdb=[0-9.]+ comp=(\d+)/", "comp_m": r"db pdb=[0-9.]+ comp=\d+/(\d+)",
        "agn_h": r"agn=(\d+)/", "agn_m": r"agn=\d+/(\d+)",
        "rm": r"rm=(\d+)", "add": r"add=(\d+)", "bab": r"bab=(\d+)",
        "twn": r"twn=(\d+)", "tws": r"tws=(\d+)", "twspan": r"twspan=(\d+)",
    }
    legacy_comp = (r"db comp=(\d+)/(\d+)", None)
    for line in open(path):
        if " mv=" not in line or " size=" not in line:
            continue
        row = {}
        ok = True
        for k, p in pat.items():
            m = re.search(p, line)
            if m:
                row[k] = float(m.group(1))
            elif k in ("pdb",):
                row[k] = float("nan")
            elif k in ("comp_h", "comp_m"):
                m2 = re.search(legacy_comp[0], line)
                row[k] = float(m2.group(1 if k == "comp_h" else 2)) if m2 else 0.0
            elif k in ("mv", "size"):
                ok = False
                break
            else:
                row[k] = 0.0
        if ok:
            rows.append(row)
    return rows


def main():
    log, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else log
    R = series(log)
    if len(R) < 3:
        sys.exit("not enough report lines parsed")
    mv = [r["mv"] for r in R]

    def diff_rate(key_h, key_m=None):
        """Per-interval rate of key_h/(key_h+key_m), or Δkey_h/Δmv."""
        vals = []
        for a, b in zip(R, R[1:]):
            if key_m is None:
                dmv = b["mv"] - a["mv"]
                vals.append((b[key_h] - a[key_h]) / dmv if dmv > 0 else 0.0)
            else:
                dh = b[key_h] - a[key_h]
                dm = b[key_m] - a[key_m]
                vals.append(dh / (dh + dm) if dh + dm > 0 else float("nan"))
        return vals

    mv1 = mv[1:]
    fig, ax = plt.subplots(2, 3, figsize=(24, 11), dpi=140)

    a = ax[0][0]
    a.plot(mv, [r["size"] for r in R], lw=1.6, label="size")
    a.plot(mv, [r["target"] for r in R], lw=1, ls="--", label="target")
    a.set_title("circuit size vs target"); a.legend()

    # Two different questions, so two curves. `shaped/size` is structural --
    # the store emits width-2 comp gates and nothing else, so it reads how much
    # of the circuit the DB produced. `polf` is a twist odometer: negation
    # twists flip one control's polarity and move a gate out of g57 form
    # without touching its shape, saturating near 1/2. Plotting only g57/size
    # (as this panel used to) mixes the two and reads a twist dose as erosion.
    a = ax[0][1]
    a.plot(mv, [r["shaped"] / r["size"] for r in R], lw=1.6, label="shaped/size (DB material)")
    a.plot(mv, [r["polf"] for r in R], lw=1.6, ls="--", label="polf (twist odometer)")
    a.axhline(0.5, color="0.6", ls=":", lw=1)
    a.set_ylim(0, 1); a.set_title("DB shape vs twist polarity"); a.legend(fontsize=7)

    a = ax[0][2]
    a.plot(mv1, diff_rate("agn_h", "agn_m"), lw=1.2, label="agnostic")
    a.plot(mv1, diff_rate("comp_h", "comp_m"), lw=1.2, label="compressing")
    a.set_ylim(0, 1); a.set_title("DB hit rate per interval (/attempt)"); a.legend()

    a = ax[1][0]
    a.plot(mv1, diff_rate("size"), lw=1.2, label="net Δsize/move")
    dbnet = [ (b["add"] - b["rm"] - (c["add"] - c["rm"])) / (b["mv"] - c["mv"]) if b["mv"] > c["mv"] else 0.0
              for c, b in zip(R, R[1:]) ]
    a.plot(mv1, dbnet, lw=1.2, label="DB (add-rm)/move")
    a.axhline(0, color="gray", lw=0.6)
    a.set_title("growth balance (gates/move)"); a.legend()

    a = ax[1][1]
    a.plot(mv, [r["twspan"] / r["size"] for r in R], lw=1.6, label="coverage x")
    a2 = None
    a.set_title("twist coverage (twspan/size)")
    a.legend(loc="upper left")

    a = ax[1][2]
    a.plot(mv, [r["pdb"] for r in R], lw=1.6, label="p_db_eff")
    a.plot(mv1, diff_rate("bab"), lw=0.9, label="build-aborts/move")
    a.set_title("controller signal + sampler health"); a.legend()

    for row_ in ax:
        for a in row_:
            a.tick_params(labelsize=12)
            a.ticklabel_format(style="plain")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("dashboard ->", out)


if __name__ == "__main__":
    main()
