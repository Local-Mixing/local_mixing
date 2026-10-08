#!/usr/bin/env python3
"""Depth-vs-eff and splice-fraction-vs-eff curves for the long30 arms.

Usage: .venv/bin/python curves.py ARM [ARM...]
For each ARM (e.g. long30/c): expects hmap/long_<tag>_mv*.bin maps already
computed for its snapshots; emits hmap/long_<tag>_ridge.txt (per-snapshot
ridge stats via security_tests/heatmaps machinery), and after all arms,
hmap/long_curves.png plotting depth and DB-splice fraction against eff, plus
hmap/long_curves.csv.
"""
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
RIDGE = os.path.join(ROOT, "security_tests", "heatmaps", "plot_hmap_ridge.py")


def mv_eff_map(log_path):
    mv_eff = {}
    cur = None
    for line in open(log_path):
        m = re.search(r"mv=(\d+) size=", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"profile: phase=\d+ eff=([0-9.]+)", line)
        if m and cur:
            mv_eff[cur] = float(m.group(1))
    return mv_eff


def splice_series(log_path):
    """Cumulative (mv, total, curated) at each report block."""
    out = []
    cur_mv = None
    tot = cur = None
    for line in open(log_path):
        m = re.search(r"mv=(\d+) size=", line)
        if m:
            cur_mv = int(m.group(1))
        if "splice sizes out->in" in line:
            tot = sum(int(c) for _, _, c in re.findall(r"(\d+)->(\d+):(\d+)", line))
        if "splice sizes (curated)" in line:
            cur = sum(int(c) for _, _, c in re.findall(r"(\d+)->(\d+):(\d+)", line))
            if cur_mv:
                out.append((cur_mv, tot or 0, cur))
    return out


def ridge_for(tag):
    stems = sorted(
        (
            os.path.join("hmap", f[:-4])
            for f in os.listdir("hmap")
            if f.startswith(f"long_{tag}_mv") and f.endswith(".bin")
        ),
        key=lambda s: int(re.search(r"mv(\d+)$", s).group(1)),
    )
    titles = ";".join(f"{int(re.search(r'mv(\d+)$', s).group(1)) / 1e6:.1f}M" for s in stems)
    res = subprocess.run(
        [PY, RIDGE, "--out", f"hmap/long_{tag}_plates.png", "--no-ridge",
         "--nperm", "1000", "--titles", titles, *stems],
        capture_output=True, text=True, check=True,
    )
    open(f"hmap/long_{tag}_ridge.txt", "w").write(res.stdout)
    depths = {}
    for line in res.stdout.splitlines():
        m = re.match(r"long_" + tag + r"_mv(\d+)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", line)
        if m:
            depths[int(m.group(1))] = float(m.group(2))
    return depths


def main():
    rows = []
    for arm in sys.argv[1:]:
        tag = os.path.basename(arm)
        log = os.path.join(arm, "stage3.log")
        mveff = mv_eff_map(log)
        near = lambda mv: mveff[min(mveff, key=lambda k: abs(k - mv))]
        depths = ridge_for(tag)
        spl = splice_series(log)
        for mv, depth in sorted(depths.items()):
            # splice fraction over the 2-eff interval ENDING at this snapshot
            older = [x for x in spl if x[0] <= mv]
            base = older[-1] if older else (0, 0, 0)
            prev_mv = mv - 2_800_000
            prevs = [x for x in spl if x[0] <= max(prev_mv, 1)]
            prev = prevs[-1] if prevs else (0, 0, 0)
            dm = max(base[0] - prev[0], 1)
            rows.append({
                "arm": tag, "mv": mv, "eff": round(near(mv), 2), "depth": depth,
                "db_frac": round((base[1] - prev[1]) / dm, 4),
                "cur_frac": round((base[2] - prev[2]) / dm, 4),
            })
    with open("hmap/long_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "mv", "eff", "depth", "db_frac", "cur_frac"])
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for tag, color in (("c", "tab:blue"), ("bp", "tab:orange")):
            sub = [r for r in rows if r["arm"] == tag]
            if not sub:
                continue
            a1.plot([r["eff"] for r in sub], [r["depth"] for r in sub],
                    "o-", color=color, label=f"ncdb ({tag})")
            a2.plot([r["eff"] for r in sub], [100 * r["db_frac"] for r in sub],
                    "o-", color=color, label=f"{tag} all DB")
            a2.plot([r["eff"] for r in sub], [100 * r["cur_frac"] for r in sub],
                    "s--", color=color, alpha=0.6, label=f"{tag} curated")
        a1.axhline(0.3409, color="gray", ls=":", label="old cdb @1M (eff~2)")
        a1.set_xlabel("eff"); a1.set_ylabel("ridge depth"); a1.legend(fontsize=8)
        a1.set_title("stage2-vs-snapshot ridge depth")
        a2.set_xlabel("eff"); a2.set_ylabel("% of moves (per 2-eff window)")
        a2.legend(fontsize=8); a2.set_title("successful DB splices")
        fig.tight_layout()
        fig.savefig("hmap/long_curves.png", dpi=140)
        print("wrote hmap/long_curves.png")
    except Exception as e:  # matplotlib optional
        print(f"plot skipped: {e}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
