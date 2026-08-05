#!/usr/bin/env python3
"""Per-arm ABSOLUTE spread metrics from xpanel state files.

For each xp_*.state: per-origin descendant count (gates carrying that
origin) and descendant span (max - min position, in gates). Origins index
the nR20_mixed input gates and survive both stages (pure crossing mints no
synthetic origins; merges attribute the survivor to one parent — the
July convention, slightly conservative).

Writes xpanel_spread.csv: one row per arm with distribution summaries.
Run in the xpanel dir: python3 xpanel_spread.py
"""
import csv
import glob
import re
import statistics as st

NAME = re.compile(r"xp_r([\d.]+)_b([\d.]+)_c(\d)_t(\d+)\.state")
SYNTH = 4294967295

rows = []
for path in sorted(glob.glob("xp_*.state")):
    m = NAME.search(path)
    if not m:
        continue
    r, b, c, t = m.groups()
    fams = {}  # origin -> [count, min_pos, max_pos]
    n = 0
    with open(path) as f:
        it = iter(f)
        for line in it:
            if line.startswith("gates "):
                n = int(line.split()[1])
                break
        for pos in range(n):
            line = next(it)
            meta = line.rsplit(" | ", 1)[1].split()
            o = int(meta[0])
            if o == SYNTH:
                continue
            fam = fams.get(o)
            if fam is None:
                fams[o] = [1, pos, pos]
            else:
                fam[0] += 1
                fam[2] = pos
    cnts = [f[0] for f in fams.values()]
    spans = [f[2] - f[1] for f in fams.values()]
    multi = [s for f, s in zip(fams.values(), spans) if f[0] >= 2]
    q = lambda xs, p: st.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs[0] if xs else 0)
    rows.append({
        "arm": path[:-6], "r": r, "b": b, "c": c, "t": t,
        "size": n, "origins_alive": len(fams),
        "cnt_med": st.median(cnts) if cnts else 0,
        "cnt_mean": round(st.mean(cnts), 2) if cnts else 0,
        "cnt_p90": q(cnts, 90),
        "singleton_frac": round(sum(1 for x in cnts if x == 1) / len(cnts), 4) if cnts else 0,
        "span_med": st.median(multi) if multi else 0,
        "span_mean": round(st.mean(multi), 0) if multi else 0,
        "span_p90": q(multi, 90),
        "multi_fams": len(multi),
    })
    print(f"{path}: size={n} fams={len(fams)} cnt_med={rows[-1]['cnt_med']} span_med={rows[-1]['span_med']}")

with open("xpanel_spread.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote xpanel_spread.csv ({len(rows)} arms)")
