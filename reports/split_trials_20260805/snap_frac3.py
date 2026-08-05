#!/usr/bin/env python3
"""Fraction of input nodes with >= K descendants, per snapshot state.

Scans xp_r2_b3_c1_snap.mpmct1.mv*.mpmct1.state (+ the final
xp_r2_b3_c1_snap.state), counts descendants per origin, and writes
snap_frac3.csv: move, size, families, frac>=2, frac>=3, frac>=5, med_span.
Denominator = ALL input nodes of nR20_mixed (179,132), per the
absolute-population convention.
"""
import csv
import glob
import re
import statistics as st

N_INPUT = 179132
SYNTH = 4294967295

def metrics(path):
    fams = {}
    n = 0
    with open(path) as f:
        it = iter(f)
        move = None
        for line in it:
            if line.startswith("moves "):
                move = int(line.split()[1])
            if line.startswith("gates "):
                n = int(line.split()[1])
                break
        for pos in range(n):
            meta = next(it).rsplit(" | ", 1)[1].split()
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
    spans = [f[2] - f[1] for f in fams.values() if f[0] >= 2]
    ge = lambda k: round(sum(1 for x in cnts if x >= k) / N_INPUT, 4)
    q = lambda xs, p: st.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else 0
    return {
        "move": move, "size": n, "families": len(fams),
        "frac_ge2": ge(2), "frac_ge3": ge(3), "frac_ge5": ge(5),
        "span_med": st.median(spans) if spans else 0,
        "span_mean": round(st.mean(spans), 0) if spans else 0,
        "span_p90": q(spans, 90),
    }

paths = sorted(glob.glob("xp_r2_b3_c1_snap.mpmct1.mv*.mpmct1.state"),
               key=lambda p: int(re.search(r"mv(\d+)", p).group(1)))
paths += [p for p in ["xp_r2_b3_c1_snap.state"] if glob.glob(p)]
rows = []
for p in paths:
    r = metrics(p)
    rows.append(r)
    print(p, r)
with open("snap_frac3.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote snap_frac3.csv ({len(rows)} points)")
