#!/usr/bin/env python3
"""Gate-makeup census for mpmct1 circuits (split-stage trials 2026-08-05).

Buckets: comp (g57-family) by width and polarity relation; non-comp
conjunctions by width, with the 1-control class split CNOT/NCNOT.
Usage: gate_census.py FILE [FILE...]  — one column per file.
"""
import sys
from collections import Counter

def census(path):
    c = Counter()
    with open(path) as f:
        hdr = f.readline().split()
        assert hdr[0] == "mpmct1", path
        wires, n = int(hdr[1]), int(hdr[2])
        for line in f:
            t = line.split()
            if not t:
                continue
            comp, k = int(t[1]), int(t[2])
            pols = [int(t[3 + 2 * i + 1]) for i in range(k)]
            if comp:
                if k == 2:
                    c["g57 opp-pol (true g57)" if pols[0] != pols[1] else "g57 same-pol"] += 1
                else:
                    c[f"comp w{k}"] += 1
            else:
                if k == 0:
                    c["X (0-ctrl)"] += 1
                elif k == 1:
                    c["CNOT (1-ctrl pos)" if pols[0] else "NCNOT (1-ctrl neg)"] += 1
                elif k == 2:
                    c[f"AND2 ({2 - sum(pols)} neg)"] += 1
                else:
                    c[f"conj w{k}"] += 1
    c["TOTAL"] = n
    c["wires"] = wires
    return c

files = sys.argv[1:]
cs = [census(p) for p in files]
keys = sorted(set().union(*cs) - {"TOTAL", "wires"})
w = max(len(k) for k in keys) + 2
print("".ljust(w) + "".join(p.split("/")[-1][:26].rjust(28) for p in files))
for k in ["TOTAL", "wires"] + keys:
    row = k.ljust(w)
    for c in cs:
        v = c.get(k, 0)
        pct = "" if k in ("TOTAL", "wires") or not c["TOTAL"] else f" ({100*v/c['TOTAL']:5.1f}%)"
        row += f"{v}{pct}".rjust(28)
    print(row)
