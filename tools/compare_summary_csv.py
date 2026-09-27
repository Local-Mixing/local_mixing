#!/usr/bin/env python3
"""Side-by-side of selected rows from two summary.csv files (legal-slice inputs vs full-random inputs).
usage: compare_legal_full.py <legal summary.csv> <full summary.csv> <out.md> row-substring [row-substring ...]"""
import csv, sys
legal, full, out = sys.argv[1], sys.argv[2], sys.argv[3]
keys = sys.argv[4:]
def load(p):
    rows = list(csv.reader(open(p)))
    return rows[0][1:], {r[0]: r[1:] for r in rows[1:]}
labs, L = load(legal); labs2, F = load(full)
assert labs == labs2, (labs, labs2)
def pick(d, key):
    for k in d:
        if key.lower() in k.lower():
            return k, d[k]
    return None, None
def fmt(x):
    try:
        v = float(x)
        if v == int(v) and abs(v) >= 1: return f"{int(v):,}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
    except Exception:
        return x
with open(out, "w") as f:
    f.write("| statistic | inputs | " + " | ".join(labs) + " |\n|---|---|" + "---|" * len(labs) + "\n")
    for key in keys:
        k, lv = pick(L, key); _, fv = pick(F, key)
        if lv is None or fv is None:
            f.write(f"| {key} | (missing) |" + " |" * len(labs) + "\n"); continue
        f.write(f"| {k} | legal slice | " + " | ".join(fmt(x) for x in lv) + " |\n")
        f.write(f"| | full random | " + " | ".join(fmt(x) for x in fv) + " |\n")
print(open(out).read())
