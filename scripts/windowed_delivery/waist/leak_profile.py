import sys
from collections import Counter
for tag in sys.argv[1:]:
    rows = []
    for line in open(f'{tag}/exp_C_vs_out.exposure.tsv'):
        if line.startswith('#'): continue
        g, w, t = map(int, line.split('\t'))
        rows.append((g, w, t))
    last = {}
    for g, w, t in rows:
        last[w] = max(last.get(w, -1), g)
    outs = set(last.values())
    interior = [(g, w, t) for g, w, t in rows if t > 0 and g not in outs]
    n = len(rows)
    q = Counter(min(4, 4 * g // n) for g, _, _ in interior)
    ts = Counter(t for _, _, t in interior)
    vis_out = sum(1 for g, w, t in rows if t > 0 and g in outs)
    print(f"{tag:8s} outputs visible {vis_out}/{len(outs)}  interior leaks {len(interior):3d}  "
          f"by C-quarter {[q.get(i,0) for i in range(4)]}  t* {dict(sorted(ts.items()))}")
