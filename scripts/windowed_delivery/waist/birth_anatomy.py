"""Birth-site decomposition + compensation pricing for celebrity values
(RC's interleave+compensate scheme, 2026-10-06).

Pass over the stage-2 stream classifying every birth of a high-mult value:
  transfer   -- second CNOT of a reciprocal packet (partner value re-hop)
  bracket    -- >=2-ctrl write restoring a value this wire held k writes ago
  cnot-re    -- 1-ctrl CNOT re-bearing a previously-seen value (drains, closes)
  hop        -- value re-appears on a different wire via other gate shapes
  fresh      -- first appearance
Then, for each base interval of a celebrity value (birth .. next write of the
wire), count the reads that would need a compensation gate if the interval
were eliminated, and the width of each compensation."""
import sys, random, hashlib
from collections import defaultdict
sys.path.insert(0, '/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001')
from carrier_refresh import load, MASK, NBITS, NBYTES

a_path, label = sys.argv[1:3]
na, A = load(a_path)
rng = random.Random(20261004)
inputs = [rng.getrandbits(NBITS) for _ in range(na)]
def norm(v): return v ^ MASK if v & 1 else v

# per-wire read positions + reading-gate extra-ctrl count (ctrls minus the wire)
reads_at = defaultdict(list)          # wire -> [(gate index, extra ctrls)]
for i, (t, comp, cps, pol) in enumerate(A):
    for c in cps:
        reads_at[c].append((i, len(cps) - 1))

births = []                            # (idx, wire, fp, kind)
mult = defaultdict(int)
last_seen = {}                         # fp -> (wire, idx)
hist = defaultdict(list)               # wire -> last 24 fps (most recent last)
st = list(inputs)
HDEPTH = 24
for i, (t, comp, cps, pol) in enumerate(A):
    acc = MASK
    for c, p in zip(cps, pol):
        acc &= st[c] if p else st[c] ^ MASK
    if comp: acc ^= MASK
    st[t] ^= acc
    fp = norm(st[t])
    if not fp:
        hist[t].append(None)
        continue
    d = hashlib.blake2b(fp.to_bytes(NBYTES, "little"), digest_size=16).digest()
    mult[d] += 1
    seen = last_seen.get(d)
    if seen is None:
        kind = "fresh"
    elif len(cps) == 1:
        # reciprocal packet: previous or next gate is the mirrored CNOT
        prv = A[i-1] if i else None
        nxt = A[i+1] if i + 1 < len(A) else None
        def mirrors(g):
            return g is not None and len(g[2]) == 1 and g[0] == cps[0] and g[2][0] == t
        if mirrors(prv) or mirrors(nxt):
            kind = "transfer"
        else:
            kind = "cnot-re"
    else:
        h = hist[t]
        kind = "bracket" if d in {x for x in h[-HDEPTH:] if x} else (
            "hop" if seen[0] != t else "multi-re")
    births.append((i, t, d, kind))
    last_seen[d] = (t, i)
    hist[t].append(d)
    if len(hist[t]) > 2 * HDEPTH:
        del hist[t][: -HDEPTH]

# hist stores fps; bracket check wants digest -- store digests in hist instead
# (done above: hist holds d)

CL = [(257, 10**9, ">=257"), (65, 256, "65-256"), (5, 64, "5-64")]
print(f"{label}: gates={len(A)} births={len(births)} distinct={len(mult)}")
for lo, hi, nm in CL:
    vals = {d for d, m in mult.items() if lo <= m <= hi}
    sub = [b for b in births if b[2] in vals]
    by = defaultdict(int)
    for _, _, _, k in sub:
        by[k] += 1
    tot = len(sub)
    row = "  ".join(f"{k}:{v} ({100*v/tot:.0f}%)" for k, v in sorted(by.items(), key=lambda x: -x[1]))
    print(f"  class {nm:>7}: {len(vals):>5} values, {tot:>7} births -> {row}")

# ---- compensation pricing: base intervals of >=65 values ----
celeb = {d for d, m in mult.items() if m >= 65}
wr_next = defaultdict(list)            # wire -> sorted write positions
for i, (t, _, _, _) in enumerate(A):
    wr_next[t].append(i)
import bisect
tot_intervals = tot_reads = 0
free = cheap = 0
width = defaultdict(int)               # compensation ctrl count histogram
per_int = []
for i, w, d, kind in births:
    if d not in celeb or kind == "fresh":
        continue
    ws = wr_next[w]
    k = bisect.bisect_right(ws, i)
    end = ws[k] if k < len(ws) else len(A)
    rs = reads_at.get(w, [])
    lo_r = bisect.bisect_left(rs, (i + 1, -1))
    hi_r = bisect.bisect_left(rs, (end, -1))
    n = hi_r - lo_r
    tot_intervals += 1
    tot_reads += n
    per_int.append(n)
    if n == 0: free += 1
    elif n <= 2: cheap += 1
    for _, extra in rs[lo_r:hi_r]:
        # compensation = moved term (<=2 literals) + the read's other ctrls
        width[2 + extra] += 1
# ---- keep-4 strategy: per value, keep the 4 costliest intervals, eliminate
# the rest cheapest-first; price = compensations; result mult <= 5 (4 kept
# + fresh birth) ... keep K tuned so final mult <= 4 where possible ----
by_val = defaultdict(list)
for i, w, d, kind in births:
    if d not in celeb or kind == "fresh":
        continue
    ws = wr_next[w]
    k = bisect.bisect_right(ws, i)
    end = ws[k] if k < len(ws) else len(A)
    rs = reads_at.get(w, [])
    n = bisect.bisect_left(rs, (end, -1)) - bisect.bisect_left(rs, (i + 1, -1))
    by_val[d].append(n)
for KEEP in (3, 4):
    cost = 0; killed = 0; hidden = 0; residual = []
    for d, ns in by_val.items():
        ns.sort()
        elim = ns[:max(0, len(ns) - KEEP)] if len(ns) + 1 > KEEP + 1 else []
        # final mult = kept intervals + 1 fresh birth
        final = (len(ns) - len(elim)) + 1
        cost += sum(elim); killed += len(elim)
        if final <= 4: hidden += 1
        else: residual.append((final, d))
    print(f"keep-{KEEP}: eliminate {killed} births, compensations {cost} "
          f"({100*cost/len(A):.1f}% of stage-2); values pushed to mult<=4: "
          f"{hidden}/{len(by_val)}; residual >=5: {len(residual)}")
# per-value cost histogram for keep-3
costs = sorted(sum(ns[:max(0,len(ns)-3)]) for ns in by_val.values())
print("per-value elimination cost (keep-3): med", costs[len(costs)//2],
      "p90", costs[int(.9*len(costs))], "max", costs[-1])
per_int.sort()
if per_int:
    print(f"\ncompensation pricing (base intervals of mult>=65 values, non-fresh births):")
    print(f"  intervals {tot_intervals}, reads inside {tot_reads} "
          f"(= compensation gates if ALL eliminated; {100*tot_reads/len(A):.1f}% of stage-2)")
    print(f"  free (0 reads): {free} ({100*free/tot_intervals:.0f}%), 1-2 reads: {cheap} "
          f"({100*cheap/tot_intervals:.0f}%), median reads {per_int[len(per_int)//2]}, "
          f"p90 {per_int[int(.9*len(per_int))]}, max {per_int[-1]}")
    print(f"  compensation widths (ctrls): " +
          ", ".join(f"{k}: {v}" for k, v in sorted(width.items())))
