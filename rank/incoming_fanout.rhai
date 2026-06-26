// INCOMING replacement ranking (#9) — which equivalent replacement to splice in.
//
// Spec: pick the replacement that moves the GLOBAL fanout histogram toward the target distribution
//   25/40/20/10/5 over fanout buckets 0/1/2/3/>3, measured as L1 distance of the candidate's own
//   fanout-bucket fractions to the target. Candidates whose max global fanout exceeds MAX_FANOUT
//   are pushed to the back (a hard cap; used only if nothing else is available). Tiebreak: fewer
//   low-leeway gates first.
//
// `cands`: array of maps, each {size, wires_spanned, low_leeway_count, zero_fanout_count,
//          max_fanout, fanout_buckets:[c0,c1,c2,c3,c>3]}.
// Returns: array of candidate indices, best-first.
//
// Tweak the two knobs below to retune.

// ---- knobs ----
fn target()    { [0.25, 0.40, 0.20, 0.10, 0.05] }  // desired fanout 0/1/2/3/>3 fractions
fn max_fanout() { 50 }                               // hard global-fanout cap (match --max-fanout)

// L1 distance of a candidate's fanout-bucket distribution to the target.
fn fanout_l1(c) {
    let n = c.size;
    if n < 1 { n = 1; }
    let tg = target();
    let d = 0.0;
    for k in 0..5 {
        let frac = c.fanout_buckets[k].to_float() / n.to_float();
        let diff = frac - tg[k];
        if diff < 0.0 { diff = -diff; }
        d += diff;
    }
    d
}

fn rank(cands) {
    let cap = max_fanout();
    let n = cands.len();
    let idx = [];
    for i in 0..n { idx.push(i); }

    // Selection sort by: (over-cap last, then L1 to target ascending, then low_leeway ascending).
    for a in 0..n {
        let best = a;
        for b in (a + 1)..n {
            let cb = cands[idx[b]];
            let cc = cands[idx[best]];
            let over_b = if cb.max_fanout > cap { 1 } else { 0 };
            let over_c = if cc.max_fanout > cap { 1 } else { 0 };

            let better = false;
            if over_b < over_c {
                better = true;
            } else if over_b == over_c {
                let lb = fanout_l1(cb);
                let lc = fanout_l1(cc);
                if lb < lc {
                    better = true;
                } else if lb == lc && cb.low_leeway_count < cc.low_leeway_count {
                    better = true;
                }
            }
            if better { best = b; }
        }
        let t = idx[a]; idx[a] = idx[best]; idx[best] = t;
    }
    idx
}
