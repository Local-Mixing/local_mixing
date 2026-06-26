// OUTGOING window ranking (#11) — which window of the current circuit to replace (shooting game).
//
// Spec: Pareto-rank candidate windows over three MAXIMIZED criteria
//   (wires_spanned, low_leeway_count, zero_fanout_count).
// A dominates B iff A >= B on all three and A > B on at least one. Emit non-dominated Pareto
// fronts best-first; within a front order by size ASCENDING (smaller window ranks higher).
//
// `cands`: array of maps, each {size, wires_spanned, low_leeway_count, zero_fanout_count,
//          max_fanout, fanout_buckets:[c0,c1,c2,c3,c>3]}.
// Returns: array of candidate indices, best-first. (Any omitted index is appended by the host.)
//
// Tweak ideas: add/remove criteria in `dominates`, or change the within-front tiebreak below.

fn dominates(a, b) {
    let ge = a.wires_spanned   >= b.wires_spanned
          && a.low_leeway_count >= b.low_leeway_count
          && a.zero_fanout_count >= b.zero_fanout_count;
    let gt = a.wires_spanned   >  b.wires_spanned
          || a.low_leeway_count >  b.low_leeway_count
          || a.zero_fanout_count >  b.zero_fanout_count;
    ge && gt
}

fn rank(cands) {
    let n = cands.len();
    let done = [];
    for i in 0..n { done.push(false); }

    let out = [];
    let placed = 0;
    while placed < n {
        // Non-dominated front among the not-yet-placed candidates.
        let front = [];
        for i in 0..n {
            if done[i] { continue; }
            let dominated = false;
            for j in 0..n {
                if done[j] || j == i { continue; }
                if dominates(cands[j], cands[i]) { dominated = true; break; }
            }
            if !dominated { front.push(i); }
        }

        // Within-front tiebreak: size ascending (selection sort, stable enough here).
        let m = front.len();
        for a in 0..m {
            let best = a;
            for b in (a + 1)..m {
                if cands[front[b]].size < cands[front[best]].size { best = b; }
            }
            let t = front[a]; front[a] = front[best]; front[best] = t;
        }

        for i in front {
            out.push(i);
            done[i] = true;
            placed += 1;
        }
    }
    out
}
