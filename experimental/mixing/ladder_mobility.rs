//! Collective vs individual mobility of a borrowed-carrier ladder.
//!
//! `leeway_by_width` answers "how far can THIS gate float", which is the wrong
//! question for a ladder: its gates are individually mobile but collectively
//! pinned, because the double sweep's two copies must stay ordered around the
//! target gates. This tool measures the right quantity — the RIGID
//! TRANSLATION distance of the whole ladder, i.e. how many positions the group
//! can move as a unit before some member hits a non-member collider.
//!
//! DETECTION (the T-bracket). `emit_narrow_fragment` realizes a wide fragment
//! as two blocks `[T, R_{m-1}..R_0, R_1..R_{m-1}]`, where `T` writes the
//! fragment target `t` and reads the TOP borrowed wire `b`, and the top rung
//! writes `b`. So on `b`'s touch list (every gate that reads or writes `b`, in
//! circuit order) a ladder leaves
//!
//!     [reads by gates targeting t] [writes to b] [reads by gates targeting t] [writes to b]
//!
//! for any rung count: 1-2 reads per T (emit_g57_form may add a CNOT) and 1-4
//! writes per rung group. Lower rungs are NOT recovered, so the group returned
//! is a SUBSET of the ladder and its rigid-translation distance is an UPPER
//! BOUND on the whole ladder's. Run on `ladder_cap 0` output as the null.
//!
//! Usage: ladder_mobility <circuit.mpmct1> [sample=3000] [cap=4096] [seed=1]

use local_mixing::engine::format::read_mpmct;
use local_mixing::circuit::xgate::XGate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Read,
    Write,
}

fn pct(v: &[usize], p: f64) -> usize {
    if v.is_empty() {
        return 0;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn mean(v: &[usize]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<usize>() as f64 / v.len() as f64
    }
}

// Ordinary two-sided float box (identical to stats::leeway_at / leeway_by_width).
fn leeway(gates: &[XGate], i: usize, cap: usize) -> usize {
    let g = &gates[i];
    let (mut l, mut j) = (0usize, i);
    while j > 0 && l < cap && !XGate::collides(g, &gates[j - 1]) {
        j -= 1;
        l += 1;
    }
    let (mut r, mut k) = (0usize, i + 1);
    while k < gates.len() && r < cap && !XGate::collides(g, &gates[k]) {
        k += 1;
        r += 1;
    }
    l + r
}

/// How far the member set can translate as a RIGID BODY, left + right.
///
/// One rightward step moves members right-to-left, each swapping with the gate
/// currently to its right; that neighbour is never itself a member (the
/// rightmost member's neighbour is outside the set by maximality, and every
/// other member's right slot has just been vacated), so the step is legal iff
/// no member collides with its neighbour. Members never pass each other, so an
/// internal collider — which every adjacent ladder pair is — does not block
/// rigid motion. Simulated on a local window; a failed step is rolled back.
fn rigid_translate(gates: &[XGate], members: &[usize], cap: usize) -> (usize, usize) {
    let (lo, hi) = (members[0], *members.last().unwrap());
    let wlo = lo.saturating_sub(cap);
    let whi = (hi + cap + 1).min(gates.len());
    let mut win: Vec<&XGate> = gates[wlo..whi].iter().collect();
    let mut pos: Vec<usize> = members.iter().map(|&p| p - wlo).collect();
    let k = pos.len();

    let mut right = 0usize;
    'outer: while right < cap {
        if pos[k - 1] + 1 >= win.len() {
            break;
        }
        let mut done = 0usize;
        for j in (0..k).rev() {
            let p = pos[j];
            if XGate::collides(win[p], win[p + 1]) {
                for jj in k - done..k {
                    let q = pos[jj];
                    win.swap(q, q - 1);
                    pos[jj] = q - 1;
                }
                break 'outer;
            }
            win.swap(p, p + 1);
            pos[j] = p + 1;
            done += 1;
        }
        right += 1;
    }
    // Restore the window, then run the mirror image leftwards.
    let mut win: Vec<&XGate> = gates[wlo..whi].iter().collect();
    let mut pos: Vec<usize> = members.iter().map(|&p| p - wlo).collect();
    let mut left = 0usize;
    'outer2: while left < cap {
        if pos[0] == 0 {
            break;
        }
        let mut done = 0usize;
        for j in 0..k {
            let p = pos[j];
            if XGate::collides(win[p], win[p - 1]) {
                for jj in (0..done).rev() {
                    let q = pos[jj];
                    win.swap(q, q + 1);
                    pos[jj] = q + 1;
                }
                break 'outer2;
            }
            win.swap(p, p - 1);
            pos[j] = p - 1;
            done += 1;
        }
        left += 1;
    }
    (left, right)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: ladder_mobility <circuit.mpmct1> [sample] [cap] [seed]");
    let sample: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let cap: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let sweeps: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let (gates, wires) = read_mpmct(&path).expect("read mpmct1");
    let n = gates.len();
    eprintln!("[ladmob] {} gates, {} wires, cap {}/side, sample {}", n, wires, cap, sample);

    // touches[w] = (gate index, kind, that gate's target), in circuit order.
    let mut touches: Vec<Vec<(usize, Kind, u16)>> = vec![Vec::new(); wires];
    for (i, g) in gates.iter().enumerate() {
        for &(w, _) in &g.ctrls {
            touches[w as usize].push((i, Kind::Read, g.target));
        }
        touches[g.target as usize].push((i, Kind::Write, g.target));
    }

    // --- T-bracket scan -------------------------------------------------
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (b, tl) in touches.iter().enumerate() {
        let mut s = 0usize;
        while s < tl.len() {
            // Segment: a maximal run of one kind, and for reads a single target.
            let seg = |from: usize, want: Kind| -> Option<(usize, u16)> {
                if from >= tl.len() || tl[from].1 != want {
                    return None;
                }
                let t = tl[from].2;
                let mut e = from + 1;
                while e < tl.len() && tl[e].1 == want && (want == Kind::Write || tl[e].2 == t) {
                    e += 1;
                }
                Some((e, t))
            };
            let ok = (|| {
                let (e1, t1) = seg(s, Kind::Read)?;
                if e1 - s > 2 || t1 as usize == b {
                    return None;
                }
                let (e2, _) = seg(e1, Kind::Write)?;
                if e2 - e1 > 4 {
                    return None;
                }
                let (e3, t2) = seg(e2, Kind::Read)?;
                if e3 - e2 > 2 || t2 != t1 {
                    return None;
                }
                let (e4, _) = seg(e3, Kind::Write)?;
                if e4 - e3 > 4 || e4 - e1 != (e2 - e1) + (e3 - e2) + (e4 - e3) {
                    return None;
                }
                // The two rung groups must realize the same function: same
                // read-wire set, same write count.
                let rd = |lo: usize, hi: usize| -> Vec<u16> {
                    let mut v: Vec<u16> = (lo..hi)
                        .flat_map(|x| gates[tl[x].0].ctrls.iter().map(|&(w, _)| w))
                        .collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                };
                if rd(e1, e2) != rd(e3, e4) || (e2 - e1) != (e4 - e3) {
                    return None;
                }
                let mut m: Vec<usize> = (s..e4).map(|x| tl[x].0).collect();
                m.sort_unstable();
                m.dedup();
                Some((e4, m))
            })();
            match ok {
                Some((e4, m)) => {
                    groups.push(m);
                    s = e4;
                }
                None => s += 1,
            }
        }
    }

    println!("[ladmob] file={}", path);
    println!("[ladmob] gates={} T-bracket groups={} ({:.2}% of gates in a group)",
        n,
        groups.len(),
        100.0 * groups.iter().map(|g| g.len()).sum::<usize>() as f64 / n as f64);
    if groups.is_empty() {
        println!("[ladmob] no ladders detected (expected on ladder_cap 0 output)");
    }

    // --- sample groups ---------------------------------------------------
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..groups.len()).collect();
    if idx.len() > sample {
        for i in 0..sample {
            let j = rng.random_range(i..idx.len());
            idx.swap(i, j);
        }
        idx.truncate(sample);
    }

    let (mut rig, mut indiv, mut spans, mut sizes) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut frozen_chain, mut frozen4, mut n4) = (0usize, 0usize, 0usize);
    for &gi in &idx {
        let m = &groups[gi];
        let (l, r) = rigid_translate(&gates, m, cap);
        rig.push(l + r);
        spans.push(m[m.len() - 1] - m[0]);
        sizes.push(m.len());
        // Individual leeway of the members: the number leeway_by_width reports.
        let mut mi = 0usize;
        for &p in m {
            mi += leeway(&gates, p, cap);
        }
        indiv.push(mi / m.len());
        // Is the internal order frozen? Every consecutive pair colliding means
        // no commuting shuffle can ever permute the group.
        let chain = m.windows(2).all(|w| XGate::collides(&gates[w[0]], &gates[w[1]]));
        if chain {
            frozen_chain += 1;
        }
        // Size 4 = one gate per sweep stage (T, R, T, R): the pure chain, with
        // no within-stage commuting pair to dilute the test.
        if m.len() == 4 {
            n4 += 1;
            if chain {
                frozen4 += 1;
            }
        }
    }
    // Geometry null: same member count, same relative offsets, random
    // location. If a spread-out k-set is hard to move rigidly wherever you put
    // it, the ladder result is geometry, not structure.
    let mut null: Vec<usize> = Vec::new();
    for &gi in &idx {
        let m = &groups[gi];
        let span = m[m.len() - 1] - m[0];
        if span + 1 >= n {
            continue;
        }
        let s = rng.random_range(0..n - span - 1);
        let ctrl: Vec<usize> = m.iter().map(|&p| s + (p - m[0])).collect();
        let (l, r) = rigid_translate(&gates, &ctrl, cap);
        null.push(l + r);
    }
    if !rig.is_empty() {
        null.sort_unstable();
        println!(
            "[ladmob] NULL rigid, same geometry random location: mean={:.1} med={} p90={} zero={:.3}",
            mean(&null), pct(&null, 0.5), pct(&null, 0.9),
            null.iter().filter(|&&v| v == 0).count() as f64 / null.len().max(1) as f64
        );
        rig.sort_unstable();
        indiv.sort_unstable();
        spans.sort_unstable();
        sizes.sort_unstable();
        println!("[ladmob] sampled {} groups  size mean={:.2} med={}", rig.len(), mean(&sizes), pct(&sizes, 0.5));
        println!(
            "[ladmob] RIGID (whole group)   mean={:.1} med={} p90={} zero={:.3}",
            mean(&rig), pct(&rig, 0.5), pct(&rig, 0.9),
            rig.iter().filter(|&&v| v == 0).count() as f64 / rig.len() as f64
        );
        println!(
            "[ladmob] INDIV (member avg)    mean={:.1} med={} p90={}",
            mean(&indiv), pct(&indiv, 0.5), pct(&indiv, 0.9)
        );
        println!(
            "[ladmob] SPAN  (last - first)  mean={:.1} med={} p90={}",
            mean(&spans), pct(&spans, 0.5), pct(&spans, 0.9)
        );
        println!(
            "[ladmob] order-frozen chains (every consecutive member collides): {}/{} = {:.3}",
            frozen_chain, rig.len(), frozen_chain as f64 / rig.len() as f64
        );
        if n4 > 0 {
            println!(
                "[ladmob]   restricted to size-4 groups (one gate per sweep stage): {}/{} = {:.4}",
                frozen4, n4, frozen4 as f64 / n4 as f64
            );
        }
    }

    // --- in-circuit control: single gates, by width ------------------------
    // Same circuit, same length, no normalization needed: an UNLADDERED wide
    // gate against a ladder that lives beside it.
    // Sample per width rather than scanning all n gates at cap 4096.
    let mut by_w: Vec<Vec<usize>> = Vec::new();
    let mut tries = 0usize;
    while tries < sample * 200 {
        tries += 1;
        let i = rng.random_range(0..n);
        let w = gates[i].width().min(15);
        if by_w.len() <= w {
            by_w.resize_with(w + 1, Vec::new);
        }
        if by_w[w].len() >= sample {
            continue;
        }
        by_w[w].push(leeway(&gates, i, cap));
    }
    println!("[ladmob] single-gate leeway, same circuit (sampled with replacement):");
    for (w, v) in by_w.iter_mut().enumerate() {
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        println!(
            "[ladmob]   width {:>2}  n={:>6}  mean={:>8.1} med={:>6} p90={:>6}",
            w, v.len(), mean(v), pct(v, 0.5), pct(v, 0.9)
        );
    }

    if sweeps == 0 {
        return;
    }
    // --- tagged commuting shuffle: NET DISPLACEMENT by width ---------------
    // Gibbs remove-and-reinsert on the commutation class, the sampler
    // `commute_shuffle_exp` uses: pick a gate, find its nearest colliding
    // predecessor and successor in the CURRENT order, and relocate it
    // uniformly among the slots strictly between them. That resamples its
    // exact conditional, so the chain is uniform on the class and mixes in
    // ~m log m moves instead of the adjacent-transposition chain's ~m^3.
    // Semantics are preserved exactly; no merges, no replacements, so
    // displacement here is attributable to mobility alone.
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut pos: Vec<u32> = (0..n as u32).collect();
    let moves = sweeps * n;
    for _ in 0..moves {
        let j = rng.random_range(0..n);
        let g = perm[j];
        let gg = &gates[g as usize];
        let mut lo = j;
        while lo > 0 && j - lo < cap && !XGate::collides(gg, &gates[perm[lo - 1] as usize]) {
            lo -= 1;
        }
        let mut hi = j;
        while hi + 1 < n && hi - j < cap && !XGate::collides(gg, &gates[perm[hi + 1] as usize]) {
            hi += 1;
        }
        if lo == hi {
            continue;
        }
        let k = rng.random_range(lo..=hi);
        if k > j {
            perm.copy_within(j + 1..=k, j);
            perm[k] = g;
            for s in j..=k {
                pos[perm[s] as usize] = s as u32;
            }
        } else if k < j {
            perm.copy_within(k..j, k + 1);
            perm[k] = g;
            for s in k..=j {
                pos[perm[s] as usize] = s as u32;
            }
        }
    }
    let disp = |i: usize| -> usize { (pos[i] as i64 - i as i64).unsigned_abs() as usize };

    let mut dw: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let w = gates[i].width().min(15);
        if dw.len() <= w {
            dw.resize_with(w + 1, Vec::new);
        }
        dw[w].push(disp(i));
    }
    println!("[ladmob] --- {} sweeps of the tagged commuting shuffle ---", sweeps);
    println!("[ladmob] |net displacement| by width (ALL gates):");
    for (w, v) in dw.iter_mut().enumerate() {
        if v.is_empty() {
            continue;
        }
        v.sort_unstable();
        println!(
            "[ladmob]   width {:>2}  n={:>8}  mean={:>8.1} med={:>6} p90={:>6}",
            w, v.len(), mean(v), pct(v, 0.5), pct(v, 0.9)
        );
    }
    if idx.is_empty() {
        return;
    }
    // Ladder groups: how far does the BODY move, and does it stay a body?
    let (mut memb, mut cent, mut span0, mut span1) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut order_kept = 0usize;
    for &gi in &idx {
        let m = &groups[gi];
        let mut acc = 0usize;
        for &p in m {
            acc += disp(p);
        }
        memb.push(acc / m.len());
        let before: f64 = m.iter().map(|&p| p as f64).sum::<f64>() / m.len() as f64;
        let after: f64 = m.iter().map(|&p| pos[p] as f64).sum::<f64>() / m.len() as f64;
        cent.push((after - before).abs() as usize);
        span0.push(m[m.len() - 1] - m[0]);
        let np: Vec<u32> = m.iter().map(|&p| pos[p]).collect();
        span1.push((*np.iter().max().unwrap() - *np.iter().min().unwrap()) as usize);
        if np.windows(2).all(|w| w[0] < w[1]) {
            order_kept += 1;
        }
    }
    memb.sort_unstable();
    cent.sort_unstable();
    span0.sort_unstable();
    span1.sort_unstable();
    println!(
        "[ladmob] ladder MEMBER |disp| mean={:.1} med={}   ladder BODY (centroid) |disp| mean={:.1} med={}",
        mean(&memb), pct(&memb, 0.5), mean(&cent), pct(&cent, 0.5)
    );
    println!(
        "[ladmob] ladder span before mean={:.1} med={}   after mean={:.1} med={}",
        mean(&span0), pct(&span0, 0.5), mean(&span1), pct(&span1, 0.5)
    );
    println!(
        "[ladmob] ladder internal ORDER preserved after the shuffle: {}/{} = {:.4}",
        order_kept, idx.len(), order_kept as f64 / idx.len() as f64
    );
}
