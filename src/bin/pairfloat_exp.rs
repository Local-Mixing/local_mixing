//! Pair-insert-and-float experiment: before each gate of a random g57 circuit
//! C, insert k identical gate pairs (g·g = identity); then float each pair's
//! two members in opposite directions, stopping at the circuit end or at a
//! collider they can no longer pass. A separated pair brackets the interval
//! between its members: every cut inside reads a nonlinearly re-encoded state
//! (the bracket target at degree 1 + #controls). Equivalence stays exact and
//! is sample-verified per output.
//!
//! Modes (arg 6):
//!   g57      — comp g57 pair gates, commuting swaps only (the original run)
//!   conj     — non-comp conjunction pairs (2 ctrls, random polarities)
//!   adaptive — conj pairs whose wires are chosen for MOBILITY: target from
//!              the quietest wires (longest support-free stretch around the
//!              cut), controls from wires no nearby gate targets; sampled
//!              from the top few to avoid neighbor pile-up.
//!   gaptile  — per-WIRE, not per-cut: for each wire w, each maximal stretch
//!              of C between consecutive touches of w is a bracket slot;
//!              insert `k` (= stacking s) distinct conj pairs targeting w
//!              there, controls drawn from wires untargeted in that gap (so
//!              the pair commutes with the whole interior), floated apart to
//!              the gap ends by commuting swaps. Forced diversity over all
//!              wires; every wire masked at every cut but its own use points.
//! Cross budget B (arg 7, conj/adaptive only): on collision, spend one unit
//! to cross via the exact splitting rules (postmix::rules R1/R2/R3, g57
//! colliders pre-split first); the lead shot piece floats on, other pieces
//! stay where they land. B=0 = commuting swaps only.
//!
//! Usage: pairfloat_exp <outbase> [n=128] [m=3000] [ks=5] [seed=1]
//!                      [mode=g57] [budget=0]
//! `ks`: int K (run k=1..=K) or comma list ("8,11,16,24"). Writes
//! <outbase>.source_c.g57, <outbase>.k0.mpmct1 (= C) and <outbase>.k<k>.mpmct1.

use local_mixing::postmix::format::write_mpmct;
use local_mixing::postmix::rules::{cross, presplit, Outcome, Role};
use local_mixing::postmix::xgate::{XGate, eval_u1024};
use local_mixing::circuit::circuit::U1024;
use local_mixing::random::random_data::random_circuit;
use rand::SeedableRng;
use rand::rngs::StdRng;

const K_MAX: usize = 12;
const SCAN_CAP: usize = 1500;

fn mask_bits(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn pick_distinct(n: usize, count: usize, excl: &[u16]) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(count);
    while out.len() < count {
        let v = fastrand::usize(..n) as u16;
        if !out.contains(&v) && !excl.contains(&v) {
            out.push(v);
        }
    }
    out
}

fn random_g57_xgate(n: usize) -> XGate {
    let w = pick_distinct(n, 3, &[]);
    XGate::from_g57([w[0], w[1], w[2]])
}

fn random_conj_xgate(n: usize) -> XGate {
    let w = pick_distinct(n, 3, &[]);
    XGate::conj(w[0], [(w[1], fastrand::bool()), (w[2], fastrand::bool())])
        .expect("distinct wires cannot contradict")
}

// Per-wire occurrence index over the BASE circuit, for adaptive wire choice.
struct WireIndex {
    supp: Vec<Vec<u32>>, // positions where wire is in a gate's support
    tgt: Vec<Vec<u32>>,  // positions where wire is a gate's target
}

impl WireIndex {
    fn build(base: &[XGate], n: usize) -> WireIndex {
        let mut supp = vec![Vec::new(); n];
        let mut tgt = vec![Vec::new(); n];
        for (p, g) in base.iter().enumerate() {
            tgt[g.target as usize].push(p as u32);
            supp[g.target as usize].push(p as u32);
            for &(w, _) in g.ctrls.iter() {
                supp[w as usize].push(p as u32);
            }
        }
        WireIndex { supp, tgt }
    }

    // Gap around cut i (insertion is BEFORE base gate i): distance leftward to
    // the nearest listed position < i plus distance rightward to the nearest
    // >= i, both capped.
    fn gap(list: &[u32], i: usize) -> usize {
        let k = list.partition_point(|&p| (p as usize) < i);
        let dl = if k == 0 { SCAN_CAP } else { (i - list[k - 1] as usize).min(SCAN_CAP) };
        let dr = if k == list.len() { SCAN_CAP } else { (list[k] as usize - i + 1).min(SCAN_CAP) };
        dl + dr
    }
}

// Adaptive conj gate at cut i: target sampled from the top-8 support-quiet
// wires (excluding this cut's earlier pair targets), controls from the top-16
// target-quiet wires. Sampling from the top few (not argmax) keeps adjacent
// cuts from piling onto the same quiet wire and blocking each other.
fn adaptive_conj_xgate(idx: &WireIndex, n: usize, i: usize, used_targets: &[u16]) -> XGate {
    let mut t_scores: Vec<(usize, u16)> = (0..n as u16)
        .filter(|w| !used_targets.contains(w))
        .map(|w| (WireIndex::gap(&idx.supp[w as usize], i), w))
        .collect();
    t_scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let t = t_scores[fastrand::usize(..t_scores.len().min(8))].1;

    let mut c_scores: Vec<(usize, u16)> = (0..n as u16)
        .filter(|&w| w != t)
        .map(|w| (WireIndex::gap(&idx.tgt[w as usize], i), w))
        .collect();
    c_scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let top = c_scores.len().min(16);
    let a = fastrand::usize(..top);
    let b = loop {
        let b = fastrand::usize(..top);
        if b != a {
            break b;
        }
    };
    XGate::conj(t, [(c_scores[a].1, fastrand::bool()), (c_scores[b].1, fastrand::bool())])
        .expect("distinct wires cannot contradict")
}

#[derive(Default)]
struct FloatStats {
    swaps: u64,
    crossings: u64,
    presplits: u64,
    blocked: u64,
    died: u64,
    end_stops: u64,
    coll_stops: u64,
    consumed: u64,
}

// Float the gate carrying `id` at `pos` to its extreme. Returns final pos.
#[allow(clippy::too_many_arguments)]
fn float_one(
    gates: &mut Vec<XGate>,
    cell_id: &mut Vec<u32>,
    id: u32,
    mut pos: usize,
    right: bool,
    mut budget: usize,
    rng: &mut StdRng,
    st: &mut FloatStats,
) -> usize {
    loop {
        let next = if right {
            if pos + 1 >= gates.len() {
                st.end_stops += 1;
                return pos;
            }
            pos + 1
        } else {
            if pos == 0 {
                st.end_stops += 1;
                return pos;
            }
            pos - 1
        };
        let g = gates[pos].clone();
        let h = gates[next].clone();
        if !XGate::collides(&g, &h) {
            gates.swap(pos, next);
            cell_id.swap(pos, next);
            st.swaps += 1;
            pos = next;
            continue;
        }
        if budget == 0 {
            st.coll_stops += 1;
            return pos;
        }
        match cross(&g, &h, K_MAX, rng) {
            Outcome::R0Swap => {
                gates.swap(pos, next);
                cell_id.swap(pos, next);
                st.swaps += 1;
                pos = next;
            }
            Outcome::PresplitColliding => {
                let pieces = presplit(&h, rng);
                let plen = pieces.len();
                if cell_id[next] != u32::MAX {
                    st.consumed += 1;
                }
                gates.splice(next..next + 1, pieces);
                cell_id.splice(next..next + 1, std::iter::repeat(u32::MAX).take(plen));
                st.presplits += 1;
                if !right {
                    // cells [0..next+plen) shifted our slot rightward
                    pos += plen - 1;
                }
                // retry against the now-adjacent piece
            }
            Outcome::Rewrite { seq, .. } => {
                let start = pos.min(next);
                let mut cells: Vec<(XGate, Role)> = seq;
                if !right {
                    cells.reverse();
                }
                let lead = if right {
                    cells.iter().rposition(|(_, r)| *r == Role::ShotPiece)
                } else {
                    cells.iter().position(|(_, r)| *r == Role::ShotPiece)
                };
                let ids: Vec<u32> = cells
                    .iter()
                    .enumerate()
                    .map(|(j, _)| if Some(j) == lead { id } else { u32::MAX })
                    .collect();
                let new_gates: Vec<XGate> = cells.into_iter().map(|(g, _)| g).collect();
                gates.splice(start..start + 2, new_gates);
                cell_id.splice(start..start + 2, ids);
                st.crossings += 1;
                budget -= 1;
                match lead {
                    Some(j) => pos = start + j,
                    None => {
                        st.died += 1;
                        return start;
                    }
                }
            }
            Outcome::Blocked(_) => {
                st.blocked += 1;
                return pos;
            }
        }
    }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let out = a.next().expect("usage: pairfloat_exp <outbase> [n m ks seed mode budget]");
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let ks_arg = a.next().unwrap_or_else(|| "5".to_string());
    let ks: Vec<usize> = if ks_arg.contains(',') {
        ks_arg.split(',').map(|s| s.trim().parse().expect("bad k list")).collect()
    } else {
        (1..=ks_arg.parse::<usize>().expect("bad k")).collect()
    };
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mode = a.next().unwrap_or_else(|| "g57".to_string());
    let budget: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    assert!(
        matches!(mode.as_str(), "g57" | "conj" | "adaptive" | "gaptile"),
        "mode must be g57|conj|adaptive|gaptile"
    );
    assert!(
        budget == 0 || mode != "g57",
        "cross budget needs conjunction shots (mode conj|adaptive)"
    );
    println!("[pf] n={n} m={m} ks={ks:?} seed={seed} mode={mode} budget={budget}");

    // Fixed C across all k and all modes (fastrand-seeded).
    fastrand::seed(seed);
    let c = random_circuit(n, m);
    std::fs::write(format!("{out}.source_c.g57"), c.repr()).expect("write source C");
    let base: Vec<XGate> = c.gates.iter().map(|&g| XGate::from_g57(g)).collect();
    println!("[pf] wrote source C ({m} g57 gates) to {out}.source_c.g57");
    write_mpmct(&format!("{out}.k0.mpmct1"), &base, n).expect("write k0");
    let widx = WireIndex::build(&base, n);

    for &k in &ks {
        fastrand::seed(seed ^ (0xF10A7 + k as u64));
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0553D ^ ((k as u64) << 20) ^ budget as u64);

        let mut gates: Vec<XGate> = Vec::new();
        let mut cell_id: Vec<u32> = Vec::new();
        let mut dir: Vec<bool> = Vec::new();
        let mut skipped_gaps = 0u64;
        if mode == "gaptile" {
            // Per-wire gap enumeration. `k` is the stacking count s.
            // pre[i] = pairs co-located just before base gate i; pre[m] = tail.
            let mut pre: Vec<Vec<(XGate, bool)>> = vec![Vec::new(); m + 1];
            for w in 0..n as u16 {
                let mut pts: Vec<i64> = vec![-1];
                pts.extend(widx.supp[w as usize].iter().map(|&p| p as i64));
                pts.push(m as i64);
                for win in pts.windows(2) {
                    let (a, b) = (win[0], win[1]);
                    if b - a < 2 {
                        continue; // no interior cut to mask
                    }
                    let (lo, hi) = ((a + 1) as usize, (b - 1) as usize);
                    let mut targeted = vec![false; n];
                    for p in lo..=hi {
                        targeted[base[p].target as usize] = true;
                    }
                    let cands: Vec<u16> = (0..n as u16)
                        .filter(|&x| x != w && !targeted[x as usize])
                        .collect();
                    if cands.len() < 2 {
                        skipped_gaps += 1;
                        continue;
                    }
                    for _ in 0..k {
                        let i1 = fastrand::usize(..cands.len());
                        let mut i2 = fastrand::usize(..cands.len());
                        while i2 == i1 {
                            i2 = fastrand::usize(..cands.len());
                        }
                        let g = XGate::conj(
                            w,
                            [(cands[i1], fastrand::bool()), (cands[i2], fastrand::bool())],
                        )
                        .expect("distinct wires cannot contradict");
                        pre[lo].push((g.clone(), false)); // float left
                        pre[lo].push((g, true)); // float right
                    }
                }
            }
            for i in 0..m {
                for (g, d) in pre[i].drain(..) {
                    cell_id.push(dir.len() as u32);
                    dir.push(d);
                    gates.push(g);
                }
                cell_id.push(u32::MAX);
                gates.push(base[i].clone());
            }
            for (g, d) in pre[m].drain(..) {
                cell_id.push(dir.len() as u32);
                dir.push(d);
                gates.push(g);
            }
        } else {
            for i in 0..m {
                let mut used_t: Vec<u16> = Vec::with_capacity(k);
                for _ in 0..k {
                    let g = match mode.as_str() {
                        "g57" => random_g57_xgate(n),
                        "conj" => random_conj_xgate(n),
                        _ => adaptive_conj_xgate(&widx, n, i, &used_t),
                    };
                    used_t.push(g.target);
                    for d in [false, true] {
                        cell_id.push(dir.len() as u32);
                        dir.push(d);
                        gates.push(g.clone());
                    }
                }
                cell_id.push(u32::MAX);
                gates.push(base[i].clone());
            }
        }
        let total = gates.len();

        // Float in random order, each to completion. Positions found by scan
        // (splices invalidate any cached index); a floater consumed by a
        // rewrite or presplit is simply never found.
        let nf = dir.len();
        let mut order: Vec<u32> = (0..nf as u32).collect();
        fastrand::shuffle(&mut order);
        let mut st = FloatStats::default();
        let (mut travel_sum, mut travel_max, mut floated) = (0u64, 0usize, 0u64);
        for &id in &order {
            let Some(pos) = cell_id.iter().position(|&c| c == id) else {
                continue; // consumed by an earlier rewrite
            };
            let start = pos;
            let end = float_one(&mut gates, &mut cell_id, id, pos, dir[id as usize], budget, &mut rng, &mut st);
            let t = if end >= start { end - start } else { start - end };
            travel_sum += t as u64;
            travel_max = travel_max.max(t);
            floated += 1;
        }

        // Exact-equivalence spot check against C.
        let low = mask_bits(n);
        let mut vrng = 0x5EEDu64 ^ seed ^ (k as u64) << 32;
        let mut bytes = [0u8; 128];
        for i in 0..200 {
            for b in bytes.iter_mut() {
                vrng ^= vrng << 13;
                vrng ^= vrng >> 7;
                vrng ^= vrng << 17;
                *b = (vrng >> 24) as u8;
            }
            let input = U1024::from_little_endian(&bytes) & low;
            let got = eval_u1024(&gates, input) & low;
            let want = eval_u1024(&base, input) & low;
            assert_eq!(got, want, "k={k}: floated != C on sample {i}");
        }

        let path = format!("{out}.k{k}.mpmct1");
        write_mpmct(&path, &gates, n).expect("write mpmct1");
        println!(
            "[pf] k={k}: {} gates (from {total}; {floated}/{nf} floated), travel mean {:.1} max {travel_max}, \
             swaps {} cross {} presplit {} | stops: {} coll / {} end / {} blocked / {} died / {} consumed | skipgap {}; \
             verify PASSED (200 samples); wrote {path}",
            gates.len(),
            travel_sum as f64 / floated.max(1) as f64,
            st.swaps,
            st.crossings,
            st.presplits,
            st.coll_stops,
            st.end_stops,
            st.blocked,
            st.died,
            st.consumed,
            skipped_gaps
        );
    }
}
