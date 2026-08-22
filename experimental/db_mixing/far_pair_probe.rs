//! far_pair_probe — store-coverage probe for the pair geometry
//! (docs/NONLOCAL_PHASE_A.md).
//!
//! Samples far commuting pairs from a circuit the way `collect_pair` would
//! fuse them (seed, rightward commutation-box scan to the first collider,
//! farthest partner), keys each fused 2-gate window, and enumerates EVERY
//! stored candidate from both stores via `db_probe`. Reports hit rates, the
//! free/pay size mix, and the ENTANGLED-candidate fraction: candidates whose
//! wire-sharing graph connects the two gates' wire blocks, i.e. spellings
//! that syntactically couple regions the pair never coupled functionally —
//! the quantity that decides whether disjoint-pair fusion buys cross-block
//! structure or only litter-union transport.
//!
//! Requires FROZEN_DB_DIR (and FROZEN_CURATED_DIR for the curated side),
//! exactly as fmix does. Run it on fleet phase-A material; the store does not
//! work over slow mounts.

use clap::Parser;
use local_mixing::db_mixing::db_replace::db_probe;
use local_mixing::engine::format::{read_g57_file, read_mpmct};
use local_mixing::circuit::xgate::{XGate, max_wire};
use local_mixing::engine::xpoly::XPolyBudget;
use local_mixing::db_mixing::frozen::FrozenDb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser)]
struct Args {
    /// Circuit to sample pairs from.
    #[arg(long)]
    input: String,
    /// Input format: mpmct1 or g57.
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Pairs to sample.
    #[arg(long, default_value_t = 20_000)]
    samples: usize,
    /// Commutation-box scan cap (gates examined past the seed).
    #[arg(long, default_value_t = 4096)]
    scan_cap: usize,
    /// Pick the partner uniformly from the box instead of the farthest gate.
    #[arg(long, default_value_t = false)]
    uniform: bool,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

/// Union wires within each gate (a gate is a hyperedge over its wire set);
/// afterwards two wires share a root iff the candidate's gates connect them.
struct WireUf {
    parent: Vec<u16>,
}

impl WireUf {
    fn new(n: usize) -> Self {
        Self { parent: (0..n as u16).collect() }
    }
    fn find(&mut self, x: u16) -> u16 {
        let p = self.parent[x as usize];
        if p == x {
            return x;
        }
        let r = self.find(p);
        self.parent[x as usize] = r;
        r
    }
    fn union(&mut self, a: u16, b: u16) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

fn gate_wires(g: &XGate) -> Vec<u16> {
    std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)).collect()
}

/// Does the candidate connect block A to block B through shared wires
/// (scratch wires included)?
fn entangles(cand: &[XGate], a: &[u16], b: &[u16], num_wires: usize) -> bool {
    let mut uf = WireUf::new(num_wires);
    for g in cand {
        let ws = gate_wires(g);
        for w in &ws[1..] {
            uf.union(ws[0], *w);
        }
    }
    let roots_a: Vec<u16> = a.iter().map(|&w| uf.find(w)).collect();
    b.iter().any(|&w| {
        let r = uf.find(w);
        roots_a.contains(&r)
    })
}

/// Candidate is a permutation of the window (identity/reorder — the trivial
/// spellings the fmix bans refuse).
fn is_trivial(cand: &[XGate], window: &[XGate]) -> bool {
    if cand.len() != window.len() {
        return false;
    }
    let mut used = vec![false; window.len()];
    'outer: for g in cand {
        for (i, h) in window.iter().enumerate() {
            if !used[i] && g == h {
                used[i] = true;
                continue 'outer;
            }
        }
        return false;
    }
    true
}

fn main() {
    let args = Args::parse();
    let gates: Vec<XGate> = match args.g_format.as_str() {
        "g57" => read_g57_file(&args.input).expect("read input (g57)"),
        "mpmct1" => read_mpmct(&args.input).expect("read input (mpmct1)").0,
        o => panic!("unknown --g-format {o}"),
    };
    let n = gates.len();
    assert!(n >= 2, "need at least two gates");
    let num_wires = max_wire(&gates) as usize + 1;
    let db = FrozenDb::from_env();
    let budget = XPolyBudget::default();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let mut empty_boxes = 0u64;
    let mut truncs = 0u64;
    let mut box_sum = 0u64;
    let mut box_max = 0u64;
    let mut windows = 0u64;
    let mut overlap_windows = 0u64;
    let mut hits = 0u64;
    let mut hits_curated = 0u64;
    let mut cand_total = 0u64;
    let mut cand_free = 0u64;
    let mut cand_trivial = 0u64;
    let mut cand_entangled = 0u64;
    let mut disjoint_windows = 0u64;
    let mut disjoint_hits = 0u64;
    let mut disjoint_any_entangled = 0u64;
    let mut len_hist = [0u64; 16];

    for _ in 0..args.samples {
        let i = rng.random_range(0..n);
        // collect_pair scans in the SEED's stored direction, which is L on
        // ~half of real seeds (and then emits the window as [partner, seed]).
        // A serialized circuit carries no per-gate direction, so draw the scan
        // direction 50/50 to match collect_pair's aggregate L/R window mix
        // rather than biasing toward right-anchored pairs.
        let go_right = rng.random_bool(0.5);
        let mut cands: Vec<usize> = Vec::new();
        let mut hops_scanned = 0usize;
        if go_right {
            let mut j = i + 1;
            while j < n && hops_scanned < args.scan_cap && !XGate::collides(&gates[i], &gates[j]) {
                cands.push(j);
                j += 1;
                hops_scanned += 1;
            }
            if j < n && hops_scanned >= args.scan_cap {
                truncs += 1;
            }
        } else {
            let mut j = i;
            while j > 0 && hops_scanned < args.scan_cap && !XGate::collides(&gates[i], &gates[j - 1]) {
                cands.push(j - 1);
                j -= 1;
                hops_scanned += 1;
            }
            if j > 0 && hops_scanned >= args.scan_cap {
                truncs += 1;
            }
        }
        let Some(&far) = cands.last() else {
            empty_boxes += 1;
            continue;
        };
        let pick = if args.uniform { cands[rng.random_range(0..cands.len())] } else { far };
        let hops = i.abs_diff(pick) as u64 - 1;
        box_sum += hops;
        box_max = box_max.max(hops);

        // Window in link order (leftmost first), mirroring collect_pair's
        // Dir::L order [partner, seed] on a leftward scan.
        let (lo, hi) = if pick < i { (pick, i) } else { (i, pick) };
        let window = vec![gates[lo].clone(), gates[hi].clone()];
        windows += 1;
        let wa = gate_wires(&window[0]);
        let wb = gate_wires(&window[1]);
        let disjoint = wa.iter().all(|w| !wb.contains(w));
        if disjoint {
            disjoint_windows += 1;
        } else {
            overlap_windows += 1;
        }

        let found = db_probe(&window, num_wires, &db, budget, &mut rng);
        if found.is_empty() {
            continue;
        }
        hits += 1;
        if found.iter().any(|(_, cur, _)| *cur) {
            hits_curated += 1;
        }
        if disjoint {
            disjoint_hits += 1;
        }
        let mut any_ent = false;
        for (cand, _cur, _rev) in &found {
            cand_total += 1;
            len_hist[cand.len().min(15)] += 1;
            if cand.len() <= 2 {
                cand_free += 1;
            }
            if is_trivial(cand, &window) {
                cand_trivial += 1;
                continue;
            }
            if disjoint && entangles(cand, &wa, &wb, num_wires) {
                cand_entangled += 1;
                any_ent = true;
            }
        }
        if disjoint && any_ent {
            disjoint_any_entangled += 1;
        }
    }

    let fused = windows.max(1) as f64;
    println!(
        "[far_pair_probe] samples={} windows={} empty={} trunc={} box avg={:.1} max={}",
        args.samples, windows, empty_boxes, truncs, box_sum as f64 / fused, box_max
    );
    println!(
        "[far_pair_probe] hits={} ({:.1}%) curated={} | disjoint windows={} ({:.1}%) hits={} any-entangled={} ({:.1}% of disjoint hits)",
        hits,
        100.0 * hits as f64 / fused,
        hits_curated,
        disjoint_windows,
        100.0 * disjoint_windows as f64 / fused,
        disjoint_hits,
        disjoint_any_entangled,
        100.0 * disjoint_any_entangled as f64 / disjoint_hits.max(1) as f64,
    );
    println!(
        "[far_pair_probe] candidates={} free(<=2)={} trivial={} entangled={} ({:.1}% of candidates on disjoint hits)",
        cand_total,
        cand_free,
        cand_trivial,
        cand_entangled,
        100.0 * cand_entangled as f64 / cand_total.max(1) as f64,
    );
    let hist: Vec<String> =
        len_hist.iter().enumerate().filter(|(_, c)| **c > 0).map(|(l, c)| format!("{l}:{c}")).collect();
    println!("[far_pair_probe] candidate gate-count histogram: {}", hist.join(" "));
}
