//! Minimal-HALVES identity construction.
//!
//! The whole-identity test is empty: all 40,011,376 identities built from
//! arbitrary store pairs contain a compressible window, 86.8% of them a 4-gate
//! window collapsing to 2. The cause is that the regular store keeps every
//! spelling of a function, not only the shortest, so the halves are already
//! reducible and the identity inherits it.
//!
//! This filters the halves instead. A circuit is minimal when no window of it --
//! including the whole circuit, which catches "a shorter member of my own class
//! exists" -- has a strictly shorter verified equivalent. Identities are then
//! formed only from pairs of minimal circuits, so no window INSIDE either half
//! can compress by construction and only junction-straddling windows can.
//!
//! That is the operational reading of "A and B^-1 cannot be closely related by
//! local rewriting": the halves are irreducible, and the identity is kept only
//! if gluing them creates no new local shortening either.
//!
//! ```text
//! minimal_halves_probe FROZEN_REGULAR_DIR [--shards N] [--wires W]
//! ```

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::cancel_adjacent_duplicates;
use local_mixing::circuit::xgate::XGate;
use local_mixing::db_mixing::db_replace::{db_g57_to_xgate, db_probe};
use local_mixing::db_mixing::frozen::{FrozenDb, scan_shard};
use local_mixing::engine::rules::verify_rewrite;
use local_mixing::engine::xpoly::XPolyBudget;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::time::Instant;

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn class_members(value: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut position = 0usize;
    while position < value.len() {
        let len = value[position] as usize;
        position += 1;
        if len == 0 || len % 3 != 0 || position + len > value.len() {
            break;
        }
        out.push(value[position..position + len].to_vec());
        position += len;
    }
    out
}

fn simplify(mut identity: CircuitSeq) -> CircuitSeq {
    identity.canonicalize();
    cancel_adjacent_duplicates(&mut identity.gates, None::<&mut Vec<()>>);
    identity
}

/// Shortest verified equivalent found for any window in `lo..=hi` gate lengths.
/// `hi = n` includes the whole circuit; `hi = n - 1` restricts to proper
/// sub-windows, which is required for identities (an identity always reduces
/// as a whole, so including it would reject everything).
fn has_shorter_window(
    gates: &[XGate],
    hi: usize,
    db: &FrozenDb,
    budget: XPolyBudget,
    num_wires: usize,
    rng: &mut StdRng,
    probes: &mut u64,
) -> bool {
    let n = gates.len();
    for len in 2..=hi.min(n) {
        for at in 0..=(n - len) {
            let window = &gates[at..at + len];
            if local_mixing::engine::xpoly::xgate_used_wires(window).len() > 30 {
                continue;
            }
            *probes += 1;
            for (replacement, _, _) in db_probe(window, num_wires, db, budget, rng) {
                if replacement.len() < len && verify_rewrite(window, &replacement) {
                    return true;
                }
            }
        }
    }
    false
}

fn to_xgates(c: &CircuitSeq) -> Vec<XGate> {
    c.gates.iter().map(|&g| db_g57_to_xgate(g)).collect()
}

#[derive(Default)]
struct Stats {
    qualifying: u64,
    members: u64,
    minimal_members: u64,
    classes_with_2plus_minimal: u64,
    pairs: u64,
    identities: u64,
    surviving: u64,
    probes: u64,
    member_len: Vec<u64>,
    minimal_member_len: Vec<u64>,
    surviving_len: Vec<u64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            member_len: vec![0; 32],
            minimal_member_len: vec![0; 32],
            surviving_len: vec![0; 32],
            ..Default::default()
        }
    }
    fn merge(&mut self, o: Stats) {
        self.qualifying += o.qualifying;
        self.members += o.members;
        self.minimal_members += o.minimal_members;
        self.classes_with_2plus_minimal += o.classes_with_2plus_minimal;
        self.pairs += o.pairs;
        self.identities += o.identities;
        self.surviving += o.surviving;
        self.probes += o.probes;
        for i in 0..32 {
            self.member_len[i] += o.member_len[i];
            self.minimal_member_len[i] += o.minimal_member_len[i];
            self.surviving_len[i] += o.surviving_len[i];
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!("usage: minimal_halves_probe FROZEN_REGULAR_DIR [--shards N] [--wires W]");
        std::process::exit(2);
    };
    let shards: usize = flag(&args, "--shards", 256usize).clamp(1, 256);
    let wires: usize = flag(&args, "--wires", 32usize);

    let db = FrozenDb::open(dir, None);
    let budget = XPolyBudget::default();
    let start = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);

    let stats = (0..shards)
        .into_par_iter()
        .map(|shard| {
            let mut s = Stats::new();
            let mut rng = StdRng::seed_from_u64(0x_5eed_beef ^ shard as u64);
            scan_shard(dir, shard, &mut |value| {
                let members = class_members(value);
                if members.len() < 2 {
                    return;
                }
                s.qualifying += 1;
                // Test each member once, then pair only the minimal ones.
                let circuits: Vec<CircuitSeq> =
                    members.iter().map(|b| CircuitSeq::from_blob(b)).collect();
                let mut minimal: Vec<&CircuitSeq> = Vec::new();
                for c in &circuits {
                    s.members += 1;
                    s.member_len[c.gates.len().min(31)] += 1;
                    let g = to_xgates(c);
                    let n = g.len();
                    if !has_shorter_window(&g, n, &db, budget, wires, &mut rng, &mut s.probes) {
                        s.minimal_members += 1;
                        s.minimal_member_len[c.gates.len().min(31)] += 1;
                        minimal.push(c);
                    }
                }
                if minimal.len() < 2 {
                    return;
                }
                s.classes_with_2plus_minimal += 1;
                for left in 0..minimal.len() {
                    for right in 0..minimal.len() {
                        if left == right {
                            continue;
                        }
                        s.pairs += 1;
                        let a = minimal[left];
                        let b = minimal[right];
                        let mut reverse_b = b.gates.clone();
                        reverse_b.reverse();
                        let gates: Vec<[u16; 3]> =
                            a.gates.iter().copied().chain(reverse_b).collect();
                        let identity = simplify(CircuitSeq { gates });
                        if identity.gates.len() < 3 {
                            continue;
                        }
                        s.identities += 1;
                        let g = to_xgates(&identity);
                        let n = g.len();
                        // Proper sub-windows only: the whole thing is an identity.
                        if !has_shorter_window(
                            &g,
                            n - 1,
                            &db,
                            budget,
                            wires,
                            &mut rng,
                            &mut s.probes,
                        ) {
                            s.surviving += 1;
                            s.surviving_len[n.min(31)] += 1;
                        }
                    }
                }
            });
            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if k % 32 == 0 || k == shards {
                eprintln!(
                    "[halves] {k}/{shards} shards, {:.0}s",
                    start.elapsed().as_secs_f64()
                );
            }
            s
        })
        .reduce(Stats::new, |mut a, b| {
            a.merge(b);
            a
        });

    let e = start.elapsed().as_secs_f64();
    println!(
        "\nshards={shards} wires={wires} elapsed={e:.1}s probes={}",
        stats.probes
    );
    println!("qualifying-classes={}", stats.qualifying);
    println!(
        "members={}  MINIMAL-members={} ({:.4}%)",
        stats.members,
        stats.minimal_members,
        stats.minimal_members as f64 * 100.0 / stats.members.max(1) as f64
    );
    println!(
        "classes with >=2 minimal members={}",
        stats.classes_with_2plus_minimal
    );
    println!(
        "minimal-pairs={}  identities={}",
        stats.pairs, stats.identities
    );
    println!(
        "SURVIVING (no junction reduction)={} ({:.4}% of identities)",
        stats.surviving,
        stats.surviving as f64 * 100.0 / stats.identities.max(1) as f64
    );

    println!("\ncircuit length: all members vs minimal members");
    for i in 0..32 {
        if stats.member_len[i] > 0 {
            println!(
                "  n={i:<3} members={:>12} minimal={:>12} ({:.2}%)",
                stats.member_len[i],
                stats.minimal_member_len[i],
                stats.minimal_member_len[i] as f64 * 100.0 / stats.member_len[i] as f64
            );
        }
    }
    println!("\nsurviving identity lengths:");
    for i in 0..32 {
        if stats.surviving_len[i] > 0 {
            println!("  n={i:<3} {:>12}", stats.surviving_len[i]);
        }
    }
}
