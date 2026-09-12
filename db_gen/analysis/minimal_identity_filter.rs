//! Enumerate identities from the frozen REGULAR store and test each for
//! minimality under regular-DB compression.
//!
//! An identity is `a ++ reverse(b)` for two distinct circuits `a`, `b` in one
//! equivalence class (CCX gates are self-inverse, so this computes identity).
//! `scan_shard` yields values without keys, which is all identity construction
//! needs, and shards partition by key hash so any subset is an unbiased sample.
//!
//! MINIMALITY, as [`db_replace`]'s doc states it: the identity must not be
//! shortenable by local rewriting, because that is what makes a split's two
//! halves meaningfully different rather than trivial respellings. Tested here as
//! production would actually shorten it -- every contiguous window is probed
//! against the regular store through `db_probe`, the same path compression uses,
//! and a strictly shorter equivalent anywhere means not minimal. Commutation and
//! adjacent cancellation are applied first, so a spelling that collapses is
//! judged on its collapsed form.
//!
//! ```text
//! FROZEN_DB_DIR is not used; pass the regular dir explicitly.
//! minimal_identity_filter FROZEN_REGULAR_DIR [--shards N] [--max-identities M] [--wires W]
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
use std::collections::BTreeMap;
use std::time::Instant;

/// Per-shard tallies, merged after the parallel sweep.
#[derive(Default)]
struct Stats {
    classes: u64,
    qualifying: u64,
    spellings: u64,
    too_short: u64,
    collapsed: u64,
    identities: u64,
    minimal: u64,
    probes: u64,
    unverified: u64,
    equal_pairs: u64,
    minimal_equal: u64,
    edge_hits: u64,
    len_hist: Vec<u64>,
    minimal_len_hist: Vec<u64>,
    hit_hist: BTreeMap<(usize, usize), u64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            len_hist: vec![0; 64],
            minimal_len_hist: vec![0; 64],
            ..Default::default()
        }
    }

    fn merge(&mut self, other: Stats) {
        self.classes += other.classes;
        self.qualifying += other.qualifying;
        self.spellings += other.spellings;
        self.too_short += other.too_short;
        self.collapsed += other.collapsed;
        self.identities += other.identities;
        self.minimal += other.minimal;
        self.probes += other.probes;
        self.unverified += other.unverified;
        self.equal_pairs += other.equal_pairs;
        self.minimal_equal += other.minimal_equal;
        self.edge_hits += other.edge_hits;
        for (index, value) in other.len_hist.iter().enumerate() {
            self.len_hist[index] += value;
        }
        for (index, value) in other.minimal_len_hist.iter().enumerate() {
            self.minimal_len_hist[index] += value;
        }
        for (key, value) in other.hit_hist {
            *self.hit_hist.entry(key).or_insert(0) += value;
        }
    }
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Split a legacy `[len][blob]...` value into its member circuits.
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

/// True when some PROPER contiguous sub-window has a strictly shorter, verified
/// equivalent in the regular store -- i.e. local rewriting would shorten this
/// circuit.
///
/// Windows run to `n - 1`, never `n`. The full-circuit window is excluded on
/// purpose: an identity reduces to the empty circuit by definition, so probing
/// it always reports "compressible" and every identity would fail. Minimality is
/// about LOCAL moves, which is what makes a split's halves unrelated.
///
/// Every shorter candidate is confirmed with `verify_rewrite` before it counts.
/// The store can answer with a value that is not actually equivalent -- reversed
/// frames and value-convention mismatches both do this -- and an unverified hit
/// would wrongly condemn a minimal identity.
fn compressible(
    identity: &CircuitSeq,
    db: &FrozenDb,
    budget: XPolyBudget,
    num_wires: usize,
    rng: &mut StdRng,
    probes: &mut u64,
    unverified: &mut u64,
    hit: &mut Option<(usize, usize, usize)>,
) -> bool {
    let gates: Vec<XGate> = identity.gates.iter().map(|&g| db_g57_to_xgate(g)).collect();
    let n = gates.len();
    if n < 3 {
        return false;
    }
    for len in 2..n {
        for at in 0..=(n - len) {
            let window = &gates[at..at + len];
            if local_mixing::engine::xpoly::xgate_used_wires(window).len() > 30 {
                continue;
            }
            *probes += 1;
            for (replacement, _, _) in db_probe(window, num_wires, db, budget, rng) {
                if replacement.len() < len {
                    if verify_rewrite(window, &replacement) {
                        *hit = Some((len, replacement.len(), at));
                        return true;
                    }
                    *unverified += 1;
                }
            }
        }
    }
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!(
            "usage: minimal_identity_filter FROZEN_REGULAR_DIR [--shards N] [--max-identities M] [--wires W]"
        );
        std::process::exit(2);
    };
    let shards: usize = flag(&args, "--shards", 1usize).clamp(1, 256);
    let max_identities: u64 = flag(&args, "--max-identities", 0u64);
    let wires: usize = flag(&args, "--wires", 32usize);

    let db = FrozenDb::open(dir, None);
    let budget = XPolyBudget::default();
    let start = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);

    // Shards are independent: each owns its file and its tallies, so the sweep
    // parallelises cleanly and only the merge is shared.
    let stats = (0..shards)
        .into_par_iter()
        .map(|shard| {
            let mut s = Stats::new();
            // Seed per shard so a run stays reproducible regardless of thread order.
            let mut rng = StdRng::seed_from_u64(0x5eed ^ shard as u64);
            let mut stop = false;
            scan_shard(dir, shard, &mut |value| {
                if stop {
                    return;
                }
                s.classes += 1;
                let members = class_members(value);
                if members.len() < 2 {
                    return;
                }
                s.qualifying += 1;
                for left in 0..members.len() {
                    for right in 0..members.len() {
                        if left == right {
                            continue;
                        }
                        let a = CircuitSeq::from_blob(&members[left]);
                        let b = CircuitSeq::from_blob(&members[right]);
                        let mut reverse_a = a.gates.clone();
                        reverse_a.reverse();
                        let mut reverse_b = b.gates.clone();
                        reverse_b.reverse();
                        let candidates = [
                            a.gates.iter().copied().chain(reverse_b).collect::<Vec<_>>(),
                            reverse_a
                                .into_iter()
                                .chain(b.gates.iter().copied())
                                .collect::<Vec<_>>(),
                        ];
                        for gates in candidates {
                            s.spellings += 1;
                            let raw_len = gates.len();
                            let identity = simplify(CircuitSeq { gates });
                            if identity.gates.len() < raw_len {
                                s.collapsed += 1;
                            }
                            if identity.gates.len() < 3 {
                                s.too_short += 1;
                                continue;
                            }
                            s.identities += 1;
                            s.len_hist[identity.gates.len().min(63)] += 1;
                            let equal_halves = members[left].len() == members[right].len();
                            if equal_halves {
                                s.equal_pairs += 1;
                            }
                            let mut hit = None;
                            if !compressible(
                                &identity,
                                &db,
                                budget,
                                wires,
                                &mut rng,
                                &mut s.probes,
                                &mut s.unverified,
                                &mut hit,
                            ) {
                                s.minimal += 1;
                                s.minimal_len_hist[identity.gates.len().min(63)] += 1;
                                if equal_halves {
                                    s.minimal_equal += 1;
                                }
                            } else if let Some((wlen, rlen, at)) = hit {
                                *s.hit_hist.entry((wlen, rlen)).or_insert(0u64) += 1;
                                if at == 0 || at + wlen == identity.gates.len() {
                                    s.edge_hits += 1;
                                }
                            }
                            if max_identities > 0 && s.identities >= max_identities {
                                stop = true;
                                return;
                            }
                        }
                    }
                }
            });
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 16 == 0 || n == shards {
                eprintln!(
                    "[filter] {n}/{shards} shards, {:.0}s elapsed",
                    start.elapsed().as_secs_f64()
                );
            }
            s
        })
        .reduce(Stats::new, |mut a, b| {
            a.merge(b);
            a
        });

    let Stats {
        classes,
        qualifying,
        spellings,
        too_short,
        collapsed,
        identities,
        minimal,
        probes,
        unverified,
        equal_pairs,
        minimal_equal,
        edge_hits,
        len_hist,
        minimal_len_hist,
        hit_hist,
    } = stats;

    let elapsed = start.elapsed().as_secs_f64();
    println!("\nscanned-shards={shards} (of 256)  wires={wires}");
    println!("classes={classes}  qualifying={qualifying}");
    println!(
        "spellings={spellings}  collapsed-by-simplify={collapsed}  dropped(<3 gates)={too_short}"
    );
    println!("identities={identities}");
    println!(
        "MINIMAL={minimal} ({:.4}% of identities)",
        minimal as f64 * 100.0 / identities.max(1) as f64
    );
    println!("window-probes={probes}  shorter-but-NOT-equivalent={unverified}");
    println!(
        "from-equal-length-pairs={equal_pairs} (minimal among those: {minimal_equal})  first-hit-at-an-edge={edge_hits}"
    );
    println!("\nfirst compressing move (window length -> replacement length):");
    for ((wlen, rlen), count) in &hit_hist {
        println!("  {wlen:>3} -> {rlen:<3}  {count:>12}");
    }
    println!(
        "elapsed={elapsed:.1}s  identities/s={:.0}  probes/s={:.0}",
        identities as f64 / elapsed.max(1e-9),
        probes as f64 / elapsed.max(1e-9)
    );

    println!("\nidentity length: all vs minimal");
    for len in 0..64 {
        if len_hist[len] > 0 {
            println!(
                "  n={len:<3} all={:>12} minimal={:>12} ({:.2}%)",
                len_hist[len],
                minimal_len_hist[len],
                minimal_len_hist[len] as f64 * 100.0 / len_hist[len] as f64
            );
        }
    }

    if shards < 256 && identities > 0 && max_identities == 0 {
        let scale = 256.0 / shards as f64;
        println!("\n=== full-store projection (x{scale:.0}) ===");
        println!("identities {:>18.0}", identities as f64 * scale);
        println!("minimal    {:>18.0}", minimal as f64 * scale);
        println!(
            "single-thread hours {:>10.2}  (at {:.0} identities/s)",
            (identities as f64 * scale) / (identities as f64 / elapsed.max(1e-9)) / 3600.0,
            identities as f64 / elapsed.max(1e-9)
        );
    }
}
