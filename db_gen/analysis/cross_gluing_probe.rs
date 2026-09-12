//! Cross-class one-gate gluing: manufacture 13-17 gate identities from the
//! frozen regular store.
//!
//! The store's own identity universe stops at 12 gates, so 1-gate keys never
//! see candidates past 11 gates. This probe tests the fix: for a member `a`
//! (function F, canonical frame) and a gate `g`, the class of canon(a ++ [g])
//! holds spellings of g∘F; any member `b` of it gives
//!
//!     rev(a) ++ b        computing exactly g          (|a|+|b| gates)
//!     rev(a) ++ b ++ [g] an identity                  (|a|+|b|+1 gates)
//!
//! i.e. genuine 13-17 gate identities glued from existing content, whose
//! 1-gate-deleted arcs are the 12-16 gate m1 candidates asked for. Every glue
//! is verified by canonical-key equality before it counts, and a sample of
//! emitted m1 candidates is re-validated against the canonical single-CCX key.
//!
//! ```text
//! cross_gluing_probe FROZEN_REGULAR_DIR [--shards N] [--max-sources M]
//!     [--six-gate-sample S] [--partners-per-hit P]
//!     [--partner-composite COMPOSITE_ROCKS]
//! ```
//!
//! `--partner-composite` draws partners from a curated composite store
//! instead of the regular store. Regular-store partners are mostly the
//! source's own extension respelled, so the glue collapses; curated
//! candidates come from unrelated identity arcs and keep their length.

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::cancel_adjacent_duplicates;
use local_mixing::db_generation::curated_full::split_composite_key;
use local_mixing::db_mixing::frozen::{FrozenDb, scan_shard};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_LEN: usize = 64;

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

fn simplify(mut c: CircuitSeq) -> CircuitSeq {
    c.canonicalize();
    cancel_adjacent_duplicates(&mut c.gates, None::<&mut Vec<()>>);
    c
}

fn canon_key(c: &CircuitSeq) -> Option<[u8; 16]> {
    let (key, _, _) = c.canonicalize_polys_single_hashed(false);
    key
}

#[derive(Default)]
struct Stats {
    sources: u64,
    gates_tried: u64,
    lookups: u64,
    hits: u64,
    partners: u64,
    glued: u64,
    verified: u64,
    id_len: Vec<u64>,
    m1_len: Vec<u64>,
    m1_validated: u64,
    m1_validation_failures: u64,
}

impl Stats {
    fn new() -> Self {
        Self {
            id_len: vec![0; MAX_LEN],
            m1_len: vec![0; MAX_LEN],
            ..Default::default()
        }
    }
    fn merge(&mut self, o: Stats) {
        self.sources += o.sources;
        self.gates_tried += o.gates_tried;
        self.lookups += o.lookups;
        self.hits += o.hits;
        self.partners += o.partners;
        self.glued += o.glued;
        self.verified += o.verified;
        self.m1_validated += o.m1_validated;
        self.m1_validation_failures += o.m1_validation_failures;
        for i in 0..MAX_LEN {
            self.id_len[i] += o.id_len[i];
            self.m1_len[i] += o.m1_len[i];
        }
    }
}

/// All two-control g57 gates over `wires + extra_fresh` wires that touch at
/// least one wire below `wires`. Ordered control pairs: [t, x, y] fires on
/// (NOT x) AND y, so both polarities are distinct gates.
fn candidate_gates(wires: u16, extra_fresh: u16) -> Vec<[u16; 3]> {
    let total = wires + extra_fresh;
    let mut out = Vec::new();
    for t in 0..total {
        for c1 in 0..total {
            for c2 in 0..total {
                if c1 == c2 || t == c1 || t == c2 {
                    continue;
                }
                if t < wires || c1 < wires || c2 < wires {
                    out.push([t, c1, c2]);
                }
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!(
            "usage: cross_gluing_probe FROZEN_REGULAR_DIR [--shards N] [--max-sources M] [--six-gate-sample S] [--partners-per-hit P]"
        );
        std::process::exit(2);
    };
    let shards: usize = flag(&args, "--shards", 4usize).clamp(1, 256);
    let max_sources: usize = flag(&args, "--max-sources", 2000usize);
    let six_sample: usize = flag(&args, "--six-gate-sample", 500usize);
    let partners_per_hit: usize = flag(&args, "--partners-per-hit", 4usize);
    let partner_composite = args
        .iter()
        .position(|a| a == "--partner-composite")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let db = FrozenDb::open(dir, None);
    let composite: Option<rocksdb::DB> = partner_composite.as_deref().map(|path| {
        let mut options = rocksdb::Options::default();
        options.create_if_missing(false);
        options.set_compression_type(rocksdb::DBCompressionType::Zstd);
        options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
        rocksdb::DB::open_for_read_only(&options, path, false)
            .expect("open partner composite store")
    });
    let start = Instant::now();

    // Source reservoir: every 7-11 gate member seen, plus a sample of 6-gate
    // members, all in their class's canonical frame.
    let mut long_sources: Vec<Vec<u8>> = Vec::new();
    let mut six_sources: Vec<Vec<u8>> = Vec::new();
    let mut rng = StdRng::seed_from_u64(0x91_De);
    let mut six_seen = 0u64;
    for shard in 0..shards {
        scan_shard(dir, shard, &mut |value| {
            for member in class_members(value) {
                let gates = member.len() / 3;
                if (7..=11).contains(&gates) {
                    if long_sources.len() < max_sources {
                        long_sources.push(member);
                    }
                } else if gates == 6 {
                    six_seen += 1;
                    if six_sources.len() < six_sample {
                        six_sources.push(member);
                    } else {
                        let j = rng.random_range(0..six_seen);
                        if (j as usize) < six_sample {
                            six_sources[j as usize] = member;
                        }
                    }
                }
            }
        });
        if long_sources.len() >= max_sources {
            break;
        }
    }
    eprintln!(
        "[glue] sources: {} long (7-11g) + {} six-gate, {:.0}s",
        long_sources.len(),
        six_sources.len(),
        start.elapsed().as_secs_f64()
    );
    let sources: Vec<Vec<u8>> = long_sources.into_iter().chain(six_sources).collect();

    let toffoli_key = canon_key(&CircuitSeq {
        gates: vec![[0, 1, 2]],
    })
    .expect("single CCX must canonicalize");
    eprintln!(
        "[glue] canonical 1-gate key = {}",
        toffoli_key
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let done = AtomicU64::new(0);
    let total = Mutex::new(Stats::new());
    sources.par_iter().for_each(|blob| {
        let mut s = Stats::new();
        let mut rng = StdRng::seed_from_u64(0x91ce ^ blob.len() as u64);
        let a = CircuitSeq::from_blob(blob);
        let wires = a.gates.iter().flatten().copied().max().unwrap_or(0) + 1;
        s.sources += 1;
        for g in candidate_gates(wires, 2) {
            s.gates_tried += 1;
            let mut a_plus = a.clone();
            a_plus.gates.push(g);
            let (key2, perm2, used2) = a_plus.canonicalize_polys_single_hashed(false);
            let Some(key2) = key2 else { continue };
            s.lookups += 1;
            let raw_partners: Vec<Vec<u8>> = match &composite {
                None => match db.get_regular(&key2) {
                    Some(value) => class_members(&value),
                    None => continue,
                },
                Some(store) => {
                    // First 24 same-key candidates of >=4 gates in store
                    // order; enough spread for a probe.
                    let mut found = Vec::new();
                    for item in store.iterator(rocksdb::IteratorMode::From(
                        &key2,
                        rocksdb::Direction::Forward,
                    )) {
                        let Ok((record, _)) = item else { break };
                        if record.len() < 16 || record[..16] != key2[..] {
                            break;
                        }
                        if let Ok((_, blob)) = split_composite_key(&record) {
                            if blob.len() >= 12 {
                                found.push(blob.to_vec());
                            }
                        }
                        if found.len() >= 24 {
                            break;
                        }
                    }
                    if found.is_empty() {
                        continue;
                    }
                    found
                }
            };
            s.hits += 1;
            let inv2 = perm2.invert();
            let canon_to_orig: HashMap<u16, u16> = used2
                .iter()
                .enumerate()
                .map(|(dense, &orig)| (inv2.data[dense] as u16, orig))
                .collect();
            let mut partners = raw_partners;
            // Random subset keeps the probe cheap without biasing to the
            // value's storage order.
            while partners.len() > partners_per_hit {
                let i = rng.random_range(0..partners.len());
                partners.swap_remove(i);
            }
            for pb in partners {
                s.partners += 1;
                let b = CircuitSeq::from_blob(&pb);
                // Canonical-frame wires of b map back through the closure
                // frame; unseen wires get fresh indices past a_plus's range.
                let mut fresh_map: HashMap<u16, u16> = HashMap::new();
                let mut next_fresh = wires + 2;
                let mut map_wire = |w: u16| -> u16 {
                    if let Some(&orig) = canon_to_orig.get(&w) {
                        orig
                    } else {
                        *fresh_map.entry(w).or_insert_with(|| {
                            let v = next_fresh;
                            next_fresh += 1;
                            v
                        })
                    }
                };
                let b_mapped: Vec<[u16; 3]> = b
                    .gates
                    .iter()
                    .map(|&[t, c1, c2]| [map_wire(t), map_wire(c1), map_wire(c2)])
                    .collect();
                // rev(a) ++ b computes g; with [g] appended it is an identity.
                let mut m1_gates: Vec<[u16; 3]> = a.gates.iter().rev().copied().collect();
                m1_gates.extend(b_mapped);
                let m1 = simplify(CircuitSeq { gates: m1_gates });
                s.glued += 1;
                if canon_key(&m1) != Some(toffoli_key) {
                    // Degenerate g (e.g. its class collapses) or frame slip;
                    // count and move on -- production skips these.
                    s.m1_validation_failures += 1;
                    continue;
                }
                s.verified += 1;
                s.m1_validated += 1;
                s.m1_len[m1.gates.len().min(MAX_LEN - 1)] += 1;
                let mut id_gates = m1.gates.clone();
                id_gates.push(g);
                let identity = simplify(CircuitSeq { gates: id_gates });
                s.id_len[identity.gates.len().min(MAX_LEN - 1)] += 1;
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 500 == 0 {
            eprintln!(
                "[glue] {n}/{} sources, {:.0}s",
                sources.len(),
                start.elapsed().as_secs_f64()
            );
        }
        total.lock().unwrap().merge(s);
    });

    let s = total.into_inner().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "sources={}  gates-tried={}  lookups={}",
        s.sources, s.gates_tried, s.lookups
    );
    println!(
        "hits={} ({:.2}% of lookups)  partners-considered={}  glued={}",
        s.hits,
        s.hits as f64 * 100.0 / s.lookups.max(1) as f64,
        s.partners,
        s.glued
    );
    println!(
        "verified-m1={} ({:.2}% of glued)  validation-failures={}",
        s.verified,
        s.verified as f64 * 100.0 / s.glued.max(1) as f64,
        s.m1_validation_failures
    );
    println!("\nm1 candidate lengths (post-simplify, all verified vs canonical CCX key):");
    for i in 0..MAX_LEN {
        if s.m1_len[i] > 0 {
            println!("  gates={i:<3} {:>10}", s.m1_len[i]);
        }
    }
    println!("\nglued identity lengths (m1 ++ [g], post-simplify):");
    for i in 0..MAX_LEN {
        if s.id_len[i] > 0 {
            println!("  gates={i:<3} {:>10}", s.id_len[i]);
        }
    }
    println!(
        "\nelapsed={elapsed:.1}s  lookups/s={:.0}",
        s.lookups as f64 / elapsed.max(1e-9)
    );
}
