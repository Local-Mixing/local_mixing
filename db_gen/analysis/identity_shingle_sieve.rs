//! Greedy structural-novelty sieve over frozen-store identities.
//!
//! The curated store's per-key bloat is manufactured by the split enumeration:
//! every identity contributes one candidate per key even after orbit collapse,
//! so hot-key size equals the number of identities kept. This measures how many
//! identities survive a *shingle* filter: an identity is accepted only when
//! none of its cyclic L-gate arcs (both directions, wires relabelled by first
//! use) has been claimed by a previously accepted identity. Accepted sets
//! therefore share no contiguous L-gate fragment up to relabelling -- a direct,
//! tunable reading of "no two stored circuits are too similar in structure".
//!
//! Identities are first deduplicated by dihedral orbit (rotation + reversal +
//! relabelling), which alone removes the a.rev(b)/b.rev(a) spelling and
//! rotation redundancy the pair enumeration manufactures.
//!
//! Greedy acceptance is order-dependent; shards run in parallel, so counts
//! wobble a little between runs. This is a sizing instrument, not a builder.
//!
//! ```text
//! identity_shingle_sieve FROZEN_REGULAR_DIR [--shards N] [--lengths 4,5,6,8]
//! ```

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::cancel_adjacent_duplicates;
use local_mixing::db_mixing::frozen::scan_shard;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

const LOCK_SHARDS: usize = 1024;
const MAX_LEN: usize = 64;

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
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

/// Gate triples narrowed to u8, verbatim: g57 controls are
/// polarity-asymmetric ([a,x,y] fires on NOT x AND y), no reordering.
fn norm(gates: &[[u16; 3]]) -> Vec<[u8; 3]> {
    gates
        .iter()
        .map(|&[t, a, b]| [t as u8, a as u8, b as u8])
        .collect()
}

/// First-use relabelling; control positions preserved.
fn relabel(word: &[[u8; 3]]) -> Vec<u8> {
    let mut map = [255u8; 256];
    let mut next = 0u8;
    let mut out = Vec::with_capacity(word.len() * 3);
    for gate in word {
        for &wire in gate {
            if map[wire as usize] == 255 {
                map[wire as usize] = next;
                next = next.wrapping_add(1);
            }
            out.push(map[wire as usize]);
        }
    }
    out
}

/// Dihedral-orbit fingerprint: min over both directions and all rotations of
/// the relabelled serialization.
fn orbit_hash(word: &[[u8; 3]]) -> u128 {
    let n = word.len();
    let mut best: Option<Vec<u8>> = None;
    let mut reversed: Vec<[u8; 3]> = word.to_vec();
    reversed.reverse();
    for seq in [word, reversed.as_slice()] {
        let mut rotated = Vec::with_capacity(n);
        for start in 0..n {
            rotated.clear();
            rotated.extend_from_slice(&seq[start..]);
            rotated.extend_from_slice(&seq[..start]);
            let form = relabel(&rotated);
            if best.as_ref().is_none_or(|b| form < *b) {
                best = Some(form);
            }
        }
    }
    xxh3_128(best.as_deref().unwrap_or_default())
}

/// All cyclic arcs of `len` gates, both directions, relabelled and hashed.
fn arc_hashes(word: &[[u8; 3]], len: usize, out: &mut Vec<u64>) {
    out.clear();
    let n = word.len();
    if len > n {
        return;
    }
    let mut reversed: Vec<[u8; 3]> = word.to_vec();
    reversed.reverse();
    let mut arc = Vec::with_capacity(len);
    for seq in [word, reversed.as_slice()] {
        for start in 0..n {
            arc.clear();
            for offset in 0..len {
                arc.push(seq[(start + offset) % n]);
            }
            out.push(xxh3_64(&relabel(&arc)));
        }
    }
}

struct ShardedSet<T> {
    shards: Vec<Mutex<FxHashSet<T>>>,
}

impl<T: std::hash::Hash + Eq + Copy> ShardedSet<T> {
    fn new() -> Self {
        Self {
            shards: (0..LOCK_SHARDS)
                .map(|_| Mutex::new(FxHashSet::default()))
                .collect(),
        }
    }
    fn shard_of(&self, hash: u64) -> &Mutex<FxHashSet<T>> {
        &self.shards[(hash as usize) & (LOCK_SHARDS - 1)]
    }
    fn len(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.lock().unwrap().len() as u64)
            .sum()
    }
}

/// One sieve per shingle length: accepted iff no arc already claimed.
struct Sieve {
    len: usize,
    arcs: ShardedSet<u64>,
    accepted: AtomicU64,
    accepted_len: Vec<AtomicU64>,
}

impl Sieve {
    fn new(len: usize) -> Self {
        Self {
            len,
            arcs: ShardedSet::new(),
            accepted: AtomicU64::new(0),
            accepted_len: (0..MAX_LEN).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn offer(&self, word: &[[u8; 3]], scratch: &mut Vec<u64>) {
        arc_hashes(word, self.len, scratch);
        if scratch.is_empty() {
            // Shorter than the shingle: structurally novel by definition.
            self.accepted.fetch_add(1, Ordering::Relaxed);
            self.accepted_len[word.len().min(MAX_LEN - 1)].fetch_add(1, Ordering::Relaxed);
            return;
        }
        for &h in scratch.iter() {
            if self.arcs.shard_of(h).lock().unwrap().contains(&h) {
                return;
            }
        }
        // Claim. A racing acceptance of an overlapping identity can slip in
        // between check and claim; for a sizing instrument that is noise.
        for &h in scratch.iter() {
            self.arcs.shard_of(h).lock().unwrap().insert(h);
        }
        self.accepted.fetch_add(1, Ordering::Relaxed);
        self.accepted_len[word.len().min(MAX_LEN - 1)].fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!(
            "usage: identity_shingle_sieve FROZEN_REGULAR_DIR [--shards N] [--lengths 4,5,6,8]"
        );
        std::process::exit(2);
    };
    let shards: usize = flag_val(&args, "--shards")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
        .clamp(1, 256);
    let lengths: Vec<usize> = flag_val(&args, "--lengths")
        .unwrap_or_else(|| "4,5,6,8".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let sieves: Vec<Sieve> = lengths.iter().map(|&l| Sieve::new(l)).collect();
    let orbits: ShardedSet<u128> = ShardedSet::new();
    let spellings = AtomicU64::new(0);
    let identities = AtomicU64::new(0);
    let orbit_unique = AtomicU64::new(0);
    let orbit_len: Vec<AtomicU64> = (0..MAX_LEN).map(|_| AtomicU64::new(0)).collect();
    let done = AtomicU64::new(0);
    let start = Instant::now();

    (0..shards).into_par_iter().for_each(|shard| {
        let mut scratch: Vec<u64> = Vec::new();
        scan_shard(dir, shard, &mut |value| {
            let members = class_members(value);
            if members.len() < 2 {
                return;
            }
            let circuits: Vec<CircuitSeq> =
                members.iter().map(|b| CircuitSeq::from_blob(b)).collect();
            for left in 0..circuits.len() {
                for right in 0..circuits.len() {
                    if left == right {
                        continue;
                    }
                    // One spelling per ordered pair: a ++ reverse(b). The
                    // reverse_a ++ b spelling is the same dihedral orbit
                    // (reversal), and (b, a) covers the complementary word.
                    let a = &circuits[left];
                    let b = &circuits[right];
                    let mut gates = a.gates.clone();
                    gates.extend(b.gates.iter().rev().copied());
                    spellings.fetch_add(1, Ordering::Relaxed);
                    let identity = simplify(CircuitSeq { gates });
                    if identity.gates.len() < 3 {
                        continue;
                    }
                    identities.fetch_add(1, Ordering::Relaxed);
                    let word = norm(&identity.gates);
                    let oh = orbit_hash(&word);
                    if !self_insert(&orbits, oh) {
                        continue;
                    }
                    orbit_unique.fetch_add(1, Ordering::Relaxed);
                    orbit_len[word.len().min(MAX_LEN - 1)].fetch_add(1, Ordering::Relaxed);
                    for sieve in &sieves {
                        sieve.offer(&word, &mut scratch);
                    }
                }
            }
        });
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16 == 0 || n as usize == shards {
            eprintln!(
                "[sieve] {n}/{shards} shards, {:.0}s, orbit-unique={}",
                start.elapsed().as_secs_f64(),
                orbit_unique.load(Ordering::Relaxed)
            );
        }
    });

    let elapsed = start.elapsed().as_secs_f64();
    println!("\nshards={shards} (of 256)  elapsed={elapsed:.1}s");
    println!("spellings={}", spellings.load(Ordering::Relaxed));
    println!(
        "identities(>=3 gates)={}",
        identities.load(Ordering::Relaxed)
    );
    println!(
        "orbit-unique={} ({:.2}% of identities)",
        orbit_unique.load(Ordering::Relaxed),
        orbit_unique.load(Ordering::Relaxed) as f64 * 100.0
            / identities.load(Ordering::Relaxed).max(1) as f64
    );
    println!("\norbit-unique identity lengths:");
    for len in 0..MAX_LEN {
        let count = orbit_len[len].load(Ordering::Relaxed);
        if count > 0 {
            println!("  n={len:<3} {count:>12}");
        }
    }
    for sieve in &sieves {
        let accepted = sieve.accepted.load(Ordering::Relaxed);
        println!(
            "\n=== shingle L={} ===\naccepted={} ({:.3}% of orbit-unique)  claimed-arcs={}",
            sieve.len,
            accepted,
            accepted as f64 * 100.0 / orbit_unique.load(Ordering::Relaxed).max(1) as f64,
            sieve.arcs.len()
        );
        for len in 0..MAX_LEN {
            let count = sieve.accepted_len[len].load(Ordering::Relaxed);
            if count > 0 {
                println!("  n={len:<3} {count:>12}");
            }
        }
    }
    if shards < 256 {
        let scale = 256.0 / shards as f64;
        println!("\n=== full-store projection (x{scale:.0}) ===");
        println!(
            "orbit-unique {:>16.0}",
            orbit_unique.load(Ordering::Relaxed) as f64 * scale
        );
        for sieve in &sieves {
            println!(
                "L={} accepted {:>16.0}",
                sieve.len,
                sieve.accepted.load(Ordering::Relaxed) as f64 * scale
            );
        }
    }
}

fn self_insert(set: &ShardedSet<u128>, hash: u128) -> bool {
    set.shards[(hash as u64 as usize) & (LOCK_SHARDS - 1)]
        .lock()
        .unwrap()
        .insert(hash)
}
