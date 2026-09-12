//! Per-key diversity-filter prototype: orbit collapse + shingle sieve.
//!
//! Streams one key's candidates in deterministic store order and applies, in
//! sequence:
//!
//! 1. ORBIT COLLAPSE -- drop any candidate whose identity closure (candidate
//!    ++ reverse(minimal circuit)) has the same dihedral-orbit fingerprint as
//!    an earlier candidate. Rotation/reflection siblings of one generating
//!    identity collapse to the first-seen representative.
//! 2. SHINGLE SIEVE (per L) -- drop any orbit representative that shares a
//!    contiguous L-gate subword (either direction, wires relabelled by first
//!    use) with an earlier survivor. What remains is a set in which no two
//!    circuits share an L-gate stretch of structure.
//!
//! Reports survivor counts per gate count and reservoir-samples three
//! populations (raw / orbit / sieve at each L) for offline similarity
//! comparison with db_gen/analysis/sample_similarity.py.
//!
//! ```text
//! curated_key_filter COMPOSITE_ROCKS KEY_HEX [--lengths 4,5,6,8]
//!     [--sample N] [--sample-prefix P]
//! ```

use local_mixing::db_generation::curated_full::{FUNCTION_KEY_BYTES, split_composite_key};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use rustc_hash::FxHashSet;
use std::error::Error;
use std::io::Write as _;
use std::sync::mpsc::sync_channel;
use std::time::Instant;
use xxhash_rust::xxh3::{xxh3_64, xxh3_128};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const MAX_GATES: usize = 64;
const CHUNK: usize = 262_144;

fn parse_key(text: &str) -> AnyResult<[u8; FUNCTION_KEY_BYTES]> {
    if text.len() != FUNCTION_KEY_BYTES * 2 {
        return Err(format!("key must be {} hex chars", FUNCTION_KEY_BYTES * 2).into());
    }
    let mut key = [0u8; FUNCTION_KEY_BYTES];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)?;
    }
    Ok(key)
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Gate triples verbatim: g57 controls are polarity-asymmetric, no reordering.
fn norm_gates(blob: &[u8]) -> Vec<[u8; 3]> {
    blob.chunks_exact(3).map(|g| [g[0], g[1], g[2]]).collect()
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

/// Linear subword hashes of `len` gates, both directions, relabelled.
fn subword_hashes(word: &[[u8; 3]], len: usize, out: &mut Vec<u64>) {
    out.clear();
    let n = word.len();
    if len > n {
        return;
    }
    let mut reversed: Vec<[u8; 3]> = word.to_vec();
    reversed.reverse();
    for seq in [word, reversed.as_slice()] {
        for start in 0..=(n - len) {
            out.push(xxh3_64(&relabel(&seq[start..start + len])));
        }
    }
}

struct Reservoir {
    items: Vec<(usize, Vec<u8>)>,
    seen: u64,
    cap: usize,
}

impl Reservoir {
    fn new(cap: usize) -> Self {
        Self {
            items: Vec::new(),
            seen: 0,
            cap,
        }
    }
    fn offer(&mut self, gates: usize, blob: &[u8], rng: &mut StdRng) {
        self.seen += 1;
        if self.items.len() < self.cap {
            self.items.push((gates, blob.to_vec()));
        } else {
            let j = rng.random_range(0..self.seen);
            if (j as usize) < self.cap {
                self.items[j as usize] = (gates, blob.to_vec());
            }
        }
    }
    fn write(&self, path: &str) -> AnyResult<()> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        for (gates, blob) in &self.items {
            let hex: String = blob.iter().map(|b| format!("{b:02x}")).collect();
            writeln!(out, "{gates}\t{hex}")?;
        }
        Ok(())
    }
}

struct Sieve {
    len: usize,
    claimed: FxHashSet<u64>,
    survivors: u64,
    by_len: Vec<u64>,
    sample: Reservoir,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(key_hex)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: curated_key_filter COMPOSITE_ROCKS KEY_HEX [--lengths 4,5,6,8] [--sample N] [--sample-prefix P]"
        );
        std::process::exit(2);
    };
    let key = parse_key(key_hex)?;
    let lengths: Vec<usize> = flag_val(&args, "--lengths")
        .unwrap_or_else(|| "4,5,6,8".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let sample: usize = flag_val(&args, "--sample")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let prefix = flag_val(&args, "--sample-prefix").unwrap_or_else(|| format!("filter_{key_hex}"));

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    let db = DB::open_for_read_only(&options, path.as_str(), false)?;

    let scan = |mut handle: Box<dyn FnMut(Vec<Vec<u8>>) + '_>| -> AnyResult<u64> {
        let mut total = 0u64;
        let mut chunk: Vec<Vec<u8>> = Vec::with_capacity(CHUNK);
        for item in db.iterator(IteratorMode::From(&key, Direction::Forward)) {
            let (record, _) = item?;
            if record.len() < FUNCTION_KEY_BYTES || record[..FUNCTION_KEY_BYTES] != key[..] {
                break;
            }
            let (_, blob) = split_composite_key(&record)?;
            chunk.push(blob.to_vec());
            total += 1;
            if chunk.len() == CHUNK {
                handle(std::mem::take(&mut chunk));
                chunk.reserve(CHUNK);
            }
        }
        if !chunk.is_empty() {
            handle(chunk);
        }
        Ok(total)
    };

    // Pass 1: minimal circuit.
    let start = Instant::now();
    let mut minimal: Option<Vec<u8>> = None;
    scan(Box::new(|chunk| {
        for blob in &chunk {
            if minimal
                .as_ref()
                .is_none_or(|m| (blob.len(), blob.as_slice()) < (m.len(), m.as_slice()))
            {
                minimal = Some(blob.clone());
            }
        }
    }))?;
    let minimal = minimal.ok_or("key has no candidates")?;
    let closure_tail: Vec<[u8; 3]> = {
        let mut m = norm_gates(&minimal);
        m.reverse();
        m
    };
    eprintln!("[filter] minimal={} gates; pass 2", minimal.len() / 3);

    // Pass 2: deterministic sequential acceptance, parallel hashing.
    let mut rng = StdRng::seed_from_u64(0xd1_5eed);
    let mut raw_sample = Reservoir::new(sample);
    let mut orbit_sample = Reservoir::new(sample);
    let mut orbit_seen: FxHashSet<u128> = FxHashSet::default();
    let mut orbit_by_len = vec![0u64; MAX_GATES + 1];
    let mut exact_by_len = vec![0u64; MAX_GATES + 1];
    let mut sieves: Vec<Sieve> = lengths
        .iter()
        .map(|&l| Sieve {
            len: l,
            claimed: FxHashSet::default(),
            survivors: 0,
            by_len: vec![0; MAX_GATES + 1],
            sample: Reservoir::new(sample),
        })
        .collect();

    let (tx, rx) = sync_channel::<Vec<Vec<u8>>>(4);
    let db_ref = &scan;
    let total = std::thread::scope(|scope| -> AnyResult<u64> {
        let reader = scope.spawn(move || {
            db_ref(Box::new(|chunk| {
                let _ = tx.send(chunk);
            }))
        });
        let n_lengths = lengths.clone();
        let mut processed = 0u64;
        for chunk in rx.iter() {
            processed += chunk.len() as u64;
            // Parallel: orbit hash + per-L subword hashes per candidate.
            let hashed: Vec<(Vec<u8>, u128, Vec<Vec<u64>>)> = chunk
                .into_par_iter()
                .map(|blob| {
                    let gates = norm_gates(&blob);
                    let mut word = gates.clone();
                    word.extend_from_slice(&closure_tail);
                    let oh = orbit_hash(&word);
                    let mut subs = Vec::with_capacity(n_lengths.len());
                    for &l in &n_lengths {
                        let mut v = Vec::new();
                        subword_hashes(&gates, l, &mut v);
                        subs.push(v);
                    }
                    (blob, oh, subs)
                })
                .collect();
            // Sequential: acceptance in store order.
            for (blob, oh, subs) in hashed {
                let g = (blob.len() / 3).min(MAX_GATES);
                exact_by_len[g] += 1;
                if sample > 0 {
                    raw_sample.offer(g, &blob, &mut rng);
                }
                if !orbit_seen.insert(oh) {
                    continue;
                }
                orbit_by_len[g] += 1;
                if sample > 0 {
                    orbit_sample.offer(g, &blob, &mut rng);
                }
                for (sieve, hashes) in sieves.iter_mut().zip(&subs) {
                    if hashes.is_empty() {
                        // Shorter than the shingle: novel by definition.
                    } else if hashes.iter().any(|h| sieve.claimed.contains(h)) {
                        continue;
                    }
                    for &h in hashes {
                        sieve.claimed.insert(h);
                    }
                    sieve.survivors += 1;
                    sieve.by_len[g] += 1;
                    if sample > 0 {
                        sieve.sample.offer(g, &blob, &mut rng);
                    }
                }
            }
            if processed % (CHUNK as u64 * 64) == 0 {
                eprintln!(
                    "[filter] {processed} scanned, orbit={} ({:.0}/s)",
                    orbit_seen.len(),
                    processed as f64 / start.elapsed().as_secs_f64()
                );
            }
        }
        reader.join().map_err(|_| "reader thread panicked")?
    })?;

    println!("=== key={key_hex} ===");
    println!("candidates={total}  minimal-gates={}", minimal.len() / 3);
    print!("{:>6} {:>14} {:>14}", "gates", "exact", "orbit");
    for sieve in &sieves {
        print!("{:>13}L{}", "sieve-", sieve.len);
    }
    println!();
    for g in 0..=MAX_GATES {
        if exact_by_len[g] == 0 {
            continue;
        }
        print!("{g:>6} {:>14} {:>14}", exact_by_len[g], orbit_by_len[g]);
        for sieve in &sieves {
            print!("{:>15}", sieve.by_len[g]);
        }
        println!();
    }
    let exact_total: u64 = exact_by_len.iter().sum();
    let orbit_total: u64 = orbit_by_len.iter().sum();
    print!("{:>6} {exact_total:>14} {orbit_total:>14}", "total");
    for sieve in &sieves {
        print!("{:>15}", sieve.survivors);
    }
    println!();
    println!(
        "orbit collapse {:.1}x; elapsed {:.0}s",
        exact_total as f64 / orbit_total.max(1) as f64,
        start.elapsed().as_secs_f64()
    );

    if sample > 0 {
        raw_sample.write(&format!("{prefix}_raw.txt"))?;
        orbit_sample.write(&format!("{prefix}_orbit.txt"))?;
        for sieve in &sieves {
            sieve
                .sample
                .write(&format!("{prefix}_L{}.txt", sieve.len))?;
        }
        eprintln!("[filter] samples written with prefix {prefix}");
    }
    Ok(())
}
