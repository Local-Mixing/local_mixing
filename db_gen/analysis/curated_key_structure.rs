//! Structural-redundancy census over one curated key's candidate list.
//!
//! The question this answers: of a hot key's hundreds of millions of stored
//! candidates, how many are *structurally distinct*, under equivalences of
//! increasing coarseness? Each count is the survivor population of a concrete
//! candidate filter:
//!
//! 1. `exact`      -- the store's own dedup (baseline).
//! 2. `orbit`      -- candidates that glue to the same generating identity, up
//!                    to rotation, reversal, and wire relabelling. Every
//!                    candidate c under key K closes to an identity c ++
//!                    reverse(m) where m is K's minimal stored circuit; two
//!                    candidates whose closures are dihedral-equivalent are one
//!                    identity split at two places -- rotation siblings, the
//!                    redundancy the split enumeration manufactures wholesale.
//! 3. `2gram`      -- multiset of adjacent normalized gate pairs. Collapses
//!                    circuits with identical local texture.
//! 4. `multiset`   -- bag of normalized gates, order ignored entirely. The
//!                    floor: how many genuinely different gate inventories
//!                    exist.
//!
//! The relabelling inside `orbit` is first-use with original-index tie-breaks,
//! not a full canonical form under wire automorphisms, so orbit counts are
//! upper bounds (a few true orbits may count twice). Good enough to size the
//! collapse; the exact/orbit ratio is the headline.
//!
//! ```text
//! curated_key_structure COMPOSITE_ROCKS KEY_HEX [KEY_HEX..]
//!     [--sample N] [--sample-out FILE] [--max M] [--find-sizes N,N,..]
//! ```
//!
//! `--sample` reservoir-samples N candidates per gate count into FILE as
//! `gates<TAB>hex` lines for offline pairwise-similarity analysis.
//! `--find-sizes` ignores KEY_HEX and instead scans the whole store printing,
//! for each requested candidate-count target, the first key within 20% of it.

use local_mixing::db_generation::curated_full::{FUNCTION_KEY_BYTES, split_composite_key};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::collections::HashSet;
use std::error::Error;
use std::io::Write as _;
use std::sync::mpsc::sync_channel;
use std::time::Instant;
use xxhash_rust::xxh3::xxh3_128;

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

/// Gate triples of a blob, verbatim: g57 controls are polarity-asymmetric
/// ([a,x,y] fires on NOT x AND y), so no reordering is valid.
fn norm_gates(blob: &[u8]) -> Vec<[u8; 3]> {
    blob.chunks_exact(3).map(|g| [g[0], g[1], g[2]]).collect()
}

/// First-use relabelling of a gate word; control positions preserved.
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

/// Dihedral-orbit fingerprint of a gate word: min over both directions and all
/// rotations of the relabelled serialization.
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

fn multiset_hash(gates: &[[u8; 3]]) -> u128 {
    let mut sorted: Vec<[u8; 3]> = gates.to_vec();
    sorted.sort_unstable();
    xxh3_128(&sorted.concat())
}

fn twogram_hash(gates: &[[u8; 3]]) -> u128 {
    let mut grams: Vec<u64> = gates
        .windows(2)
        .map(|pair| xxhash_rust::xxh3::xxh3_64(&pair.concat()))
        .collect();
    grams.sort_unstable();
    let bytes: Vec<u8> = grams.iter().flat_map(|g| g.to_le_bytes()).collect();
    xxh3_128(&bytes)
}

struct Reservoir {
    per_len: Vec<Vec<Vec<u8>>>,
    seen: Vec<u64>,
    cap: usize,
}

impl Reservoir {
    fn new(cap: usize) -> Self {
        Self {
            per_len: (0..=MAX_GATES).map(|_| Vec::new()).collect(),
            seen: vec![0; MAX_GATES + 1],
            cap,
        }
    }
    fn offer(&mut self, blob: &[u8], rng: &mut StdRng) {
        let gates = (blob.len() / 3).min(MAX_GATES);
        self.seen[gates] += 1;
        if self.per_len[gates].len() < self.cap {
            self.per_len[gates].push(blob.to_vec());
        } else {
            let j = rng.random_range(0..self.seen[gates]);
            if (j as usize) < self.cap {
                self.per_len[gates][j as usize] = blob.to_vec();
            }
        }
    }
}

fn open_store(path: &str) -> AnyResult<DB> {
    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    Ok(DB::open_for_read_only(&options, path, false)?)
}

/// Stream one key's candidate blobs in chunks through `handle`.
fn scan_key(
    db: &DB,
    key: &[u8; FUNCTION_KEY_BYTES],
    max: u64,
    mut handle: impl FnMut(Vec<Vec<u8>>),
) -> AnyResult<u64> {
    let mut total = 0u64;
    let mut chunk: Vec<Vec<u8>> = Vec::with_capacity(CHUNK);
    for item in db.iterator(IteratorMode::From(key, Direction::Forward)) {
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
        if max > 0 && total >= max {
            break;
        }
    }
    if !chunk.is_empty() {
        handle(chunk);
    }
    Ok(total)
}

fn find_sizes(db: &DB, targets: &[u64]) -> AnyResult<()> {
    let mut current: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut count = 0u64;
    let mut found: Vec<Option<([u8; FUNCTION_KEY_BYTES], u64)>> = vec![None; targets.len()];
    let check = |key: Option<[u8; FUNCTION_KEY_BYTES]>,
                 count: u64,
                 found: &mut Vec<Option<([u8; FUNCTION_KEY_BYTES], u64)>>| {
        if let Some(k) = key {
            for (i, &t) in targets.iter().enumerate() {
                let close = count as f64 >= t as f64 * 0.8 && count as f64 <= t as f64 * 1.2;
                if close && found[i].is_none() {
                    found[i] = Some((k, count));
                }
            }
        }
    };
    for item in db.iterator(IteratorMode::Start) {
        let (record, _) = item?;
        let Ok((key, _)) = split_composite_key(&record) else {
            continue;
        };
        if current != Some(key) {
            check(current, count, &mut found);
            current = Some(key);
            count = 0;
        }
        count += 1;
    }
    check(current, count, &mut found);
    for (i, entry) in found.iter().enumerate() {
        match entry {
            Some((key, n)) => {
                let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
                println!("target={} key={hex} candidates={n}", targets[i]);
            }
            None => println!("target={} (no key within 20%)", targets[i]),
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!(
            "usage: curated_key_structure COMPOSITE_ROCKS KEY_HEX.. [--sample N] [--sample-out F] [--max M] [--find-sizes N,N,..]"
        );
        std::process::exit(2);
    };
    let db = open_store(path)?;

    if let Some(list) = flag_val(&args, "--find-sizes") {
        let targets: Vec<u64> = list.split(',').filter_map(|s| s.parse().ok()).collect();
        return find_sizes(&db, &targets);
    }

    let sample: usize = flag_val(&args, "--sample")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let sample_out = flag_val(&args, "--sample-out");
    let max: u64 = flag_val(&args, "--max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let keys: Vec<[u8; FUNCTION_KEY_BYTES]> = args[2..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .map(|a| parse_key(a))
        .collect::<AnyResult<_>>()?;
    if keys.is_empty() {
        return Err("no keys given".into());
    }

    for key in &keys {
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        eprintln!("[structure] key={hex}: pass 1 (minimal circuit)");
        let start = Instant::now();

        // Pass 1: the key's minimal stored circuit, for identity closure.
        let mut minimal: Option<Vec<u8>> = None;
        scan_key(&db, key, max, |chunk| {
            for blob in &chunk {
                if minimal
                    .as_ref()
                    .is_none_or(|m| (blob.len(), blob.as_slice()) < (m.len(), m.as_slice()))
                {
                    minimal = Some(blob.clone());
                }
            }
        })?;
        let minimal = minimal.ok_or("key has no candidates")?;
        let closure_tail: Vec<[u8; 3]> = {
            let mut m = norm_gates(&minimal);
            m.reverse();
            m
        };
        eprintln!(
            "[structure] minimal={} gates ({}), pass 2 (fingerprints)",
            minimal.len() / 3,
            minimal
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );

        // Pass 2: fingerprint in parallel per chunk, insert sequentially --
        // 1.5B set inserts through shared mutexes would serialize on locks,
        // while a single owning thread absorbs them at memory speed.
        let mut orbit: Vec<HashSet<u128>> = (0..=MAX_GATES).map(|_| HashSet::new()).collect();
        let mut twogram: Vec<HashSet<u128>> = (0..=MAX_GATES).map(|_| HashSet::new()).collect();
        let mut multiset: Vec<HashSet<u128>> = (0..=MAX_GATES).map(|_| HashSet::new()).collect();
        let mut exact: Vec<u64> = vec![0; MAX_GATES + 1];
        let mut reservoir = Reservoir::new(sample);
        let mut rng = StdRng::seed_from_u64(0x5eed_51e7e);

        let (tx, rx) = sync_channel::<Vec<Vec<u8>>>(4);
        let db_ref = &db;
        let scanned = std::thread::scope(|scope| -> AnyResult<u64> {
            let reader = scope.spawn(move || {
                scan_key(db_ref, key, max, |chunk| {
                    let _ = tx.send(chunk);
                })
            });
            let mut seen = 0u64;
            for chunk in rx.iter() {
                seen += chunk.len() as u64;
                let prints: Vec<(usize, u128, u128, u128)> = chunk
                    .par_iter()
                    .map(|blob| {
                        let gates = norm_gates(blob);
                        let g = gates.len().min(MAX_GATES);
                        let ms = multiset_hash(&gates);
                        let tg = twogram_hash(&gates);
                        let mut word = gates;
                        word.extend_from_slice(&closure_tail);
                        (g, ms, tg, orbit_hash(&word))
                    })
                    .collect();
                for (g, ms, tg, ob) in prints {
                    exact[g] += 1;
                    multiset[g].insert(ms);
                    twogram[g].insert(tg);
                    orbit[g].insert(ob);
                }
                if sample > 0 {
                    for blob in &chunk {
                        reservoir.offer(blob, &mut rng);
                    }
                }
                if seen % (CHUNK as u64 * 64) == 0 {
                    eprintln!(
                        "[structure] scanned {seen} ({:.0}/s)",
                        seen as f64 / start.elapsed().as_secs_f64()
                    );
                }
            }
            // rx.iter() ends when the reader drops its moved sender.
            reader.join().map_err(|_| "reader thread panicked")?
        })?;

        println!("\n=== key={hex} ===");
        println!("candidates={scanned}  minimal-gates={}", minimal.len() / 3);
        println!(
            "{:>6} {:>14} {:>14} {:>14} {:>14}",
            "gates", "exact", "orbit", "2gram", "multiset"
        );
        let mut totals = (0u64, 0u64, 0u64, 0u64);
        for g in 0..=MAX_GATES {
            let e = exact[g];
            if e == 0 {
                continue;
            }
            let (o, t2, ms) = (
                orbit[g].len() as u64,
                twogram[g].len() as u64,
                multiset[g].len() as u64,
            );
            println!("{g:>6} {e:>14} {o:>14} {t2:>14} {ms:>14}");
            totals.0 += e;
            totals.1 += o;
            totals.2 += t2;
            totals.3 += ms;
        }
        println!(
            "{:>6} {:>14} {:>14} {:>14} {:>14}",
            "total", totals.0, totals.1, totals.2, totals.3
        );
        println!(
            "collapse: orbit {:.1}x  2gram {:.1}x  multiset {:.1}x  ({:.0}s)",
            totals.0 as f64 / totals.1.max(1) as f64,
            totals.0 as f64 / totals.2.max(1) as f64,
            totals.0 as f64 / totals.3.max(1) as f64,
            start.elapsed().as_secs_f64()
        );

        if sample > 0 {
            let file = sample_out
                .clone()
                .unwrap_or_else(|| format!("sample_{hex}.txt"));
            let mut out = std::io::BufWriter::new(std::fs::File::create(&file)?);
            for (gates, blobs) in reservoir.per_len.iter().enumerate() {
                for blob in blobs {
                    let blob_hex: String = blob.iter().map(|b| format!("{b:02x}")).collect();
                    writeln!(out, "{gates}\t{blob_hex}")?;
                }
            }
            eprintln!("[structure] sample written to {file}");
        }
    }
    Ok(())
}
