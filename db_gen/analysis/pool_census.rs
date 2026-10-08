//! Store-wide pool census over a curated-full composite store.
//!
//! For every function key ("pool"), aggregate its candidate circuits by gate
//! count, and report the distributions that decide the v3 true-friend design:
//!
//!   1. Pool sizes: how many keys hold how many candidates.
//!   2. Length make-up: for pools of each size class, which circuit sizes the
//!      candidates have.
//!   3. Same-length cohorts: for each gate count L, how many candidates share
//!      a (key, L) cell -- the raw material for EQUAL-LENGTH true friendships,
//!      an upper bound on what the pairwise partition can certify.
//!
//! Read-only; never writes to the store. Writes CSVs into `--out`.
//!
//! ```text
//! pool_census COMPOSITE_ROCKS [--out DIR] [--top N]
//! ```

use local_mixing::db_generation::curated_full::{
    COMPOSITE_COMPLETE_MARKER, COMPOSITE_FORMAT_MARKER, FUNCTION_KEY_BYTES, split_composite_key,
};
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::error::Error;
use std::io::Write as _;
use std::path::{Path, PathBuf};

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Gate-count bins; the last bin absorbs anything longer.
const NG: usize = 64;
/// Size-class bins: exact 1..=8, then doubling ranges 9-16, 17-32, ...
const NB: usize = 40;

fn size_bucket(n: u64) -> usize {
    debug_assert!(n >= 1);
    if n <= 8 {
        (n - 1) as usize
    } else {
        let p = (64 - (n - 1).leading_zeros()) as usize;
        (8 + p - 4).min(NB - 1)
    }
}

fn bucket_label(i: usize) -> String {
    if i < 8 {
        format!("{}", i + 1)
    } else {
        let p = i - 8 + 4;
        format!("{}-{}", (1u64 << (p - 1)) + 1, 1u64 << p)
    }
}

struct TopKey {
    total: u64,
    key: [u8; FUNCTION_KEY_BYTES],
    min_gates: usize,
    lens: Vec<(usize, u64)>,
}

struct Stats {
    keys: u64,
    candidates: u64,
    meta: u64,
    malformed: u64,
    keys_by_poolsize: [u64; NB],
    cands_by_poolsize: [u64; NB],
    keys_by_mingates: [u64; NG],
    cands_by_len: [u64; NG],
    keys_with_len: [u64; NG],
    keys_with_len_ge2: [u64; NG],
    cands_in_ge2: [u64; NG],
    max_cohort: [u64; NG],
    cohort_hist: Vec<[u64; NB]>,
    poolsize_x_len: Vec<[u64; NG]>,
    top: Vec<TopKey>,
}

impl Stats {
    fn new() -> Self {
        Self {
            keys: 0,
            candidates: 0,
            meta: 0,
            malformed: 0,
            keys_by_poolsize: [0; NB],
            cands_by_poolsize: [0; NB],
            keys_by_mingates: [0; NG],
            cands_by_len: [0; NG],
            keys_with_len: [0; NG],
            keys_with_len_ge2: [0; NG],
            cands_in_ge2: [0; NG],
            max_cohort: [0; NG],
            cohort_hist: vec![[0; NB]; NG],
            poolsize_x_len: vec![[0; NG]; NB],
            top: Vec::new(),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.keys += other.keys;
        self.candidates += other.candidates;
        self.meta += other.meta;
        self.malformed += other.malformed;
        for i in 0..NB {
            self.keys_by_poolsize[i] += other.keys_by_poolsize[i];
            self.cands_by_poolsize[i] += other.cands_by_poolsize[i];
            for l in 0..NG {
                self.poolsize_x_len[i][l] += other.poolsize_x_len[i][l];
            }
        }
        for l in 0..NG {
            self.keys_by_mingates[l] += other.keys_by_mingates[l];
            self.cands_by_len[l] += other.cands_by_len[l];
            self.keys_with_len[l] += other.keys_with_len[l];
            self.keys_with_len_ge2[l] += other.keys_with_len_ge2[l];
            self.cands_in_ge2[l] += other.cands_in_ge2[l];
            self.max_cohort[l] = self.max_cohort[l].max(other.max_cohort[l]);
            for b in 0..NB {
                self.cohort_hist[l][b] += other.cohort_hist[l][b];
            }
        }
        self.top.extend(other.top);
        self
    }

    fn finish_pool(&mut self, key: [u8; FUNCTION_KEY_BYTES], counts: &[u64; NG], top_n: usize) {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return;
        }
        let min_gates = counts.iter().position(|&c| c > 0).unwrap_or(NG - 1);
        self.keys += 1;
        self.candidates += total;
        self.keys_by_mingates[min_gates] += 1;
        let pb = size_bucket(total);
        self.keys_by_poolsize[pb] += 1;
        self.cands_by_poolsize[pb] += total;
        for (l, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            self.cands_by_len[l] += c;
            self.keys_with_len[l] += 1;
            if c >= 2 {
                self.keys_with_len_ge2[l] += 1;
                self.cands_in_ge2[l] += c;
            }
            self.max_cohort[l] = self.max_cohort[l].max(c);
            self.cohort_hist[l][size_bucket(c)] += 1;
            self.poolsize_x_len[pb][l] += c;
        }
        // Keep a generous per-worker surplus; the final merge re-truncates.
        self.top.push(TopKey {
            total,
            key,
            min_gates,
            lens: counts
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(l, &c)| (l, c))
                .collect(),
        });
        if self.top.len() > top_n * 4 {
            self.top.sort_unstable_by(|a, b| b.total.cmp(&a.total));
            self.top.truncate(top_n);
        }
    }
}

fn hex(key: &[u8; FUNCTION_KEY_BYTES]) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}

fn scan_prefix(db: &DB, prefix: u8, top_n: usize) -> Stats {
    let mut stats = Stats::new();
    let start = [prefix];
    let mut current: Option<[u8; FUNCTION_KEY_BYTES]> = None;
    let mut counts = [0u64; NG];
    for item in db.iterator(IteratorMode::From(&start, Direction::Forward)) {
        let Ok((record, _)) = item else { break };
        if record.is_empty() || record[0] != prefix {
            break;
        }
        if record.as_ref() == COMPOSITE_FORMAT_MARKER
            || record.as_ref() == COMPOSITE_COMPLETE_MARKER
        {
            stats.meta += 1;
            continue;
        }
        let Ok((key, blob)) = split_composite_key(&record) else {
            stats.malformed += 1;
            continue;
        };
        if blob.is_empty() || blob.len() % 3 != 0 {
            stats.malformed += 1;
            continue;
        }
        if current != Some(key) {
            if let Some(done) = current.take() {
                stats.finish_pool(done, &counts, top_n);
            }
            current = Some(key);
            counts = [0; NG];
        }
        counts[(blob.len() / 3).min(NG - 1)] += 1;
    }
    if let Some(done) = current.take() {
        stats.finish_pool(done, &counts, top_n);
    }
    stats
}

fn write_csv(path: &Path, header: &str, rows: &[String]) -> AnyResult<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "{header}")?;
    for row in rows {
        writeln!(out, "{row}")?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pool_census: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: pool_census COMPOSITE_ROCKS [--out DIR] [--top N]");
        std::process::exit(2);
    };
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let out_dir = PathBuf::from(flag("--out").map_or("pool_census_out", |s| s.as_str()));
    let top_n: usize = flag("--top").and_then(|s| s.parse().ok()).unwrap_or(100);

    let mut options = Options::default();
    options.create_if_missing(false);
    let db = DB::open_for_read_only(&options, path, false)?;
    if db.get(COMPOSITE_FORMAT_MARKER)?.as_deref() != Some(b"1") {
        return Err(format!("{path} is not a curated-full composite-v1 store").into());
    }
    let manifest = db
        .get(COMPOSITE_COMPLETE_MARKER)?
        .ok_or_else(|| format!("{path} has no completion manifest; refusing a partial store"))?;
    let (manifest_keys, manifest_candidates) = if manifest.len() == 32 {
        (
            u64::from_le_bytes(manifest[..8].try_into().unwrap()),
            u64::from_le_bytes(manifest[8..16].try_into().unwrap()),
        )
    } else {
        (0, 0)
    };

    std::fs::create_dir_all(&out_dir)?;
    let stats = (0..=255u8)
        .into_par_iter()
        .map(|prefix| scan_prefix(&db, prefix, top_n))
        .reduce(Stats::new, Stats::merge);

    let mut top = stats.top;
    top.sort_unstable_by(|a, b| b.total.cmp(&a.total));
    top.truncate(top_n);

    println!(
        "keys={} candidates={} (manifest: keys={} candidates={}) meta={} malformed={}",
        stats.keys,
        stats.candidates,
        manifest_keys,
        manifest_candidates,
        stats.meta,
        stats.malformed
    );
    if stats.keys != manifest_keys || stats.candidates != manifest_candidates {
        eprintln!("WARNING: scan disagrees with completion manifest");
    }

    println!("\npool size histogram (candidates per key):");
    println!(
        "{:>14} {:>12} {:>16} {:>8} {:>8}",
        "size", "keys", "candidates", "%keys", "%cands"
    );
    for i in 0..NB {
        if stats.keys_by_poolsize[i] == 0 {
            continue;
        }
        println!(
            "{:>14} {:>12} {:>16} {:>7.3}% {:>7.3}%",
            bucket_label(i),
            stats.keys_by_poolsize[i],
            stats.cands_by_poolsize[i],
            stats.keys_by_poolsize[i] as f64 * 100.0 / stats.keys.max(1) as f64,
            stats.cands_by_poolsize[i] as f64 * 100.0 / stats.candidates.max(1) as f64,
        );
    }

    println!("\nper-gate-count cohorts (same key, same length = equal-length friend material):");
    println!(
        "{:>5} {:>12} {:>16} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "gates", "keys", "candidates", "mean/key", "keys>=2", "%keys>=2", "%cands>=2", "max cohort"
    );
    for l in 0..NG {
        if stats.keys_with_len[l] == 0 {
            continue;
        }
        println!(
            "{:>5} {:>12} {:>16} {:>12.2} {:>12} {:>9.2}% {:>11.2}% {:>12}",
            l,
            stats.keys_with_len[l],
            stats.cands_by_len[l],
            stats.cands_by_len[l] as f64 / stats.keys_with_len[l] as f64,
            stats.keys_with_len_ge2[l],
            stats.keys_with_len_ge2[l] as f64 * 100.0 / stats.keys_with_len[l] as f64,
            stats.cands_in_ge2[l] as f64 * 100.0 / stats.cands_by_len[l].max(1) as f64,
            stats.max_cohort[l],
        );
    }

    println!("\nkeys by shortest-candidate gate count:");
    for l in 0..NG {
        if stats.keys_by_mingates[l] > 0 {
            println!("  min_gates={l:>2}  keys={}", stats.keys_by_mingates[l]);
        }
    }

    println!("\ntop {} pools:", top.len());
    for (rank, t) in top.iter().enumerate() {
        let profile: Vec<String> = t.lens.iter().map(|(l, c)| format!("{l}:{c}")).collect();
        println!(
            "  #{:<3} {}  total={}  min_gates={}  [{}]",
            rank + 1,
            hex(&t.key),
            t.total,
            t.min_gates,
            profile.join(" ")
        );
    }

    write_csv(
        &out_dir.join("pool_size_hist.csv"),
        "size_class,keys,candidates",
        &(0..NB)
            .filter(|&i| stats.keys_by_poolsize[i] > 0)
            .map(|i| {
                format!(
                    "{},{},{}",
                    bucket_label(i),
                    stats.keys_by_poolsize[i],
                    stats.cands_by_poolsize[i]
                )
            })
            .collect::<Vec<_>>(),
    )?;
    write_csv(
        &out_dir.join("length_stats.csv"),
        "gates,keys,candidates,keys_ge2,candidates_in_ge2,max_cohort",
        &(0..NG)
            .filter(|&l| stats.keys_with_len[l] > 0)
            .map(|l| {
                format!(
                    "{},{},{},{},{},{}",
                    l,
                    stats.keys_with_len[l],
                    stats.cands_by_len[l],
                    stats.keys_with_len_ge2[l],
                    stats.cands_in_ge2[l],
                    stats.max_cohort[l]
                )
            })
            .collect::<Vec<_>>(),
    )?;
    let mut cohort_rows = Vec::new();
    for l in 0..NG {
        for b in 0..NB {
            if stats.cohort_hist[l][b] > 0 {
                cohort_rows.push(format!(
                    "{},{},{}",
                    l,
                    bucket_label(b),
                    stats.cohort_hist[l][b]
                ));
            }
        }
    }
    write_csv(
        &out_dir.join("cohort_hist.csv"),
        "gates,cohort_size_class,cohorts",
        &cohort_rows,
    )?;
    let mut matrix_rows = Vec::new();
    for i in 0..NB {
        for l in 0..NG {
            if stats.poolsize_x_len[i][l] > 0 {
                matrix_rows.push(format!(
                    "{},{},{}",
                    bucket_label(i),
                    l,
                    stats.poolsize_x_len[i][l]
                ));
            }
        }
    }
    write_csv(
        &out_dir.join("poolsize_x_length.csv"),
        "pool_size_class,gates,candidates",
        &matrix_rows,
    )?;
    write_csv(
        &out_dir.join("min_gates_hist.csv"),
        "min_gates,keys",
        &(0..NG)
            .filter(|&l| stats.keys_by_mingates[l] > 0)
            .map(|l| format!("{},{}", l, stats.keys_by_mingates[l]))
            .collect::<Vec<_>>(),
    )?;
    write_csv(
        &out_dir.join("top_keys.csv"),
        "rank,key,total,min_gates,length_profile",
        &top.iter()
            .enumerate()
            .map(|(rank, t)| {
                let profile: Vec<String> = t.lens.iter().map(|(l, c)| format!("{l}:{c}")).collect();
                format!(
                    "{},{},{},{},{}",
                    rank + 1,
                    hex(&t.key),
                    t.total,
                    t.min_gates,
                    profile.join(";")
                )
            })
            .collect::<Vec<_>>(),
    )?;
    eprintln!("CSVs written to {}", out_dir.display());
    Ok(())
}
