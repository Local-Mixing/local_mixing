//! Makeup census of a frozen replacement store: how many entries
//! (permutations) it holds, how many friend circuits each permutation has,
//! and the gate-count profile of those circuits — overall, by multiplicity
//! bucket, and for the top individual permutations.
//!
//! Usage: frozen_census <store_dir> [--shards a,b,..]
//! Default: every shard_XX.frz present in the dir.

use local_mixing::replace::frozen::scan_shard;
use std::collections::BTreeMap;

/// Multiplicity bucket: exact 1..=8, then pow2 ranges (9-16, 17-32, ...).
fn mult_bucket(n: usize) -> (usize, usize) {
    if n <= 8 {
        (n, n)
    } else {
        let mut hi = 16usize;
        while hi < n {
            hi *= 2;
        }
        (hi / 2 + 1, hi)
    }
}

fn fmt_bucket((lo, hi): (usize, usize)) -> String {
    if lo == hi { format!("{lo}") } else { format!("{lo}-{hi}") }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: frozen_census <dir> [--shards a,b,..]");
    let mut shards: Option<Vec<usize>> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--shards" => {
                shards = Some(
                    args.next()
                        .expect("missing shard list")
                        .split(',')
                        .map(|s| s.parse().expect("bad shard"))
                        .collect(),
                )
            }
            _ => panic!("unknown arg {a}"),
        }
    }
    let shards = shards.unwrap_or_else(|| {
        let mut v: Vec<usize> = std::fs::read_dir(&dir)
            .expect("read store dir")
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                let hex = name.strip_prefix("shard_")?.strip_suffix(".frz")?;
                usize::from_str_radix(hex, 16).ok()
            })
            .collect();
        v.sort_unstable();
        v
    });
    eprintln!("[census] {} shards in {dir}", shards.len());

    let mut entries = 0u64;
    let mut friends_total = 0u64;
    let mut parse_fail = 0u64;
    // circuits-per-permutation histogram (exact count -> permutations)
    let mut mult_hist: BTreeMap<usize, u64> = BTreeMap::new();
    // gate-count histogram over ALL friend circuits
    let mut gates_hist: BTreeMap<usize, u64> = BTreeMap::new();
    // per multiplicity-bucket gate-count histogram
    let mut by_bucket: BTreeMap<(usize, usize), BTreeMap<usize, u64>> = BTreeMap::new();
    // top-K individual permutations by multiplicity
    const K: usize = 12;
    let mut top: Vec<(usize, BTreeMap<usize, u64>)> = Vec::new();

    for (i, &s) in shards.iter().enumerate() {
        let t0 = std::time::Instant::now();
        scan_shard(&dir, s, &mut |value: &[u8]| {
            let mut counts: Vec<usize> = Vec::new();
            let mut pos = 0usize;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if len == 0 || len % 3 != 0 || pos + len > value.len() {
                    parse_fail += 1;
                    break;
                }
                counts.push(len / 3);
                pos += len;
            }
            if counts.is_empty() {
                return;
            }
            entries += 1;
            friends_total += counts.len() as u64;
            *mult_hist.entry(counts.len()).or_insert(0) += 1;
            let bkt = mult_bucket(counts.len());
            let bh = by_bucket.entry(bkt).or_default();
            for &g in &counts {
                *gates_hist.entry(g).or_insert(0) += 1;
                *bh.entry(g).or_insert(0) += 1;
            }
            if top.len() < K || counts.len() > top.last().unwrap().0 {
                let mut gh: BTreeMap<usize, u64> = BTreeMap::new();
                for &g in &counts {
                    *gh.entry(g).or_insert(0) += 1;
                }
                top.push((counts.len(), gh));
                top.sort_by(|a, b| b.0.cmp(&a.0));
                top.truncate(K);
            }
        });
        eprintln!(
            "[census] shard {s:02x} done ({}/{}) — {entries} entries so far, {:.1}s",
            i + 1,
            shards.len(),
            t0.elapsed().as_secs_f64()
        );
    }

    println!("== frozen store census: {dir} ==");
    println!("permutations (entries): {entries}");
    println!("circuits (friends) total: {friends_total}");
    println!(
        "mean circuits/permutation: {:.3}   parse failures: {parse_fail}",
        friends_total as f64 / entries.max(1) as f64
    );

    println!("\n-- circuits-per-permutation histogram (bucketed) --");
    let mut bucketed: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    for (&m, &c) in &mult_hist {
        *bucketed.entry(mult_bucket(m)).or_insert(0) += c;
    }
    for (&b, &c) in &bucketed {
        println!("{:>9} circuits : {c:>12} permutations", fmt_bucket(b));
    }

    println!("\n-- gate-count histogram over ALL circuits --");
    for (&g, &c) in &gates_hist {
        println!("{g:>3} gates : {c:>12}");
    }

    println!("\n-- gate-count histograms by multiplicity bucket --");
    for (&b, h) in &by_bucket {
        let total: u64 = h.values().sum();
        let line: Vec<String> = h.iter().map(|(g, c)| format!("{g}g:{c}")).collect();
        println!("[{} circuits/perm] {} circuits: {}", fmt_bucket(b), total, line.join(" "));
    }

    println!("\n-- top {} permutations by circuit count --", top.len());
    for (m, h) in &top {
        let line: Vec<String> = h.iter().map(|(g, c)| format!("{g}g:{c}")).collect();
        println!("{m:>6} circuits: {}", line.join(" "));
    }
}
