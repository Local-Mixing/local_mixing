//! List the frozen-store entries that contain a friend with <= N gates (the
//! permutations realizable by very short circuits) and, per minimal gate
//! count, histogram their multiplicities.
//!
//! Note the store's bounded rebuild caps entries at 20 friends, and a short
//! friend could in principle fall past the cap in some OTHER entry — so this
//! finds entries where a short friend SURVIVED, the practical notion of "the
//! store's k-gate permutations".
//!
//! Usage: frozen_find_small <store_dir> [max_gates=2]

use local_mixing::db_mixing::frozen::scan_shard;
use std::collections::BTreeMap;

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = a.next().expect("usage: frozen_find_small <dir> [max_gates]");
    let mut quiet = false;
    let maxg: usize = a
        .next()
        .and_then(|s| if s == "--quiet" { quiet = true; None } else { s.parse().ok() })
        .unwrap_or(2);
    if a.next().as_deref() == Some("--quiet") {
        quiet = true;
    }
    let shards: Vec<usize> = {
        let mut v: Vec<usize> = std::fs::read_dir(&dir)
            .expect("read store dir")
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                usize::from_str_radix(name.strip_prefix("shard_")?.strip_suffix(".frz")?, 16).ok()
            })
            .collect();
        v.sort_unstable();
        v
    };
    // min friend gate count -> (multiplicity -> entries)
    let mut mult_by_min: BTreeMap<usize, BTreeMap<usize, u64>> = BTreeMap::new();
    for &s in &shards {
        scan_shard(&dir, s, &mut |value: &[u8]| {
            let mut counts: Vec<usize> = Vec::new();
            let mut pos = 0usize;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if len == 0 || len % 3 != 0 || pos + len > value.len() {
                    return;
                }
                counts.push(len / 3);
                pos += len;
            }
            let Some(&mn) = counts.iter().min() else { return };
            if mn <= maxg {
                let mut gh: BTreeMap<usize, usize> = BTreeMap::new();
                for &g in &counts {
                    *gh.entry(g).or_insert(0) += 1;
                }
                if !quiet {
                    let line: Vec<String> = gh.iter().map(|(g, c)| format!("{g}g:{c}")).collect();
                    println!(
                        "min {mn}g | shard {s:02x} | {:>2} circuits: {}",
                        counts.len(),
                        line.join(" ")
                    );
                }
                *mult_by_min
                    .entry(mn)
                    .or_default()
                    .entry(counts.len())
                    .or_insert(0) += 1;
            }
        });
    }
    println!("\n== multiplicity histograms by minimal friend size ==");
    for (mn, h) in &mult_by_min {
        let total: u64 = h.values().sum();
        println!("-- permutations with a {mn}-gate circuit: {total} entries --");
        for (m, c) in h {
            println!("   {m:>2} circuits : {c} permutations");
        }
    }
}
