//! Degree census of a frozen replacement store.
//!
//! Walks shard buckets sequentially (`frozen::scan_shard`), decodes each
//! entry's FIRST friend (all friends under one key compute the same function,
//! hence share one ANF degree) and histograms exact max output degree by gate
//! count. Used to pick `--db-max-degree` for the fmix DB move: a window whose
//! degree exceeds the store's true maximum can never match.
//!
//! Usage:
//!   frozen_degree_scan <store_dir> [--shards 0,1,2] [--sample K] [--limit N]
//!
//! --sample K keeps every K-th entry; --limit N stops a shard after N kept
//! entries. Shards are hash-partitioned, so one shard is an unbiased sample.

use local_mixing::postmix::db_replace::db_g57_to_xgate;
use local_mixing::postmix::xpoly::{XPolyBudget, xgates_to_polynomial};
use local_mixing::replace::frozen::scan_shard;
use std::collections::BTreeMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: frozen_degree_scan <dir> [--shards a,b,..] [--sample K] [--limit N]");
    let mut shards: Vec<usize> = vec![0];
    let mut sample = 1u64;
    let mut limit = u64::MAX;
    while let Some(a) = args.next() {
        let mut next = || args.next().expect("missing value");
        match a.as_str() {
            "--shards" => {
                shards = next().split(',').map(|s| s.parse().expect("bad shard")).collect()
            }
            "--sample" => sample = next().parse().expect("bad --sample"),
            "--limit" => limit = next().parse().expect("bad --limit"),
            _ => panic!("unknown arg {a}"),
        }
    }

    let budget = XPolyBudget::default();
    // (first-friend gate count, exact ANF degree) -> entries
    let mut hist: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    // (first-friend gate count, canonical wire span) -> entries
    let mut span_hist: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    // JOINT span x degree, and the term census restricted to the WIDE slice:
    // the question these answer is whether the entries a 24-wire verify cap
    // excludes are also high-degree (expensive to verify another way) or in
    // fact low-degree (cheap by a polynomial identity check).
    let mut joint: BTreeMap<(usize, usize), u64> = BTreeMap::new();
    let mut wide_terms_max = 0usize;
    let mut wide_total_max = 0usize;
    let mut wide_n = 0u64;
    // Poly size census: worst per-wire and per-entry term counts in the store.
    let mut max_wire_terms = 0usize;
    let mut max_total_terms = 0usize;
    let mut terms_hist: BTreeMap<usize, u64> = BTreeMap::new(); // pow2 bucket of max wire terms
    let mut kept = 0u64;
    let mut budget_fail = 0u64;
    let mut parse_fail = 0u64;

    for &s in &shards {
        let t0 = std::time::Instant::now();
        let mut walked = 0u64;
        let mut shard_kept = 0u64;
        scan_shard(&dir, s, &mut |value: &[u8]| {
            walked += 1;
            if shard_kept >= limit || (sample > 1 && walked % sample != 0) {
                return;
            }
            // Parse every friend: [byte_len][3-byte g57 triples]... Degree comes
            // from the first (all friends of a key compute the same function);
            // span is min over friends of DISTINCT wires touched — friends may
            // burn scratch wires, but the function's support can't exceed the
            // narrowest circuit that computes it.
            let mut friends: Vec<Vec<_>> = Vec::new();
            let mut pos = 0usize;
            while pos < value.len() {
                let len = value[pos] as usize;
                pos += 1;
                if len % 3 != 0 || pos + len > value.len() {
                    break;
                }
                friends.push(
                    value[pos..pos + len]
                        .chunks(3)
                        .map(|c| db_g57_to_xgate([c[0] as u16, c[1] as u16, c[2] as u16]))
                        .collect(),
                );
                pos += len;
            }
            let Some(gates) = friends.first().cloned() else {
                parse_fail += 1;
                return;
            };
            let nw = gates
                .iter()
                .flat_map(|g| std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w)))
                .max()
                .map_or(0, |w| w as usize + 1);
            let min_span = friends
                .iter()
                .map(|f| {
                    let mut ws: Vec<u16> = f
                        .iter()
                        .flat_map(|g| {
                            std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w))
                        })
                        .collect();
                    ws.sort_unstable();
                    ws.dedup();
                    ws.len()
                })
                .min()
                .unwrap_or(0);
            shard_kept += 1;
            kept += 1;
            *span_hist.entry((gates.len(), min_span)).or_insert(0) += 1;
            match xgates_to_polynomial(&gates, nw, budget) {
                Ok(polys) => {
                    let deg = polys
                        .iter()
                        .flat_map(|p| p.iter().map(|m| m.count_ones() as usize))
                        .max()
                        .unwrap_or(0);
                    *hist.entry((gates.len(), deg)).or_insert(0) += 1;
                    *joint.entry((min_span, deg)).or_insert(0) += 1;
                    let wire_terms = polys.iter().map(|p| p.len()).max().unwrap_or(0);
                    let total_terms: usize = polys.iter().map(|p| p.len()).sum();
                    max_wire_terms = max_wire_terms.max(wire_terms);
                    max_total_terms = max_total_terms.max(total_terms);
                    *terms_hist.entry(wire_terms.next_power_of_two()).or_insert(0) += 1;
                    if min_span >= 24 {
                        wide_n += 1;
                        wide_terms_max = wide_terms_max.max(wire_terms);
                        wide_total_max = wide_total_max.max(total_terms);
                    }
                }
                Err(_) => budget_fail += 1,
            }
        });
        eprintln!(
            "shard {s:02x}: walked {walked} entries, kept {shard_kept}, {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }

    println!("entries analyzed: {kept} (budget_fail {budget_fail}, parse_fail {parse_fail})");
    {
        // The wide slice, which a 24-wire exhaustive verifier cannot touch.
        let jdmax = joint.keys().map(|&(_, d)| d).max().unwrap_or(0);
        println!("\n=== JOINT span x degree (span >= 20 only) ===");
        print!("{:>6}", "span");
        for d in 0..=jdmax {
            print!(" {:>8}", format!("deg{d}"));
        }
        println!("{:>10}", "total");
        let smax = joint.keys().map(|&(s, _)| s).max().unwrap_or(0);
        for sp in 20..=smax {
            let row: Vec<u64> = (0..=jdmax).map(|d| *joint.get(&(sp, d)).unwrap_or(&0)).collect();
            let tot: u64 = row.iter().sum();
            if tot == 0 {
                continue;
            }
            print!("{sp:>6}");
            for v in &row {
                print!(" {v:>8}");
            }
            println!("{tot:>10}");
        }
        let wide: Vec<(usize, u64)> = (0..=jdmax)
            .map(|d| (d, joint.iter().filter(|(k, _)| k.0 >= 24 && k.1 == d).map(|(_, v)| *v).sum::<u64>()))
            .filter(|&(_, v)| v > 0)
            .collect();
        let wtot: u64 = wide.iter().map(|(_, v)| v).sum();
        println!("\nspan >= 24: {wtot} entries; degree histogram:");
        for (d, v) in &wide {
            println!("   deg {d}: {v}  ({:.1}%)", 100.0 * *v as f64 / wtot.max(1) as f64);
        }
        println!("span >= 24 worst poly size: max per-wire terms {wide_terms_max}, max total terms {wide_total_max} (n={wide_n})");
    }
    let gmax = hist.keys().map(|&(g, _)| g).max().unwrap_or(0);
    let dmax = hist.keys().map(|&(_, d)| d).max().unwrap_or(0);
    println!("max ANF degree seen: {dmax}");
    print!("{:>6}", "gates");
    for d in 0..=dmax {
        print!(" {:>10}", format!("deg{d}"));
    }
    println!();
    for g in 0..=gmax {
        if !(0..=dmax).any(|d| hist.contains_key(&(g, d))) {
            continue;
        }
        print!("{g:>6}");
        for d in 0..=dmax {
            print!(" {:>10}", hist.get(&(g, d)).copied().unwrap_or(0));
        }
        println!();
    }

    println!("\nmax per-wire poly terms: {max_wire_terms}; max per-entry total terms: {max_total_terms}");
    println!("per-wire terms distribution (pow2 buckets): {terms_hist:?}");

    let smax = span_hist.keys().map(|&(_, s)| s).max().unwrap_or(0);
    println!("\nmax canonical wire span seen: {smax}");
    print!("{:>6}", "gates");
    for s in 0..=smax {
        print!(" {:>10}", format!("span{s}"));
    }
    println!();
    let sg_max = span_hist.keys().map(|&(g, _)| g).max().unwrap_or(0);
    for g in 0..=sg_max {
        if !(0..=smax).any(|s| span_hist.contains_key(&(g, s))) {
            continue;
        }
        print!("{g:>6}");
        for s in 0..=smax {
            print!(" {:>10}", span_hist.get(&(g, s)).copied().unwrap_or(0));
        }
        println!();
    }
}
