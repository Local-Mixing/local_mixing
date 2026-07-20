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
                    let wire_terms = polys.iter().map(|p| p.len()).max().unwrap_or(0);
                    let total_terms: usize = polys.iter().map(|p| p.len()).sum();
                    max_wire_terms = max_wire_terms.max(wire_terms);
                    max_total_terms = max_total_terms.max(total_terms);
                    *terms_hist.entry(wire_terms.next_power_of_two()).or_insert(0) += 1;
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
