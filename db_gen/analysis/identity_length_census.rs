//! Why do frozen-store identities stop at 12 gates?
//!
//! For every ordered class pair the builder glues, record the RAW length
//! |a| + |b| and the post-simplify length, plus each qualifying class's
//! member-length profile. If multi-member classes never mix lengths summing
//! past 12 (and 7+-gate members sit in singleton classes), 13+-gate
//! identities cannot exist and long m1 candidates need a different source.
//!
//! ```text
//! identity_length_census FROZEN_REGULAR_DIR [--shards N]
//! ```

use local_mixing::circuit::CircuitSeq;
use local_mixing::circuit::cancel_adjacent_duplicates;
use local_mixing::db_mixing::frozen::scan_shard;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

const MAX_LEN: usize = 64;

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

#[derive(Default)]
struct Stats {
    raw: Vec<u64>,
    simplified: Vec<u64>,
    raw13_survive13: u64,
    raw13_total: u64,
    class_profile: BTreeMap<(usize, usize), u64>,
    members_by_len_in_multi: Vec<u64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            raw: vec![0; MAX_LEN],
            simplified: vec![0; MAX_LEN],
            members_by_len_in_multi: vec![0; MAX_LEN],
            ..Default::default()
        }
    }
    fn merge(&mut self, o: Stats) {
        for i in 0..MAX_LEN {
            self.raw[i] += o.raw[i];
            self.simplified[i] += o.simplified[i];
            self.members_by_len_in_multi[i] += o.members_by_len_in_multi[i];
        }
        self.raw13_survive13 += o.raw13_survive13;
        self.raw13_total += o.raw13_total;
        for (k, v) in o.class_profile {
            *self.class_profile.entry(k).or_insert(0) += v;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!("usage: identity_length_census FROZEN_REGULAR_DIR [--shards N]");
        std::process::exit(2);
    };
    let shards: usize = args
        .iter()
        .position(|a| a == "--shards")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
        .clamp(1, 256);

    let start = Instant::now();
    let total = Mutex::new(Stats::new());
    (0..shards).into_par_iter().for_each(|shard| {
        let mut s = Stats::new();
        scan_shard(dir, shard, &mut |value| {
            let members = class_members(value);
            if members.len() < 2 {
                return;
            }
            let lens: Vec<usize> = members.iter().map(|m| m.len() / 3).collect();
            let (lo, hi) = (*lens.iter().min().unwrap(), *lens.iter().max().unwrap());
            *s.class_profile.entry((lo, hi)).or_insert(0) += 1;
            for &l in &lens {
                s.members_by_len_in_multi[l.min(MAX_LEN - 1)] += 1;
            }
            let circuits: Vec<CircuitSeq> =
                members.iter().map(|b| CircuitSeq::from_blob(b)).collect();
            for left in 0..circuits.len() {
                for right in 0..circuits.len() {
                    if left == right {
                        continue;
                    }
                    let a = &circuits[left];
                    let b = &circuits[right];
                    let raw_len = a.gates.len() + b.gates.len();
                    s.raw[raw_len.min(MAX_LEN - 1)] += 1;
                    let mut gates = a.gates.clone();
                    gates.extend(b.gates.iter().rev().copied());
                    let mut identity = CircuitSeq { gates };
                    identity.canonicalize();
                    cancel_adjacent_duplicates(&mut identity.gates, None::<&mut Vec<()>>);
                    let n = identity.gates.len();
                    s.simplified[n.min(MAX_LEN - 1)] += 1;
                    if raw_len >= 13 {
                        s.raw13_total += 1;
                        if n >= 13 {
                            s.raw13_survive13 += 1;
                        }
                    }
                }
            }
        });
        total.lock().unwrap().merge(s);
    });

    let s = total.into_inner().unwrap();
    println!(
        "shards={shards} (of 256)  elapsed={:.1}s",
        start.elapsed().as_secs_f64()
    );
    println!("\nraw |a|+|b| vs post-simplify identity length:");
    for i in 0..MAX_LEN {
        if s.raw[i] > 0 || s.simplified[i] > 0 {
            println!(
                "  n={i:<3} raw={:>12} simplified={:>12}",
                s.raw[i], s.simplified[i]
            );
        }
    }
    println!(
        "\npairs with raw>=13: {}   still >=13 after simplify: {}",
        s.raw13_total, s.raw13_survive13
    );
    println!("\nmember lengths inside multi-member classes:");
    for i in 0..MAX_LEN {
        if s.members_by_len_in_multi[i] > 0 {
            println!("  len={i:<3} {:>12}", s.members_by_len_in_multi[i]);
        }
    }
    println!("\nclass (min-len, max-len) profile:");
    for ((lo, hi), count) in &s.class_profile {
        println!("  ({lo:>2},{hi:>2}) {count:>12}");
    }
}
