//! Classify multi-circuit ("friend") entries in a rocks DB by the gate
//! counts of their member circuits. Samples the first N entries of each of
//! the 256 shard ranges (uniform hash keys -> unbiased sample).
//!
//! usage: friend_stats <rocks_path> [entries_per_range]

use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: friend_stats <rocks_path> [per_range]");
    let per_range: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400_000);

    let db = DB::open_for_read_only(&Options::default(), path, false).expect("open rocks");

    let entries = AtomicU64::new(0);
    let multi = AtomicU64::new(0);
    // same gate count in every member circuit:
    let same_le6 = AtomicU64::new(0);
    let same_7 = AtomicU64::new(0);
    let same_8 = AtomicU64::new(0);
    let same_9 = AtomicU64::new(0);
    let same_10 = AtomicU64::new(0);
    let same_11p = AtomicU64::new(0);
    // mixed gate counts:
    let cross_le6_only = AtomicU64::new(0); // all members <=6 gates
    let cross_ge7_only = AtomicU64::new(0); // all members >=7 gates
    let cross_bridge = AtomicU64::new(0); // min<=6 AND max>=7  (m1-6 x m7-11)
    let bridge_examples: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let same_examples: Mutex<Vec<String>> = Mutex::new(Vec::new());

    (0..256u32).into_par_iter().for_each(|r| {
        let mut start = [0u8; 16];
        start[0] = r as u8;
        let it = db.iterator(IteratorMode::From(&start, Direction::Forward));
        let mut n = 0usize;
        for item in it {
            let Ok((k, v)) = item else { break };
            if k[0] != r as u8 {
                break;
            }
            entries.fetch_add(1, Ordering::Relaxed);
            // parse gate counts of member circuits
            let mut gs: Vec<usize> = Vec::new();
            let mut pos = 0usize;
            while pos < v.len() {
                let len = v[pos] as usize;
                pos += 1;
                if pos + len > v.len() {
                    break;
                }
                gs.push(len / 3);
                pos += len;
            }
            if gs.len() >= 2 {
                multi.fetch_add(1, Ordering::Relaxed);
                let gmin = *gs.iter().min().unwrap();
                let gmax = *gs.iter().max().unwrap();
                if gmin == gmax {
                    (match gmin {
                        0..=6 => &same_le6,
                        7 => &same_7,
                        8 => &same_8,
                        9 => &same_9,
                        10 => &same_10,
                        _ => &same_11p,
                    })
                    .fetch_add(1, Ordering::Relaxed);
                    if gmin >= 8 {
                        let mut ex = same_examples.lock().unwrap();
                        if ex.len() < 20 {
                            ex.push(format!("key={} gates={:?}", hex16(&k), gs));
                        }
                    }
                } else if gmax <= 6 {
                    cross_le6_only.fetch_add(1, Ordering::Relaxed);
                } else if gmin >= 7 {
                    cross_ge7_only.fetch_add(1, Ordering::Relaxed);
                } else {
                    cross_bridge.fetch_add(1, Ordering::Relaxed);
                    let mut ex = bridge_examples.lock().unwrap();
                    if ex.len() < 20 {
                        ex.push(format!("key={} gates={:?}", hex16(&k), gs));
                    }
                }
            }
            n += 1;
            if n >= per_range {
                break;
            }
        }
    });

    let e = entries.load(Ordering::Relaxed);
    let m = multi.load(Ordering::Relaxed);
    println!(
        "entries_sampled={} multi_circuit={} ({:.4}%)",
        e,
        m,
        100.0 * m as f64 / e as f64
    );
    println!(
        "same_gate_count: le6={} g7={} g8={} g9={} g10={} g11plus={}",
        same_le6.load(Ordering::Relaxed),
        same_7.load(Ordering::Relaxed),
        same_8.load(Ordering::Relaxed),
        same_9.load(Ordering::Relaxed),
        same_10.load(Ordering::Relaxed),
        same_11p.load(Ordering::Relaxed)
    );
    println!(
        "mixed_gate_count: within_m1_6={} within_m7_11={} bridge_m1_6_x_m7_11={}",
        cross_le6_only.load(Ordering::Relaxed),
        cross_ge7_only.load(Ordering::Relaxed),
        cross_bridge.load(Ordering::Relaxed)
    );
    println!("--- bridge examples (m1-6 x m7-11) ---");
    for s in bridge_examples.lock().unwrap().iter() {
        println!("{}", s);
    }
    println!("--- same-gate >=8 examples ---");
    for s in same_examples.lock().unwrap().iter() {
        println!("{}", s);
    }
}

fn hex16(k: &[u8]) -> String {
    k.iter().map(|b| format!("{:02x}", b)).collect()
}
