//! Sample value statistics from a rocks DB of (16B key -> length-prefixed
//! circuit blobs). Reads the first N entries of each of the 256 shard ranges
//! (keys are uniform hashes, so this is an unbiased ~0.2% sample).
//!
//! usage: value_stats <rocks_path> [entries_per_range]

use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::sync::atomic::{AtomicU64, Ordering};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: value_stats <rocks_path> [per_range]");
    let per_range: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);

    let db = DB::open_for_read_only(&Options::default(), path, false).expect("open rocks");

    let entries = AtomicU64::new(0);
    let vbytes = AtomicU64::new(0);
    let circuits = AtomicU64::new(0);
    let gates = AtomicU64::new(0);
    let bad = AtomicU64::new(0);
    // circuits-per-value histogram: 1, 2, 3, 4+
    let c1 = AtomicU64::new(0);
    let c2 = AtomicU64::new(0);
    let c3 = AtomicU64::new(0);
    let c4p = AtomicU64::new(0);
    // max wire index per circuit: <16, <20, <24, <32, >=32
    let w16 = AtomicU64::new(0);
    let w20 = AtomicU64::new(0);
    let w24 = AtomicU64::new(0);
    let w32 = AtomicU64::new(0);
    let wbig = AtomicU64::new(0);
    // gates per circuit histogram: <=5,6,7,8,9,10,11+
    let mut ghist: Vec<AtomicU64> = Vec::new();
    for _ in 0..12 {
        ghist.push(AtomicU64::new(0));
    }

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
            vbytes.fetch_add(v.len() as u64, Ordering::Relaxed);
            let mut pos = 0usize;
            let mut nc = 0u64;
            while pos < v.len() {
                let len = v[pos] as usize;
                pos += 1;
                if pos + len > v.len() {
                    bad.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                let blob = &v[pos..pos + len];
                pos += len;
                nc += 1;
                let g = len / 3;
                gates.fetch_add(g as u64, Ordering::Relaxed);
                let gi = g.min(11);
                ghist[gi].fetch_add(1, Ordering::Relaxed);
                let mw = blob.iter().copied().max().unwrap_or(0);
                (if mw < 16 {
                    &w16
                } else if mw < 20 {
                    &w20
                } else if mw < 24 {
                    &w24
                } else if mw < 32 {
                    &w32
                } else {
                    &wbig
                })
                .fetch_add(1, Ordering::Relaxed);
            }
            circuits.fetch_add(nc, Ordering::Relaxed);
            (match nc {
                1 => &c1,
                2 => &c2,
                3 => &c3,
                _ => &c4p,
            })
            .fetch_add(1, Ordering::Relaxed);
            n += 1;
            if n >= per_range {
                break;
            }
        }
    });

    let e = entries.load(Ordering::Relaxed);
    let vb = vbytes.load(Ordering::Relaxed);
    let ci = circuits.load(Ordering::Relaxed);
    let ga = gates.load(Ordering::Relaxed);
    println!("entries_sampled={}", e);
    println!(
        "value_bytes_total={} avg_value_bytes={:.2}",
        vb,
        vb as f64 / e as f64
    );
    println!(
        "circuits_total={} avg_circuits_per_value={:.4} avg_gates_per_circuit={:.3}",
        ci,
        ci as f64 / e as f64,
        ga as f64 / ci as f64
    );
    println!(
        "circuits_per_value_hist: 1={} 2={} 3={} 4plus={}",
        c1.load(Ordering::Relaxed),
        c2.load(Ordering::Relaxed),
        c3.load(Ordering::Relaxed),
        c4p.load(Ordering::Relaxed)
    );
    print!("gates_per_circuit_hist:");
    for (i, h) in ghist.iter().enumerate() {
        let v = h.load(Ordering::Relaxed);
        if v > 0 {
            print!(" {}{}={}", if i == 11 { ">=" } else { "" }, i, v);
        }
    }
    println!();
    println!(
        "max_wire_hist: lt16={} lt20={} lt24={} lt32={} ge32={}",
        w16.load(Ordering::Relaxed),
        w20.load(Ordering::Relaxed),
        w24.load(Ordering::Relaxed),
        w32.load(Ordering::Relaxed),
        wbig.load(Ordering::Relaxed)
    );
    println!("unparseable={}", bad.load(Ordering::Relaxed));
}
