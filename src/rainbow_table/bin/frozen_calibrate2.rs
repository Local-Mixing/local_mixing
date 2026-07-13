//! Calibration v2 — no structural assumptions. Sweeps empirical context
//! models over sampled circuits and reports achievable bits/gate for each,
//! plus projected total table size. Also dumps example circuits and checks
//! gate-ordering properties.
//!
//! usage: frozen_calibrate2 <rocks_path> [entries_per_range]

use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Default, Clone)]
struct Stats {
    entries: u64,
    circuits: u64,
    gates: u64,
    // model frequency tables, keyed by (context, symbol)
    m1: HashMap<u32, u64>, // ctx=role                sym=wire
    m2: HashMap<u32, u64>, // ctx=(role, gate_idx)    sym=wire
    m3: HashMap<u32, u64>, // ctx=(role, w)           sym=wire
    m4: HashMap<u32, u64>, // ctx=(role, w, gate_idx) sym=wire
    m5: HashMap<u32, u64>, // ctx=(w,)                sym=whole gate triple
    m6: HashMap<u32, u64>, // ctx=(w, gate_idx)       sym=whole gate triple
    gc: HashMap<u32, u64>, // circuit header: (gates, w, chained)
    sorted_adj: u64,
    unsorted_adj: u64,
    examples: Vec<String>,
}

fn bump(m: &mut HashMap<u32, u64>, k: u32) {
    *m.entry(k).or_insert(0) += 1;
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        self.entries += o.entries;
        self.circuits += o.circuits;
        self.gates += o.gates;
        for (m, om) in [
            (&mut self.m1, o.m1),
            (&mut self.m2, o.m2),
            (&mut self.m3, o.m3),
            (&mut self.m4, o.m4),
            (&mut self.m5, o.m5),
            (&mut self.m6, o.m6),
            (&mut self.gc, o.gc),
        ] {
            for (k, v) in om {
                *m.entry(k).or_insert(0) += v;
            }
        }
        self.sorted_adj += o.sorted_adj;
        self.unsorted_adj += o.unsorted_adj;
        if self.examples.len() < 20 {
            self.examples.extend(o.examples);
            self.examples.truncate(20);
        }
    }
}

/// conditional entropy in total bits given (ctx<<16 | sym) keyed table
fn cond_entropy_bits(m: &HashMap<u32, u64>) -> f64 {
    let mut ctx_tot: HashMap<u32, u64> = HashMap::new();
    for (&k, &v) in m {
        *ctx_tot.entry(k >> 16).or_insert(0) += v;
    }
    let mut bits = 0.0f64;
    for (&k, &v) in m {
        let t = ctx_tot[&(k >> 16)] as f64;
        let p = v as f64 / t;
        bits += -(v as f64) * p.log2();
    }
    bits
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: frozen_calibrate2 <rocks> [per_range]");
    let per_range: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(150_000);

    let db = DB::open_for_read_only(&Options::default(), path, false).expect("open rocks");
    let global = Mutex::new(Stats::default());

    (0..256u32).into_par_iter().for_each(|r| {
        let mut s = Stats::default();
        let mut start = [0u8; 16];
        start[0] = r as u8;
        let it = db.iterator(IteratorMode::From(&start, Direction::Forward));
        let mut n = 0usize;
        for item in it {
            let Ok((key, v)) = item else { break };
            if key[0] != r as u8 {
                break;
            }
            s.entries += 1;
            let mut pos = 0usize;
            let mut blobs: Vec<&[u8]> = Vec::new();
            while pos < v.len() {
                let len = v[pos] as usize;
                pos += 1;
                if pos + len > v.len() {
                    break;
                }
                blobs.push(&v[pos..pos + len]);
                pos += len;
            }
            let nb = blobs.len();
            for (bi, blob) in blobs.iter().enumerate() {
                s.circuits += 1;
                let g = blob.len() / 3;
                s.gates += g as u64;
                let w = blob.iter().copied().max().unwrap_or(0) as u32 + 1;
                let chained = (bi + 1 < nb) as u32;
                bump(&mut s.gc, ((g as u32) << 8 | w << 1 | chained) << 16);
                let gates: Vec<&[u8]> = blob.chunks(3).collect();
                for (gi, gate) in gates.iter().enumerate() {
                    let gi_c = (gi as u32).min(11);
                    let triple =
                        (gate[0] as u32) << 10 | (gate[1] as u32) << 5 | gate[2] as u32;
                    bump(&mut s.m5, w << 16 | triple);
                    bump(&mut s.m6, (w << 4 | gi_c) << 16 | triple);
                    for (role, &wb) in gate.iter().enumerate() {
                        let wire = wb as u32;
                        let ro = role as u32;
                        bump(&mut s.m1, ro << 16 | wire);
                        bump(&mut s.m2, (ro << 4 | gi_c) << 16 | wire);
                        bump(&mut s.m3, (ro << 6 | w) << 16 | wire);
                        bump(&mut s.m4, ((ro << 6 | w) << 4 | gi_c) << 16 | wire);
                    }
                    if gi + 1 < gates.len() {
                        if gates[gi] <= gates[gi + 1] {
                            s.sorted_adj += 1;
                        } else {
                            s.unsorted_adj += 1;
                        }
                    }
                }
                if s.examples.len() < 20 && r == 0 {
                    s.examples.push(format!("{:?}", gates));
                }
            }
            n += 1;
            if n >= per_range {
                break;
            }
        }
        global.lock().unwrap().merge(s);
    });

    let s = global.lock().unwrap();
    let g = s.gates as f64;
    let e = s.entries as f64;
    println!("entries={} circuits={} gates={} avg_gates={:.3}", s.entries, s.circuits, s.gates, g / s.circuits as f64);
    println!(
        "adjacent_gate_pairs lexicographically sorted: {:.2}%",
        100.0 * s.sorted_adj as f64 / (s.sorted_adj + s.unsorted_adj) as f64
    );

    let hdr_bits = cond_entropy_bits(&s.gc); // per circuit
    println!("header (gates,w,chain) entropy: {:.3} bits/circuit", hdr_bits / s.circuits as f64);

    let models = [
        ("M1 wire|role            ", cond_entropy_bits(&s.m1)),
        ("M2 wire|role,gate_idx   ", cond_entropy_bits(&s.m2)),
        ("M3 wire|role,w          ", cond_entropy_bits(&s.m3)),
        ("M4 wire|role,w,gate_idx ", cond_entropy_bits(&s.m4)),
    ];
    for (name, bits) in models {
        let value_bits_pe = (bits + hdr_bits) / e;
        let per_entry = 44.0 / 8.0 + 0.083 + value_bits_pe / 8.0;
        println!(
            "[{name}] {:.3} bits/gate -> {:.1} B/entry -> {:.1} GB",
            bits / g,
            per_entry,
            per_entry * 22_637_566_672.0 / 1e9
        );
    }
    // gate-as-symbol models
    for (name, m) in [("M5 triple|w        ", &s.m5), ("M6 triple|w,gate_idx", &s.m6)] {
        let bits = cond_entropy_bits(m);
        let value_bits_pe = (bits + hdr_bits) / e;
        let per_entry = 44.0 / 8.0 + 0.083 + value_bits_pe / 8.0;
        println!(
            "[{name}] {:.3} bits/gate -> {:.1} B/entry -> {:.1} GB",
            bits / g,
            per_entry,
            per_entry * 22_637_566_672.0 / 1e9
        );
    }
    println!("--- example circuits (gate triples) ---");
    for ex in s.examples.iter().take(8) {
        println!("{}", ex);
    }
}
