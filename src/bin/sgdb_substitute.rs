//! Replace every g57 gate of a circuit with a random SGDB circuit (a
//! multi-gate implementation of the single-g57-gate permutation), then
//! rerandomize gate order with the commuting shuffle, and verify functional
//! equivalence against the input on random states.
//!
//! Wire mapping per substitution: the SGDB circuit computes
//! wire0 ^= wire1 OR NOT wire2 (identity on its scratch wires 3..w), so
//! canonical wires (0,1,2) map onto the host gate's (target, pos, neg) and
//! every scratch wire maps to a distinct random OTHER host wire — any
//! injective choice preserves the function.
//!
//! Only comp=1 two-control gates (original g57s) are replaced; every other
//! gate (CNOTs, NOTs, split residues) passes through unchanged.
//!
//! Scratch placement is BALANCED by default: pass 1 draws every gate's SGDB
//! circuit and tallies the fixed per-wire slot usage (kept gates, plus each
//! replacement's canonical (0,1,2) slot multiplicities landing on the host
//! gate's (t,x,y)); pass 2 places each scratch wire by power-of-8-choices
//! against the running usage counter (sample 8 eligible wires, take the
//! least-loaded), weighted by the scratch wire's own slot multiplicity. This
//! flattens per-wire usage globally — in particular equalizing the halves
//! 0..127 vs 128..255, which the host gates alone use ~3:1. --no-balance
//! restores the old uniform scratch draw.
//!
//! Usage: sgdb_substitute <in.mpmct1> <sgdb.sgdb1> <out.mpmct1> [--seed S]
//!                        [--verify K=200] [--no-shuffle] [--no-balance]

use local_mixing::circuit::circuit::U1024;
use local_mixing::postmix::format::{read_mpmct, write_mpmct};
use local_mixing::postmix::xgate::{XGate, eval_u1024};
use local_mixing::replace::gadgets::commuting_shuffle_order;
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

type G57 = [u16; 3];

fn read_sgdb(path: &str) -> Vec<Vec<G57>> {
    let s = std::fs::read_to_string(path).expect("read sgdb");
    let mut lines = s.lines();
    let header = lines.next().expect("empty sgdb");
    assert!(header.starts_with("sgdb1 "), "bad sgdb header");
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(';')
                .map(|t| {
                    let v: Vec<u16> =
                        t.split(',').map(|x| x.parse().expect("bad triple")).collect();
                    [v[0], v[1], v[2]]
                })
                .collect()
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let inp = a.next().expect("usage: sgdb_substitute <in> <sgdb> <out> [--seed S] [--verify K]");
    let sgdb_path = a.next().expect("missing sgdb path");
    let outp = a.next().expect("missing output path");
    let (mut seed, mut verify_k, mut shuffle, mut balance) = (1u64, 200usize, true, true);
    while let Some(arg) = a.next() {
        match arg.as_str() {
            "--seed" => seed = a.next().unwrap().parse().expect("bad seed"),
            "--verify" => verify_k = a.next().unwrap().parse().expect("bad verify"),
            "--no-shuffle" => shuffle = false,
            "--no-balance" => balance = false,
            _ => panic!("unknown arg {arg}"),
        }
    }
    let (gates, wires) = read_mpmct(&inp).expect("read input");
    let sgdb = read_sgdb(&sgdb_path);
    assert!(!sgdb.is_empty(), "empty SGDB");
    let mut rng = StdRng::seed_from_u64(seed);

    // Pass 1: decide keep-vs-replace and DRAW each replacement's SGDB circuit,
    // tallying the FIXED per-wire slot usage: kept gates in full, and each
    // replacement's canonical (0,1,2) slot multiplicities on the host (t,x,y).
    // An original g57 is comp=1 with exactly two controls: (x,false),(y,true)
    // encoding t ^= x OR NOT y.
    enum Plan {
        Keep,
        Replace { x: u16, y: u16, cir: usize },
    }
    let mut usage = vec![0u64; wires];
    let mut plans: Vec<Plan> = Vec::with_capacity(gates.len());
    for g in &gates {
        if g.comp && g.ctrls.len() == 2 {
            let xw = g.ctrls.iter().find(|&&(_, p)| !p).map(|&(w, _)| w);
            let yw = g.ctrls.iter().find(|&&(_, p)| p).map(|&(w, _)| w);
            if let (Some(x), Some(y)) = (xw, yw) {
                let ci = rng.random_range(0..sgdb.len());
                for &[t, p, n] in &sgdb[ci] {
                    for w in [t, p, n] {
                        if w < 3 {
                            usage[[g.target, x, y][w as usize] as usize] += 1;
                        }
                    }
                }
                plans.push(Plan::Replace { x, y, cir: ci });
                continue;
            }
        }
        usage[g.target as usize] += 1;
        for &(w, _) in &g.ctrls {
            usage[w as usize] += 1;
        }
        plans.push(Plan::Keep);
    }

    // Pass 2: place scratch wires and emit. Balanced mode assigns each
    // scratch wire (heaviest multiplicity first) by power-of-8-choices on
    // the running usage counter; unbalanced mode is the old uniform draw.
    let mut out: Vec<XGate> = Vec::with_capacity(gates.len() * 12);
    let (mut replaced, mut passed) = (0u64, 0u64);
    // Per-output-gate litter ids: each host gate (replaced block or
    // passthrough singleton) is one litter, so the substitution itself is
    // recorded as litters for a downstream --litter-ban.
    let mut lits: Vec<u64> = Vec::with_capacity(gates.len() * 12);
    let mut lid = 0u64;
    for (g, plan) in gates.iter().zip(&plans) {
        lid += 1;
        let Plan::Replace { x, y, cir } = *plan else {
            out.push(g.clone());
            lits.push(lid);
            passed += 1;
            continue;
        };
        let (x, y) = (x, y);
        let cir = &sgdb[cir];
        let nw = cir.iter().flat_map(|t| t.iter()).max().unwrap() + 1;
        let n_scratch = nw as usize - 3;
        // slot multiplicity of each canonical scratch wire in this circuit
        let mut mult = vec![0u64; n_scratch];
        for &[t, p, n] in cir {
            for w in [t, p, n] {
                if w >= 3 {
                    mult[w as usize - 3] += 1;
                }
            }
        }
        let mut scratch: Vec<u16> = vec![u16::MAX; n_scratch];
        let mut order: Vec<usize> = (0..n_scratch).collect();
        order.sort_by_key(|&s| std::cmp::Reverse(mult[s]));
        for s in order {
            let pick = if balance {
                // 8 distinct-ish uniform candidates, keep the least-loaded
                let mut best: Option<u16> = None;
                let mut tries = 0;
                while tries < 8 {
                    let w = rng.random_range(0..wires as u16);
                    if w == g.target || w == x || w == y || scratch.contains(&w) {
                        continue;
                    }
                    tries += 1;
                    if best.is_none_or(|b| usage[w as usize] < usage[b as usize]) {
                        best = Some(w);
                    }
                }
                best.unwrap()
            } else {
                loop {
                    let w = rng.random_range(0..wires as u16);
                    if w != g.target && w != x && w != y && !scratch.contains(&w) {
                        break w;
                    }
                }
            };
            usage[pick as usize] += mult[s];
            scratch[s] = pick;
        }
        let map = |w: u16| -> u16 {
            match w {
                0 => g.target,
                1 => x,
                2 => y,
                s => scratch[s as usize - 3],
            }
        };
        for &[t, p, n] in cir {
            out.push(XGate::from_g57([map(t), map(p), map(n)]));
            lits.push(lid);
        }
        replaced += 1;
    }

    if shuffle {
        let order = commuting_shuffle_order(&mut out, &mut rng);
        lits = order.iter().map(|&i| lits[i as usize]).collect();
    }

    // functional equivalence on random states over all wires
    let mask = if wires >= 1024 { U1024::MAX } else { (U1024::one() << wires) - U1024::one() };
    let mut bad = 0usize;
    for _ in 0..verify_k {
        let mut b = [0u8; 128];
        rng.fill_bytes(&mut b);
        let xin = U1024::from_little_endian(&b) & mask;
        if eval_u1024(&gates, xin) != eval_u1024(&out, xin) {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "FUNCTIONAL VERIFY FAILED: {bad}/{verify_k} states mismatched");

    write_mpmct(&outp, &out, wires).expect("write output");
    // Litter sidecar: one id per output gate, matching gate order — each host
    // gate's block is one litter (fmix --litter-in consumes this).
    {
        use std::io::Write;
        let mut f = std::fs::File::create(format!("{outp}.litter")).expect("litter sidecar");
        writeln!(f, "litter1 {}", lits.len()).unwrap();
        for l in &lits {
            writeln!(f, "{l}").unwrap();
        }
    }
    // Balance report from the OUTPUT gates (slot appearances per wire).
    let mut fin = vec![0u64; wires];
    for g in &out {
        fin[g.target as usize] += 1;
        for &(w, _) in &g.ctrls {
            fin[w as usize] += 1;
        }
    }
    let half = wires / 2;
    let (lo, hi): (u64, u64) = (fin[..half].iter().sum(), fin[half..].iter().sum());
    let mean = fin.iter().sum::<u64>() as f64 / wires as f64;
    let var = fin.iter().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / wires as f64;
    println!(
        "[sgdb-sub] {} -> {} gates ({replaced} g57s replaced, {passed} gates passed through, shuffle={shuffle}, balance={balance}); verify {verify_k}/{verify_k} OK; wrote {outp}",
        gates.len(),
        out.len()
    );
    println!(
        "[sgdb-sub] wire balance: low-half {lo} slots, high-half {hi} slots (ratio {:.3}); per-wire CV {:.3}",
        lo as f64 / hi.max(1) as f64,
        var.sqrt() / mean
    );
}
