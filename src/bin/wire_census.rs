//! White-box structural census of a mixed circuit on the zero-slice coset.
//!
//! For each of K cut positions through the circuit depth, classify EVERY wire
//! by how it depends on the 128-bit input X (all other input wires pinned 0):
//!
//!   CONST    - always the same value across the coset (a "boundary constant";
//!              a SAT solver can be handed this as a unit clause)
//!   AFFINE   - equals c XOR <a, X> over GF(2), a != 0  (linear leak)
//!   NONLIN   - neither
//!
//! This is the attacker-side measurement that decides whether the SAT-invariant
//! attack (extract exact GF(2) relations from forward runs, inject as clauses)
//! has anything to work with. Uses the circuit ONLY as a forward oracle.
//!
//!   wire_census <circuit.mpmct1> [--cuts K] [--samples N] [--xw0 W] [--blk B]
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::XGate;
use std::time::Instant;

// Run the gate list in 64-lane bit-sliced form, snapshotting the full wire
// state at each cut position (cuts must be sorted ascending, in gate index).
fn run_with_cuts(
    gates: &[XGate],
    num_wires: usize,
    xs: &[u128],
    xw0: usize,
    blk: usize,
    cuts: &[usize],
) -> Vec<Vec<u64>> {
    assert!(xs.len() <= 64);
    let mut state = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                state[xw0 + i] |= 1u64 << lane;
            }
        }
    }
    let mut snaps = Vec::with_capacity(cuts.len());
    let mut ci = 0usize;
    for (gi, g) in gates.iter().enumerate() {
        while ci < cuts.len() && cuts[ci] == gi {
            snaps.push(state.clone());
            ci += 1;
        }
        g.apply_lanes(&mut state);
    }
    while ci < cuts.len() {
        snaps.push(state.clone());
        ci += 1;
    }
    snaps
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: wire_census <circuit.mpmct1> [--cuts K] [--samples N]");
    let mut ncuts = 12usize;
    let mut samples = 512usize;
    let mut xw0 = 0usize;
    let mut blk = 128usize;
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cuts" => ncuts = it.next().unwrap().parse().unwrap(),
            "--samples" => samples = it.next().unwrap().parse().unwrap(),
            "--xw0" => xw0 = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    let ng = gates.len();
    eprintln!("loaded {} gates, {} wires in {:.1}s", ng, num_wires, t0.elapsed().as_secs_f64());

    // cut positions: evenly spaced, including 0 (input) and ng (output)
    let cuts: Vec<usize> =
        (0..=ncuts).map(|i| (ng as f64 * i as f64 / ncuts as f64).round() as usize).collect();

    // ---- Pass 1: affine model per (cut, wire) from 129 basis inputs ----
    // c[cut][wire] : value on the all-zero input
    // col[cut][wire] : u128, bit i = d(wire)/d(X_i)
    let ncut = cuts.len();
    let mut cbit = vec![vec![false; num_wires]; ncut];
    let mut col = vec![vec![0u128; num_wires]; ncut];
    let total = 1 + blk;
    let mut g = 0usize;
    while g < total {
        let take = (total - g).min(64);
        let xs: Vec<u128> = (0..take)
            .map(|l| {
                let idx = g + l;
                if idx == 0 { 0u128 } else { 1u128 << (idx - 1) }
            })
            .collect();
        let snaps = run_with_cuts(&gates, num_wires, &xs, xw0, blk, &cuts);
        for (c_i, snap) in snaps.iter().enumerate() {
            for w in 0..num_wires {
                let lanes = snap[w];
                for l in 0..take {
                    let idx = g + l;
                    let bit = (lanes >> l) & 1 == 1;
                    if idx == 0 {
                        cbit[c_i][w] = bit;
                    } else {
                        // col bit i set iff value(e_i) != value(0)
                        if bit != cbit[c_i][w] {
                            col[c_i][w] |= 1u128 << (idx - 1);
                        }
                    }
                }
            }
        }
        g += take;
    }
    eprintln!("built per-wire affine models at {:.1}s", t0.elapsed().as_secs_f64());

    // ---- Pass 2: verify affinity on random coset inputs ----
    let mut rng: u128 = 0x9E3779B97F4A7C15;
    let mut next_rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let blk_mask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };
    // violated[cut][wire] = the affine model failed at least once
    let mut violated = vec![vec![false; num_wires]; ncut];
    let mut b = 0usize;
    while b < samples {
        let take = (samples - b).min(64);
        let xs: Vec<u128> = (0..take).map(|_| next_rand() & blk_mask).collect();
        let snaps = run_with_cuts(&gates, num_wires, &xs, xw0, blk, &cuts);
        for (c_i, snap) in snaps.iter().enumerate() {
            for w in 0..num_wires {
                let lanes = snap[w];
                // predicted lane bits from the affine model
                let mut pred: u64 = 0;
                for l in 0..take {
                    let x = xs[l];
                    let par = (x & col[c_i][w]).count_ones() & 1 == 1;
                    let v = cbit[c_i][w] ^ par;
                    if v {
                        pred |= 1u64 << l;
                    }
                }
                let mask = if take == 64 { !0u64 } else { (1u64 << take) - 1 };
                if (lanes ^ pred) & mask != 0 {
                    violated[c_i][w] = true;
                }
            }
        }
        b += take;
    }
    eprintln!("verified on {} random coset inputs at {:.1}s\n", samples, t0.elapsed().as_secs_f64());

    // ---- Report ----
    println!("wire census on the zero-slice coset  ({} gates, {} wires, X on {}..{})", ng, num_wires, xw0, xw0 + blk);
    println!("  CONST  = same value for every X (unit clause for a SAT solver)");
    println!("  AFFINE = c XOR <a,X>, a != 0        (linear leak, injectable as XOR clauses)");
    println!("  NONLIN = neither\n");
    println!("{:>10}  {:>7}  {:>7}  {:>7}   {}", "gate idx", "CONST", "AFFINE", "NONLIN", "affine+const wires (first 24)");
    for c_i in 0..ncut {
        let mut nconst = 0usize;
        let mut naff = 0usize;
        let mut nnl = 0usize;
        let mut lin_wires: Vec<usize> = Vec::new();
        for w in 0..num_wires {
            if violated[c_i][w] {
                nnl += 1;
            } else if col[c_i][w] == 0 {
                nconst += 1;
                lin_wires.push(w);
            } else {
                naff += 1;
                lin_wires.push(w);
            }
        }
        let shown: Vec<String> = lin_wires.iter().take(24).map(|w| w.to_string()).collect();
        let more = if lin_wires.len() > 24 { format!(" +{}", lin_wires.len() - 24) } else { String::new() };
        println!(
            "{:>10}  {:>7}  {:>7}  {:>7}   {}{}",
            cuts[c_i], nconst, naff, nnl,
            shown.join(","), more
        );
    }

    // ---- Output-cut detail: what the attacker sees at the boundary ----
    let last = ncut - 1;
    let mut const0 = Vec::new();
    let mut const1 = Vec::new();
    let mut affine = Vec::new();
    for w in 0..num_wires {
        if !violated[last][w] {
            if col[last][w] == 0 {
                if cbit[last][w] { const1.push(w) } else { const0.push(w) }
            } else {
                affine.push(w);
            }
        }
    }
    println!("\nOUTPUT boundary ({} wires):", num_wires);
    println!("  constant-0 wires : {}  {:?}", const0.len(), &const0[..const0.len().min(32)]);
    println!("  constant-1 wires : {}  {:?}", const1.len(), &const1[..const1.len().min(32)]);
    println!("  affine wires     : {}  {:?}", affine.len(), &affine[..affine.len().min(32)]);
    println!(
        "  => {} of {} output wires are exactly pinned/linear for a SAT solver",
        const0.len() + const1.len() + affine.len(),
        num_wires
    );
}
