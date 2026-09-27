//! Cross-cut affine invariant extraction — the attack that broke the n=32 mix.
//!
//! Collect the full wire vector at K cut positions through the circuit, for S
//! random zero-slice inputs, and find EVERY affine relation
//!
//!     XOR_{(c,w) in S} wire_w(cut c)  =  const     (holds for all x)
//!
//! spanning cuts. This catches structure a single-cut linearity census cannot:
//! the documented single-carrier gadget relation `carrier_before XOR
//! carrier_after = ledger const` is exactly of this form (both endpoints may be
//! wildly nonlinear in x while their XOR is constant).
//!
//! Method: each coordinate (cut,wire) gets a "signature" = its value across all
//! S samples. An affine relation among coordinates is precisely a GF(2) linear
//! dependency among the signature vectors (together with the all-ones vector,
//! which supplies the constant term). So
//!
//!     #relations = (D + 1) - rank{ sig_0, ..., sig_{D-1}, all-ones }
//!
//! Requires S >> D for the count to be meaningful (a dependency among D vectors
//! in GF(2)^S is spurious with prob ~2^-(S-D)).
//!
//!   invariant_scan <circuit.mpmct1> [--cuts K] [--samples S] [--xw0 W] [--blk B]
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::XGate;
use std::time::Instant;

fn run_with_cuts(
    gates: &[XGate],
    num_wires: usize,
    xs: &[u128],
    xw0: usize,
    blk: usize,
    cuts: &[usize],
) -> Vec<Vec<u64>> {
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
    let path = args.next().expect("usage: invariant_scan <circuit.mpmct1> [--cuts K] [--samples S]");
    let mut ncuts = 8usize;
    let mut samples = 8192usize;
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
    assert!(samples % 64 == 0, "--samples must be a multiple of 64");

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    let ng = gates.len();
    // cuts spread through the depth INCLUDING the input (0) and output (ng)
    let cuts: Vec<usize> =
        (0..=ncuts).map(|i| (ng as f64 * i as f64 / ncuts as f64).round() as usize).collect();
    let ncut = cuts.len();
    let dim = ncut * num_wires; // coordinates
    eprintln!(
        "loaded {} gates, {} wires ({:.1}s); {} cuts x {} wires = {} coordinates, {} samples",
        ng,
        num_wires,
        t0.elapsed().as_secs_f64(),
        ncut,
        num_wires,
        dim,
        samples
    );
    assert!(samples > dim, "need samples > coordinates ({dim}) for a meaningful rank");

    // signature[coord] = bit vector over samples
    let words = samples / 64;
    let mut sig = vec![0u64; (dim + 1) * words]; // last row = all-ones (constant term)
    for w in 0..words {
        sig[dim * words + w] = !0u64;
    }

    let mut rng: u128 = 0xD1B54A32D192ED03;
    let mut next_rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let blk_mask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };

    for wi in 0..words {
        let xs: Vec<u128> = (0..64).map(|_| next_rand() & blk_mask).collect();
        let snaps = run_with_cuts(&gates, num_wires, &xs, xw0, blk, &cuts);
        for (c_i, snap) in snaps.iter().enumerate() {
            let base = c_i * num_wires;
            for w in 0..num_wires {
                sig[(base + w) * words + wi] = snap[w];
            }
        }
        if wi % 16 == 0 {
            eprintln!("  sampled {}/{} at {:.1}s", (wi + 1) * 64, samples, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("collected signatures at {:.1}s; eliminating...", t0.elapsed().as_secs_f64());

    // GF(2) elimination over sample-space to get the rank of the coordinate set.
    // piv_row[b] = index into `basis` of the row whose leading one is bit b.
    let mut piv_row: Vec<i32> = vec![-1; samples];
    let mut basis: Vec<Vec<u64>> = Vec::new();
    let mut rank = 0usize;

    let leading = |v: &[u64]| -> Option<usize> {
        for (i, &wv) in v.iter().enumerate() {
            if wv != 0 {
                return Some(i * 64 + wv.trailing_zeros() as usize);
            }
        }
        None
    };

    for coord in 0..=dim {
        let mut v: Vec<u64> = sig[coord * words..(coord + 1) * words].to_vec();
        loop {
            match leading(&v) {
                None => break, // dependent -> contributes a relation
                Some(p) => {
                    if piv_row[p] >= 0 {
                        let b = &basis[piv_row[p] as usize];
                        for k in 0..words {
                            v[k] ^= b[k];
                        }
                    } else {
                        piv_row[p] = basis.len() as i32;
                        basis.push(v);
                        rank += 1;
                        break;
                    }
                }
            }
        }
    }

    let relations = (dim + 1) - rank;
    eprintln!("eliminated at {:.1}s\n", t0.elapsed().as_secs_f64());

    println!("cross-cut affine invariant scan");
    println!("  circuit      : {}", path);
    println!("  gates        : {}", ng);
    println!("  cuts         : {} (gate indices {:?})", ncut, cuts);
    println!("  coordinates D: {} (= cuts x wires), +1 constant", dim);
    println!("  samples S    : {}", samples);
    println!("  rank         : {}", rank);
    println!("  AFFINE RELATIONS FOUND: {}", relations);
    if relations == 0 {
        println!(
            "\n  => NO affine relation holds among ANY of the {} wire-values across the whole\n     circuit depth. The documented degree-1 carrier structure does NOT survive.\n     A SAT-invariant attack has nothing to inject.",
            dim
        );
    } else {
        println!(
            "\n  => {} exploitable affine relation(s). Expected trivial baseline: {} zeroed input\n     wires at cut 0 (they are constant on the coset by construction).",
            relations,
            num_wires - blk
        );
        let nontrivial = relations as i64 - (num_wires - blk) as i64;
        println!("     Non-trivial (beyond the input-cut constants): {}", nontrivial);
    }
}
