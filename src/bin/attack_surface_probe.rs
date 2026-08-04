//! Exhaustive attacker-surface probe for the production 6n TDP artifact.
//!
//! Builds `tdp4n_nonlinear_with_slice_zero_random_cnot` at a tiny n, enumerates
//! ALL 2^(6n) inputs of the forward map and of the gate-order-reversed map, and
//! reports exactly which output block is a function of which input block.
//!
//! Nothing here reads the artifact's gate list structurally: every claim is a
//! black-box input/output fact, which is precisely the attacker's view.

use local_mixing::circuit::CircuitSeq;
use local_mixing::postmix::xgate::{XGate, eval_u64};
use local_mixing::replace::gadgets::{packed_bit, tdp4n_nonlinear_with_slice_zero_random_cnot};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn field(state: u64, lo: usize, width: usize) -> u64 {
    (state >> lo) & ((1u64 << width) - 1)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let total = 6 * n;
    assert!(total <= 26, "exhaustive probe needs 6n <= 26");
    let states: usize = 1usize << total;
    let mut rng = StdRng::seed_from_u64(seed);

    // ---- a random all-G57 secret circuit C on n wires -------------------
    let source = CircuitSeq {
        gates: (0..8 * n)
            .map(|_| {
                let a = rng.random_range(0..n) as u16;
                let x = loop {
                    let w = rng.random_range(0..n) as u16;
                    if w != a {
                        break w;
                    }
                };
                let y = loop {
                    let w = rng.random_range(0..n) as u16;
                    if w != a && w != x {
                        break w;
                    }
                };
                [a, x, y]
            })
            .collect(),
    };
    let c_gates: Vec<XGate> = source.gates.iter().copied().map(XGate::from_g57).collect();
    let c_perm: Vec<u64> = (0..1u64 << n)
        .map(|x| eval_u64(c_gates.iter(), x))
        .collect();
    let mut c_inv = vec![0u64; 1 << n];
    for (x, &v) in c_perm.iter().enumerate() {
        c_inv[v as usize] = x as u64;
    }

    // ---- the production artifact ----------------------------------------
    let built = tdp4n_nonlinear_with_slice_zero_random_cnot(&source, n, 1, 4 * n, &mut rng);
    let gates = built.circuit.gates;
    assert_eq!(built.circuit.num_wires, total, "expected exactly 6n wires");
    let y_star: u64 = (0..n)
        .filter(|&b| packed_bit(&built.public_y, b))
        .fold(0, |a, b| a | 1 << b);
    let z_star: u64 = (0..n)
        .filter(|&b| packed_bit(&built.public_z, b))
        .fold(0, |a, b| a | 1 << b);

    println!(
        "n={n} wires={total} gates={} Y*={y_star:#x} Z*={z_star:#x}",
        gates.len()
    );

    // ---- (2) every gate is an involution --------------------------------
    let self_reading = gates
        .iter()
        .filter(|g| g.ctrls.iter().any(|&(w, _)| w == g.target))
        .count();
    println!("gates_reading_own_target={self_reading}  (0 => every gate is self-inverse)");

    // Which physical blocks are ever written / read.
    let block = |w: usize| match w / n {
        0 => "X",
        1 => "Y",
        2 => "Z",
        3 => "W",
        _ => "band",
    };
    for name in ["X", "Y", "Z", "W", "band"] {
        let t = gates
            .iter()
            .filter(|g| block(g.target as usize) == name)
            .count();
        let r = gates
            .iter()
            .filter(|g| g.ctrls.iter().any(|&(w, _)| block(w as usize) == name))
            .count();
        println!("block {name:<4} gates_targeting={t:<8} gates_reading={r}");
    }

    // ---- forward and reverse tables --------------------------------------
    let rev: Vec<XGate> = gates.iter().rev().cloned().collect();
    let mut fwd = vec![0u32; states];
    let mut bwd = vec![0u32; states];
    for s in 0..states {
        fwd[s] = eval_u64(gates.iter(), s as u64) as u32;
        bwd[s] = eval_u64(rev.iter(), s as u64) as u32;
    }

    // (2) reversal is exact inversion
    let inversion_ok = (0..states).all(|s| bwd[fwd[s] as usize] as usize == s)
        && (0..states).all(|s| fwd[bwd[s] as usize] as usize == s);
    println!("reversed_circuit_is_exact_inverse={inversion_ok}");

    // ---- (1) forward block dependence ------------------------------------
    // Report, for each output block, the minimal set of input blocks it
    // depends on, established by exhaustive search.
    let names = ["X", "Y", "Z", "W", "b0", "b1"];
    let lo = |i: usize| i * n;
    let dep = |table: &[u32], out_block: usize, xor_in: Option<usize>| -> Vec<&'static str> {
        // For each input block j, does out_block change when only block j changes?
        let mut deps = Vec::new();
        for j in 0..6 {
            let mut depends = false;
            'outer: for s in 0..states {
                let base = {
                    let o = field(table[s] as u64, lo(out_block), n);
                    match xor_in {
                        Some(k) => o ^ field(s as u64, lo(k), n),
                        None => o,
                    }
                };
                for v in 0..(1u64 << n) {
                    let s2 = (s as u64 & !(((1u64 << n) - 1) << lo(j))) | (v << lo(j));
                    let o2 = field(table[s2 as usize] as u64, lo(out_block), n);
                    let o2 = match xor_in {
                        Some(k) => o2 ^ field(s2, lo(k), n),
                        None => o2,
                    };
                    if o2 != base {
                        depends = true;
                        break 'outer;
                    }
                }
            }
            if depends {
                deps.push(names[j]);
            }
        }
        deps
    };

    println!("\n--- FORWARD  out_block <- input blocks it actually depends on ---");
    for ob in 0..6 {
        println!("  out[{}] depends on {:?}", names[ob], dep(&fwd, ob, None));
    }
    println!("  out[Y] XOR y depends on {:?}", dep(&fwd, 1, Some(1)));

    println!("\n--- REVERSE  out_block <- input blocks it actually depends on ---");
    for ob in 0..6 {
        println!("  rev[{}] depends on {:?}", names[ob], dep(&bwd, ob, None));
    }
    println!("  rev[Y] XOR v depends on {:?}", dep(&bwd, 1, Some(1)));

    // ---- (1) the payload identity on and off the fixed slice --------------
    let mask = (1u64 << n) - 1;
    let probe = |x: u64, y: u64, z: u64, w: u64, b: u64| -> u64 {
        let s = x | (y << n) | (z << (2 * n)) | (w << (3 * n)) | (b << (4 * n));
        fwd[s as usize] as u64
    };
    let mut slice_ok = true;
    for x in 0..=mask {
        for w in 0..=mask {
            for b in 0..(1u64 << (2 * n)) {
                let o = probe(x, y_star, z_star, w, b);
                if field(o, n, n) ^ y_star != c_perm[x as usize] {
                    slice_ok = false;
                }
            }
        }
    }
    println!("\npayload_identity_on_slice  out[Y]^Y* == C(x) for all x,W,band : {slice_ok}");

    // C-evaluation oracle away from the slice: out[Y]^y == C(pi_{y,z}(x))?
    let mut off_slice_is_c_of_perm = true;
    let mut off_slice_perm_nontrivial = 0usize;
    for y in 0..=mask {
        for z in 0..=mask {
            let img: Vec<u64> = (0..=mask)
                .map(|x| field(probe(x, y, z, 0, 0), n, n) ^ y)
                .collect();
            // every value of out[Y]^y must be C of something, and the induced
            // x -> C^-1(out[Y]^y) must be a bijection (i.e. a permutation pi).
            let pre: Vec<u64> = img.iter().map(|&v| c_inv[v as usize]).collect();
            let mut seen = vec![false; 1 << n];
            for &p in &pre {
                if seen[p as usize] {
                    off_slice_is_c_of_perm = false;
                }
                seen[p as usize] = true;
            }
            if pre.iter().enumerate().any(|(x, &p)| p != x as u64) {
                off_slice_perm_nontrivial += 1;
            }
            // is pi an involution?
            if pre
                .iter()
                .enumerate()
                .any(|(x, &p)| pre[p as usize] != x as u64)
            {
                println!("  !! pi_(y={y:#x},z={z:#x}) is NOT an involution");
            }
        }
    }
    println!(
        "off_slice out[Y]^y == C(pi_yz(x)) with pi_yz a permutation : {off_slice_is_c_of_perm}  \
         (non-identity for {off_slice_perm_nontrivial} of {} (y,z) pairs)",
        (mask + 1) * (mask + 1)
    );

    // ---- the two composable maps ------------------------------------------
    // A = D.C   from forward X block on the slice
    let a_perm: Vec<u64> = (0..=mask)
        .map(|x| field(probe(x, y_star, z_star, 0, 0), 0, n))
        .collect();
    let mut a_is_perm = vec![false; 1 << n];
    for &v in &a_perm {
        a_is_perm[v as usize] = true;
    }
    println!(
        "forward X block A(x)=out[X] is a permutation : {}",
        a_is_perm.iter().all(|&b| b)
    );

    // D^-1 from the REVERSE Y block, at arbitrary chosen u.
    let rprobe = |u: u64, v: u64, z: u64, w: u64, b: u64| -> u64 {
        let s = u | (v << n) | (z << (2 * n)) | (w << (3 * n)) | (b << (4 * n));
        bwd[s as usize] as u64
    };
    let d_inv: Vec<u64> = (0..=mask)
        .map(|u| field(rprobe(u, 0, 0, 0, 0), n, n))
        .collect();
    let mut d_inv_stable = true;
    for u in 0..=mask {
        for v in 0..=mask {
            for z in 0..=mask {
                for b in 0..(1u64 << (2 * n)) {
                    if field(rprobe(u, v, z, 0, b), n, n) ^ v != d_inv[u as usize] {
                        d_inv_stable = false;
                    }
                }
            }
        }
    }
    println!(
        "reverse Y block gives Dinv(u) := rev[Y]^v, independent of v,z,W,band : {d_inv_stable}"
    );

    // Dinv really is D^-1: Dinv(A(x)) == C(x)
    let d_inv_matches =
        (0..=mask).all(|x| d_inv[a_perm[x as usize] as usize] == c_perm[x as usize]);
    println!("Dinv(A(x)) == C(x) for all x  (so Dinv = D^-1 and A = D.C) : {d_inv_matches}");

    // reverse X block = A^-1 up to the preblock twist
    let a_inv_from_rev: Vec<u64> = (0..=mask)
        .map(|u| {
            // choose v so that the intermediate Y lands on Y*, and z = Z*:
            // then the preblock inverse is the identity.
            let v = y_star ^ d_inv[u as usize];
            field(rprobe(u, v, z_star, 0, 0), 0, n)
        })
        .collect();
    let a_inv_ok = (0..=mask).all(|x| a_inv_from_rev[a_perm[x as usize] as usize] == x);
    println!("reverse X block at the corrected slice == A^-1 : {a_inv_ok}");

    // ---- (4) does the band buy anything? ----------------------------------
    let mut band_bijective = true;
    let mut band_affects_carriers = false;
    for x in 0..=mask {
        for y in 0..=mask {
            let mut seen = vec![false; 1 << (2 * n)];
            let c0 = field(probe(x, y, z_star, 0, 0), 0, n);
            for b in 0..(1u64 << (2 * n)) {
                let o = probe(x, y, z_star, 0, b);
                let ob = field(o, 4 * n, 2 * n);
                if seen[ob as usize] {
                    band_bijective = false;
                }
                seen[ob as usize] = true;
                if field(o, 0, n) != c0 {
                    band_affects_carriers = true;
                }
            }
        }
    }
    println!("\nband: b -> out[band] bijective for each fixed (x,y) : {band_bijective}");
    println!("band: any input band value changes a carrier output : {band_affects_carriers}");

    // does out[band] carry information beyond a function of the carriers+b?
    // i.e. is out[band] determined by (out[X], out[Y], b)?
    let mut band_determined_by_outputs = true;
    let mut memo = std::collections::HashMap::new();
    for x in 0..=mask {
        for y in 0..=mask {
            for b in 0..(1u64 << (2 * n)) {
                let o = probe(x, y, z_star, 0, b);
                let key = (field(o, 0, n), field(o, n, n), b);
                let val = field(o, 4 * n, 2 * n);
                if *memo.entry(key).or_insert(val) != val {
                    band_determined_by_outputs = false;
                }
            }
        }
    }
    println!(
        "band: out[band] is a function of (out[X], out[Y], b) alone : {band_determined_by_outputs}"
    );
}
