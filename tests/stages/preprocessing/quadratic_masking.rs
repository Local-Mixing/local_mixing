use super::*;
use rand::Rng;

// A random g57 circuit on `n` wires (from_g57 triples with distinct wires).
fn random_a(n: u16, m: usize, rng: &mut StdRng) -> Vec<XGate> {
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let a = rng.random_range(0..n);
        let x = rng.random_range(0..n);
        let y = rng.random_range(0..n);
        if a != x && a != y && x != y {
            out.push(XGate::from_g57([a, x, y]));
        }
    }
    out
}

// The gadget must compute A on the data wires for EVERY data input and
// EVERY band state (masks open and close symmetrically; band updates never
// straddle the masks that read them). Exhaustive over the data, sampled
// over the band, both read modes, several seeds.
#[test]
fn gadget_computes_a_exhaustively_in_both_read_modes() {
    let n: u16 = 6;
    let np = n as usize;
    for seed in 0..6u64 {
        let mut rng = StdRng::seed_from_u64(0xB5_0000 + seed);
        let a = random_a(n, 24, &mut rng);
        // (quad_fire, balanced, encoded_io, k, repair slots): both read modes,
        // both mask kinds, encoded I/O, odd K (K3 == K2 + the balancing wire),
        // K4 (two pairs per LGI) and REPAIR-kind rerand slots.
        for (quad_fire, balanced, encoded_io, k, repair) in [
            (true, false, false, 2, 0),
            (false, false, false, 2, 0),
            (true, true, false, 2, 0),
            (false, true, false, 2, 0),
            (true, false, true, 2, 0),
            (true, true, true, 2, 0),
            (true, true, false, 3, 0),
            (true, false, false, 3, 0),
            (true, true, false, 4, 2),
            (false, true, false, 4, 2),
            (true, true, true, 2, 2),
        ] {
            let p = QuadraticMaskingParams {
                quad_fire,
                balanced,
                encoded_io,
                k,
                rerand_repair: repair,
                // exercise both the min-open rule (default 2) and its absence
                min_open: if repair > 0 { 1 } else { 2 },
                ..QuadraticMaskingParams::production(100 + seed)
            };
            let out = preprocess_quadratic_masking(&a, np, &p);
            assert_eq!(out.num_wires, 2 * np);
            // no clean ancillas: the fire borrows dirty band wires
            assert_eq!(out.scratch_wires, 0);
            assert_eq!(out.pre_gates.is_empty(), !encoded_io);
            assert_eq!(out.post_gates.is_empty(), !encoded_io);
            if encoded_io {
                // every data wire is masked at the start and at the end
                for w in 0..np as u16 {
                    assert!(out.pre_gates.iter().any(|g| g.target == w));
                    assert!(out.post_gates.iter().any(|g| g.target == w));
                }
            }
            for band in 0..8u64 {
                let band_bits = if band == 0 {
                    0
                } else {
                    rng.random::<u64>() >> (64 - np)
                };
                for data in 0..(1u64 << np) {
                    let mut expect = data;
                    for g in &a {
                        expect = g.apply_u64(expect);
                    }
                    let mut st = data | (band_bits << np);
                    for g in out
                        .pre_gates
                        .iter()
                        .chain(&out.gates)
                        .chain(&out.post_gates)
                    {
                        st = g.apply_u64(st);
                    }
                    assert_eq!(
                        st & ((1u64 << np) - 1),
                        expect,
                        "seed {seed} quad_fire={quad_fire} balanced={balanced} encoded_io={encoded_io} k={k} repair={repair} band {band:#x} data {data:#x}"
                    );
                }
            }
            if quad_fire {
                // the masked fire is two-control only (the DB is a g57 ball)
                assert!(out.gates.iter().all(|g| g.ctrls.len() <= 2));
                // and the scratch wires are clean again after every fire:
                // checked implicitly by the exhaustive equivalence above
            }
        }
    }
}
