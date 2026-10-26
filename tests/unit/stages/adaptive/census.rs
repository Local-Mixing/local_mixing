use super::*;
use crate::circuit::xgate::XGate;
use crate::stages::packing::cview::CutColumns;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::BTreeSet;

fn tof(t: u16, a: u16, b: u16) -> XGate {
    XGate::conj(t, [(a, true), (b, true)]).unwrap()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lm_adaptive_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn random_gates(rng: &mut StdRng, width: u16, m: usize) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let t = rng.random_range(0..width);
            let k = rng.random_range(1..=3usize);
            let mut lits: Vec<(u16, bool)> = Vec::new();
            while lits.len() < k {
                let w = rng.random_range(0..width);
                if w != t && !lits.iter().any(|&(x, _)| x == w) {
                    lits.push((w, rng.random::<bool>()));
                }
            }
            let mut g = XGate::conj(t, lits).unwrap();
            g.comp = rng.random::<bool>();
            g
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The prototype's basis test: after every gate the dynamic basis agrees with
// a fresh elimination (CutColumns) on visibility, and every target equals
// its residual plus the XOR of its coefficient wires.
// ---------------------------------------------------------------------------
#[test]
fn dynamic_matches_fresh_elimination_every_cut() {
    for seed in 0..12u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let nw = 12;
        let b = 4;
        let mut wire = vec![vec![0u64; b]; nw + 1];
        for v in wire.iter_mut().take(6) {
            for x in v.iter_mut() {
                *x = rng.random();
            }
        }
        wire[nw].fill(!0);
        let mut targets = wire[..nw].to_vec();
        for _ in 0..20 {
            targets.push((0..b).map(|_| rng.random()).collect());
        }
        for i in 0..6 {
            targets.push(
                (0..b)
                    .map(|j| wire[i][j] ^ (wire[(i + 1) % 6][j] & wire[(i + 2) % 6][j]))
                    .collect(),
            );
        }
        let mut s = DynamicBasis::new(wire, targets.clone());
        for _ in 0..100 {
            let t = rng.random_range(0..nw);
            let a = (t + 1 + rng.random_range(0..nw - 1)) % nw;
            let mut d = (a + 1) % nw;
            while d == t || d == a {
                d = (d + 1) % nw;
            }
            let mut g = match rng.random_range(0..3) {
                0 => XGate::x_gate(t as u16),
                1 => XGate::cnot(t as u16, a as u16),
                _ => XGate::conj(
                    t as u16,
                    [
                        (a as u16, rng.random::<bool>()),
                        (d as u16, rng.random::<bool>()),
                    ],
                )
                .unwrap(),
            };
            g.comp = rng.random::<bool>();
            s.gate(&g);
            let state: Vec<u64> = s.w[..nw].iter().flatten().copied().collect();
            let cols = CutColumns::build(&state, nw, b);
            for (i, row) in targets.iter().enumerate() {
                assert_eq!(s.visible[i], is_zero(&s.r[i]));
                assert_eq!(is_zero(&s.r[i]), cols.predict(row).is_some(), "seed {seed}");
                let mut rebuilt = s.r[i].clone();
                for w in 0..=nw {
                    if bit(&s.c[i], w) {
                        xor_into(&mut rebuilt, &s.w[w]);
                    }
                }
                assert_eq!(&rebuilt, row);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A tiny end-to-end census with hand-derived intervals.
//
// C (n = 6): c0 = x3 ^ x0x1 (internal), c1 = x4 ^ x2 (internal, excluded as
// public-linear), c2 = c0 ^ x1x2, c3 = c1 ^ x0, c4 = x5 ^ c2c3 (outputs).
// G (12 wires): the births on wires 3/4/5 with copies and XOR combinations
// on wire 6.
// ---------------------------------------------------------------------------
fn tiny() -> (Source, Vec<XGate>, usize) {
    let c = vec![
        tof(3, 0, 1),
        XGate::cnot(4, 2),
        tof(3, 1, 2),
        XGate::cnot(4, 0),
        tof(5, 3, 4),
    ];
    let source = Source::new(c, 6, [1].into_iter().collect());
    let g = vec![
        tof(3, 0, 1),      // G0: w3 = c0            location 12
        XGate::cnot(6, 3), // G1: w6 = c0            13
        XGate::cnot(4, 2), // G2: w4 = c1            14
        XGate::cnot(6, 4), // G3: w6 = c0 ^ c1       15
        tof(3, 1, 2),      // G4: w3 = c2            16
        XGate::cnot(4, 0), // G5: w4 = c3 = c1 ^ x0  17
        XGate::cnot(6, 4), // G6: w6 = c0 ^ x0       18
        tof(5, 3, 4),      // G7: w5 = c4            19
        XGate::cnot(6, 0), // G8: w6 = c0            20
    ];
    (source, g, 12)
}

fn iv(gate: usize, s: usize, e: usize, support: &[usize], out: bool) -> Interval {
    Interval {
        source_gate: gate,
        start_cut: s,
        end_cut: e,
        constant: false,
        support: support.to_vec(),
        is_output: out,
        validated: true,
    }
}

#[test]
fn tiny_circuit_intervals_metrics_and_round_trip() {
    let (source, g, width) = tiny();
    assert_eq!(source.internal, [0, 1].into_iter().collect());
    assert_eq!(source.outputs, [2, 3, 4].into_iter().collect());
    let c = census(&source, &g, width, 4);
    assert_eq!((c.gates, c.width, c.samples), (9, 12, 256));
    assert_eq!(
        c.affine,
        vec![
            iv(0, 1, 10, &[12], false),
            iv(1, 0, 10, &[2, 4], false),
            iv(2, 5, 10, &[16], true),
            iv(3, 0, 10, &[0, 2, 4], true),
            iv(4, 8, 10, &[19], true),
        ]
    );
    assert_eq!(
        c.direct,
        vec![
            iv(0, 2, 4, &[13], false),
            iv(0, 1, 5, &[12], false),
            iv(1, 3, 6, &[14], false),
            iv(2, 5, 10, &[16], true),
            iv(3, 6, 10, &[17], true),
            iv(4, 8, 10, &[19], true),
            iv(0, 9, 10, &[20], false),
        ]
    );
    assert!(c.audit.all_passed(), "{:?}", c.audit);
    assert_eq!(c.audit.random_inputs, AUDIT_RANDOM_INPUTS);
    assert!(c.audit.structured_inputs > 0);
    assert!(c.seconds > 0.0);

    let lm = c.metrics(&source, false);
    assert_eq!(lm.affine.all_internal_births, 2);
    assert_eq!(lm.affine.deeper_births, 1);
    assert_eq!(lm.affine.ids, [0, 1].into_iter().collect());
    assert!((lm.affine.cut_exposure_fraction - 19.0 / 20.0).abs() < 1e-12);
    assert_eq!(lm.direct.all_internal_births, 2);
    assert_eq!(lm.direct.deeper_births, 1);
    assert_eq!(lm.direct.ids, [0, 1].into_iter().collect());
    assert!((lm.direct.cut_exposure_fraction - 8.0 / 20.0).abs() < 1e-12);
    assert_eq!(c.metrics(&source, true), lm);

    let mut cov = vec![2i32; 11];
    cov[0] = 1;
    cov[10] = 0;
    assert_eq!(c.coverage(), cov);

    let set = |v: &[usize]| v.iter().copied().collect::<BTreeSet<usize>>();
    assert_eq!(
        c.interior_sets(&[0, 4, 9], &source),
        vec![set(&[0]), set(&[0])]
    );
    assert_eq!(
        c.interior_sets(&[0, 1, 9], &source),
        vec![set(&[]), set(&[0])]
    );
    assert_eq!(c.interior_sets(&[0, 9], &source), vec![set(&[0])]);

    let dir = scratch("census");
    write_dir(&c, &dir).unwrap();
    let back = read_dir(&dir).unwrap();
    assert_eq!(back.affine, c.affine);
    assert_eq!(back.direct, c.direct);
    assert_eq!(
        (back.gates, back.width, back.samples),
        (c.gates, c.width, c.samples)
    );
    assert_eq!(back.audit, c.audit);
    assert_eq!(back.seconds, c.seconds);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn validated_only_metrics_skip_failed_witnesses() {
    let (source, g, width) = tiny();
    let mut c = census(&source, &g, width, 4);
    c.affine[0].validated = false;
    let lm = c.metrics(&source, true);
    assert_eq!(lm.affine.ids, [1].into_iter().collect());
    assert_eq!(lm.affine.deeper_births, 0);
    assert!((lm.affine.cut_exposure_fraction - 10.0 / 20.0).abs() < 1e-12);
    // A wrong witness fails the audit.
    c.affine[0].support = vec![13];
    c.affine[0].start_cut = 2;
    let summary = audit(&source, &g, width, &mut c.affine, &mut c.direct);
    assert!(c.affine[0].validated); // location 13 = c0 as well
    c.affine[1].support = vec![2, 5];
    let summary2 = audit(&source, &g, width, &mut c.affine, &mut c.direct);
    assert!(!c.affine[1].validated);
    assert_eq!(summary2.failed_affine, 1);
    assert_eq!(summary2.failed_direct, 0);
    assert_eq!(summary.failed_affine, 0);
}

// ---------------------------------------------------------------------------
// Random circuits: every interval's claim holds at every cut it covers, the
// affine intervals are maximal, the direct ones span their location's
// lifetime, and every (row, cut) visible to a fresh elimination (or equal to
// a wire value) is covered.
// ---------------------------------------------------------------------------
/// A circuit that embeds C's gates in order among random gates: copies and
/// XORs of the low wires into the ancillas, random gates on the ancillas,
/// and random gates anywhere (which scramble C's values again).
fn embedded_circuit(rng: &mut StdRng, source: &Source, width: u16, m: usize) -> Vec<XGate> {
    let n = source.n as u16;
    let mut next = 0;
    let lits = |rng: &mut StdRng, t: u16, k: usize| {
        let mut v: Vec<(u16, bool)> = Vec::new();
        while v.len() < k {
            let w = rng.random_range(0..width);
            if w != t && !v.iter().any(|&(x, _)| x == w) {
                v.push((w, rng.random::<bool>()));
            }
        }
        v
    };
    (0..m)
        .map(|_| match rng.random_range(0..4) {
            0 if next < source.gates.len() => {
                next += 1;
                source.gates[next - 1].clone()
            }
            1 => {
                let t = rng.random_range(n..width);
                let mut c = rng.random_range(0..width);
                while c == t {
                    c = rng.random_range(0..width);
                }
                XGate::cnot(t, c)
            }
            2 => {
                let t = rng.random_range(n..width);
                let k = rng.random_range(1..=3usize);
                XGate::conj(t, lits(rng, t, k)).unwrap()
            }
            _ => {
                let t = rng.random_range(0..width);
                let k = rng.random_range(1..=2usize);
                let mut g = XGate::conj(t, lits(rng, t, k)).unwrap();
                g.comp = rng.random::<bool>();
                g
            }
        })
        .collect()
}

#[test]
#[allow(clippy::needless_range_loop)] // cut/batch indices drive several arrays
fn random_circuit_intervals_agree_with_per_cut_elimination() {
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(1000 + seed);
        let (n, width, m, batches) = (6usize, 16usize, 300usize, 4usize);
        let source = Source::new(random_gates(&mut rng, n as u16, 25), n, BTreeSet::new());
        let g = embedded_circuit(&mut rng, &source, width as u16, m);
        let c = census(&source, &g, width, batches);
        assert!(c.audit.all_passed(), "seed {seed}: {:?}", c.audit);
        assert!(!c.affine.is_empty() && !c.direct.is_empty(), "seed {seed}");

        let samples = HonestSamples::draw(n, batches, DISCOVERY_SEED);
        let prefixes: Vec<usize> = (1..=source.gates.len()).collect();
        let rows = born_rows(&source.gates, n, &prefixes, &samples);
        let mut state = vec![vec![0u64; batches]; width];
        for (b, w) in (0..batches).flat_map(|b| (0..n).map(move |w| (b, w))) {
            state[w][b] = samples.batch(b)[w];
        }
        let mut locs: Vec<Vec<u64>> = state.clone();
        let mut states = vec![state.clone()];
        let mut loc_of = vec![(0..width).collect::<Vec<usize>>()];
        let mut lifetime = vec![(0usize, m + 1); width + m];
        for (i, gate) in g.iter().enumerate() {
            for b in 0..batches {
                let mut lanes: Vec<u64> = (0..width).map(|w| state[w][b]).collect();
                gate.apply_lanes(&mut lanes);
                for w in 0..width {
                    state[w][b] = lanes[w];
                }
            }
            let t = gate.target as usize;
            let mut cur = loc_of.last().unwrap().clone();
            lifetime[cur[t]].1 = i + 1;
            cur[t] = width + i;
            lifetime[width + i] = (i + 1, m + 1);
            loc_of.push(cur);
            locs.push(state[t].clone());
            states.push(state.clone());
        }
        let flat = |st: &Vec<Vec<u64>>| st.iter().flatten().copied().collect::<Vec<u64>>();
        let row_of = |gid: usize| rows.rows.iter().position(|r| r.gate == gid).unwrap();

        for (fam, ivs) in [("affine", &c.affine), ("direct", &c.direct)] {
            for iv in ivs {
                assert!(iv.start_cut < iv.end_cut && iv.end_cut <= m + 1);
                assert!(iv.validated);
                assert_eq!(iv.is_output, source.outputs.contains(&iv.source_gate));
                let target = rows.vec(row_of(iv.source_gate));
                // The entry witness: live locations whose XOR is the birth.
                let mut acc = vec![0u64.wrapping_sub(iv.constant as u64); batches];
                for &l in &iv.support {
                    let (birth, death) = lifetime[l];
                    assert!(
                        birth <= iv.start_cut && iv.start_cut < death,
                        "{fam}: dead location"
                    );
                    xor_into(&mut acc, &locs[l]);
                }
                assert_eq!(acc, target, "{fam} seed {seed}");
                // Membership along the interval (the witness may change).
                for cut in iv.start_cut..iv.end_cut.min(m + 1) {
                    let cols = CutColumns::build(&flat(&states[cut]), width, batches);
                    assert!(
                        cols.predict(target).is_some(),
                        "{fam} seed {seed} cut {cut}"
                    );
                }
                if fam == "affine" {
                    if iv.start_cut > 0 {
                        let cols =
                            CutColumns::build(&flat(&states[iv.start_cut - 1]), width, batches);
                        assert!(cols.predict(target).is_none(), "not maximal on the left");
                    }
                    if iv.end_cut <= m {
                        let cols = CutColumns::build(&flat(&states[iv.end_cut]), width, batches);
                        assert!(cols.predict(target).is_none(), "not maximal on the right");
                    }
                } else {
                    assert_eq!(iv.support.len(), 1);
                    assert_eq!((iv.start_cut, iv.end_cut), lifetime[iv.support[0]]);
                }
            }
        }
        for cut in 0..=m {
            let cols = CutColumns::build(&flat(&states[cut]), width, batches);
            for ri in 0..rows.len() {
                let gid = rows.rows[ri].gate;
                let target = rows.vec(ri);
                let visible = cols.predict(target).is_some();
                let covered = c
                    .affine
                    .iter()
                    .any(|iv| iv.source_gate == gid && iv.start_cut <= cut && cut < iv.end_cut);
                assert_eq!(visible, covered, "seed {seed} cut {cut} row {ri}");
                for w in 0..width {
                    let v = &states[cut][w];
                    let equal = v == target;
                    let complement = v.iter().zip(target).all(|(a, b)| a ^ b == !0);
                    let l = loc_of[cut][w];
                    let direct = c.direct.iter().any(|iv| {
                        iv.source_gate == gid
                            && iv.support == [l]
                            && iv.start_cut <= cut
                            && cut < iv.end_cut
                            && iv.constant == complement
                    });
                    assert_eq!(
                        equal || complement,
                        direct,
                        "seed {seed} cut {cut} wire {w}"
                    );
                }
            }
        }
    }
}

/// `cargo test --release --lib adaptive::census -- --ignored --nocapture`
///
/// C's 2000 gates are embedded in the first ~8k gates of G (with copies
/// and XORs into the ancillas), so a representative share of the births
/// is exposed and the per-target work of the basis is exercised; a purely
/// random G exposes almost nothing and measures only the linear updates.
#[test]
#[ignore = "timing: a 100k-gate census on 32 wires"]
fn census_timing_100k_gates_32_wires() {
    let mut rng = StdRng::seed_from_u64(2026);
    let (n, width, m) = (16usize, 32usize, 100_000usize);
    let source = Source::new(random_gates(&mut rng, n as u16, 2000), n, BTreeSet::new());
    let g = embedded_circuit(&mut rng, &source, width as u16, m);
    let t0 = std::time::Instant::now();
    let c = discover(&source, &g, width, DEFAULT_BATCHES);
    let discovery = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let mut c2 = c.clone();
    let summary = audit(&source, &g, width, &mut c2.affine, &mut c2.direct);
    let auditing = t1.elapsed().as_secs_f64();
    let lm = c2.metrics(&source, true);
    eprintln!(
        "census 100k gates / 32 wires / |C| = 2000: discovery {discovery:.2} s ({} affine, {} direct intervals; {} / {} internal births affinely exposed), audit {auditing:.2} s ({} structured inputs, failed {}/{})",
        c.affine.len(),
        c.direct.len(),
        lm.affine.all_internal_births,
        source.internal.len(),
        summary.structured_inputs,
        summary.failed_affine,
        summary.failed_direct
    );
    assert!(summary.all_passed());
    // The random gates on the low wires scramble C's state early, so only
    // a few births stay exposed; the basis work per nonlinear gate (every
    // unexposed target's residual) is paid either way.
    assert!(lm.affine.all_internal_births >= 1);
}
