use super::*;
use crate::circuit::xgate::XGate;
use crate::stages::packing::anf::{MulStats, window_anf};
use crate::stages::packing::types::CapSemantics;

type Lanes = [u64; 4];

/// 256-lane simulation of local gates.
fn simulate(gates: &[SGate], st: &mut [Lanes]) {
    for g in gates {
        let mut acc = [!0u64; 4];
        for i in 0..g.k as usize {
            let v = st[g.c[i] as usize];
            for j in 0..4 {
                acc[j] &= if g.p[i] { v[j] } else { !v[j] };
            }
        }
        if g.comp {
            for a in acc.iter_mut() {
                *a = !*a;
            }
        }
        let t = &mut st[g.t as usize];
        for j in 0..4 {
            t[j] ^= acc[j];
        }
    }
}

fn random_state(n: usize, seed: u64) -> Vec<Lanes> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let mut l = [0u64; 4];
            for x in l.iter_mut() {
                *x = rng.next_u64();
            }
            l
        })
        .collect()
}

fn random_gates(n: usize, m: usize, rng: &mut Rng) -> Vec<SGate> {
    let mut gates = Vec::new();
    for _ in 0..m {
        let t = rng.below(n as u64) as u32;
        let k = 1 + rng.below(2);
        let mut a = rng.below(n as u64) as u32;
        while a == t {
            a = rng.below(n as u64) as u32;
        }
        let mut b = rng.below(n as u64) as u32;
        while b == t || b == a {
            b = rng.below(n as u64) as u32;
        }
        let comp = rng.below(2) == 1;
        gates.push(if k == 1 {
            SGate::cnot(t, a, rng.below(2) == 1, comp)
        } else {
            SGate::tof(t, a, rng.below(2) == 1, b, rng.below(2) == 1, comp)
        });
    }
    gates
}

/// The spec of local gates through stage 4's composer (local id = global wire).
fn spec_of<const W: usize>(gates: &[SGate], n: usize) -> WindowSpec<W> {
    let xg: Vec<XGate> = gates
        .iter()
        .map(|g| {
            let lits: Vec<(u16, bool)> =
                (0..g.k as usize).map(|i| (g.c[i] as u16, g.p[i])).collect();
            let mut x = XGate::conj(g.t as u16, lits).unwrap();
            x.comp = g.comp;
            x
        })
        .collect();
    let mut stats = MulStats::default();
    let anf = window_anf::<W>(&xg, 1 << 20, CapSemantics::Canonical, &mut stats).unwrap();
    WindowSpec {
        num_wires: n as u32,
        polys: anf
            .changed
            .iter()
            .map(|(w, p)| {
                (
                    *w as u32,
                    p.iter()
                        .map(|m| {
                            let vs: Vec<u32> = m.bits().map(|b| b as u32).collect();
                            Mono::<W>::from_vars(&vs)
                        })
                        .collect(),
                )
            })
            .collect(),
    }
}

fn check(syn: &[SGate], orig: &[SGate], n: usize, seed: u64) {
    for g in syn {
        assert!(g.t < n as u32 && g.k <= 2);
        for i in 0..g.k as usize {
            assert!(g.c[i] < n as u32 && g.c[i] != g.t);
        }
        if g.k == 2 {
            assert_ne!(g.c[0], g.c[1]);
        }
    }
    let st0 = random_state(n, seed);
    let mut s1 = st0.clone();
    simulate(syn, &mut s1);
    let mut s2 = st0;
    simulate(orig, &mut s2);
    assert_eq!(s1, s2, "synthesized circuit differs from the original");
}

#[test]
fn barenco_ladder_computes_the_product_and_restores_the_ancillas() {
    for k in 3..=9usize {
        let n = 2 * k + 2;
        let c: Vec<u32> = (1..=k as u32).collect();
        let h: Vec<u32> = (k as u32 + 1..=(2 * k as u32 - 2)).collect();
        let mut g = Vec::new();
        barenco(0, &c, &h, &mut g);
        assert_eq!(g.len(), 4 * (k - 2));
        let st0 = random_state(n, 99 + k as u64);
        let mut st = st0.clone();
        simulate(&g, &mut st);
        let mut prod = [!0u64; 4];
        for &ci in &c {
            for j in 0..4 {
                prod[j] &= st0[ci as usize][j];
            }
        }
        for w in 0..n {
            let mut want = st0[w];
            if w == 0 {
                for j in 0..4 {
                    want[j] ^= prod[j];
                }
            }
            assert_eq!(st[w], want, "k={k} wire {w}");
        }
    }
}

#[test]
fn gate_l_agrees_with_the_gate_family_and_the_simulator() {
    // every (k, flags) of the family: the polynomial gate_l reads back from
    // the gate equals the family polynomial, and the gate computes it
    let n = 3;
    for k in 0..3u8 {
        for flags in 0..8u8 {
            if k < 2 && flags > 1 {
                continue;
            }
            let g = family_gate(0, k, 1, 2, flags);
            let mut l = [Mono::<1>::ONE; 4];
            let nl = gate_l(&g, &mut l);
            let mut want: Vec<Mono<1>> = Vec::new();
            match k {
                0 => want.push(Mono::ONE),
                1 => {
                    want.push(Mono::var(1));
                    if flags & 1 != 0 {
                        want.push(Mono::ONE);
                    }
                }
                _ => {
                    want.push(Mono::from_vars(&[1, 2]));
                    if flags & 2 != 0 {
                        want.push(Mono::var(1));
                    }
                    if flags & 4 != 0 {
                        want.push(Mono::var(2));
                    }
                    if flags & 1 != 0 {
                        want.push(Mono::ONE);
                    }
                }
            }
            let mut got = l[..nl].to_vec();
            got.sort();
            want.sort();
            assert_eq!(got, want, "k={k} flags={flags}");
            // numeric: x0 ^= L(x1, x2) on every input
            for x in 0..8u64 {
                let st0: Vec<Lanes> = (0..n).map(|w| [(x >> w) & 1; 4]).collect();
                let mut st = st0.clone();
                simulate(&[g], &mut st);
                let mut lv = 0u64;
                for m in &got {
                    let mut term = 1u64;
                    for v in m.vars() {
                        term &= (x >> v) & 1;
                    }
                    lv ^= term;
                }
                assert_eq!(
                    st[0][0] & 1,
                    (st0[0][0] ^ lv) & 1,
                    "k={k} flags={flags} x={x}"
                );
            }
        }
    }
    // a no-op gate has an empty polynomial
    let mut l = [Mono::<1>::ONE; 4];
    let noop = SGate {
        t: 0,
        comp: true,
        k: 0,
        c: [0; 2],
        p: [false; 2],
    };
    assert!(noop.is_noop());
    assert_eq!(gate_l(&noop, &mut l), 0);
}

#[test]
fn canonical_cancels_even_multiplicities_and_drops_identities() {
    let a = Mono::<1>::var(1);
    let b = Mono::<1>::var(2);
    let spec = WindowSpec {
        num_wires: 4,
        polys: vec![
            (3, vec![b, a, b, Mono::var(3)]),
            (0, vec![Mono::var(0)]),
            (2, vec![a, a]),
        ],
    };
    let c = canonical(&spec);
    assert_eq!(c.polys, vec![(2, vec![]), (3, vec![a, Mono::var(3)])]);
}

#[test]
fn synthesizes_random_small_circuits_exactly() {
    let n = 8;
    let m = 12;
    let mut rng = Rng::new(7);
    let p = SynthParams {
        time_limit: 10.0,
        restarts: 4,
        ..Default::default()
    };
    let mut total_orig = 0;
    let mut total_syn = 0;
    for trial in 0..30 {
        let gates = random_gates(n, m, &mut rng);
        let spec = spec_of::<1>(&gates, n);
        let out = synthesize(&spec, &p, 7, trial);
        assert_eq!(
            out.status,
            SynthStatus::Converged,
            "trial {trial}: {}",
            out.note
        );
        assert_eq!(out.gates.len(), out.tags.len());
        check(&out.gates, &gates, n, 1000 + trial as u64);
        total_orig += gates.len();
        total_syn += out.gates.len();
    }
    // dense random circuits on 8 wires are not compressible (anf_synth's
    // selftest shows the same); the bound only guards against runaway output
    assert!(total_syn <= 2 * total_orig, "{total_syn} vs {total_orig}");
}

#[test]
fn empty_spec_gives_no_gates() {
    let spec = WindowSpec::<1> {
        num_wires: 5,
        polys: vec![],
    };
    let out = synthesize(&spec, &SynthParams::default(), 1, 0);
    assert_eq!(out.status, SynthStatus::Converged);
    assert!(out.gates.is_empty());
    assert_eq!(out.restarts_used, 0);
}

#[test]
fn output_is_a_function_of_seed_and_index() {
    let n = 8;
    let mut rng = Rng::new(11);
    let gates = random_gates(n, 10, &mut rng);
    let spec = spec_of::<2>(&gates, n);
    let p = SynthParams {
        restarts: 3,
        ..Default::default()
    };
    let a = synthesize(&spec, &p, 5, 3);
    let b = synthesize(&spec, &p, 5, 3);
    assert_eq!(a.gates, b.gates);
    assert_eq!(a.note, b.note);
    assert!(!a.deadline_hit);
    // the forward-only baseline still converges
    let mut fwd = p.clone();
    fwd.back_k = 0;
    let c = synthesize(&spec, &fwd, 5, 3);
    assert_eq!(c.status, SynthStatus::Converged);
    check(&c.gates, &gates, n, 77);
}

#[test]
fn wider_universe_and_degree_three_monomials_use_helpers() {
    // x0 ^= x1 x2 x3 (a degree-3 monomial) needs a dirty ancilla
    let spec = WindowSpec::<4> {
        num_wires: 200,
        polys: vec![(0, vec![Mono::var(0), Mono::from_vars(&[1, 2, 3])])],
    };
    let p = SynthParams {
        restarts: 2,
        ..Default::default()
    };
    let out = synthesize(&spec, &p, 1, 0);
    assert_eq!(out.status, SynthStatus::Converged, "{}", out.note);
    let n = 200;
    let st0 = random_state(n, 5);
    let mut st = st0.clone();
    simulate(&out.gates, &mut st);
    for w in 0..n {
        let mut want = st0[w];
        if w == 0 {
            for j in 0..4 {
                want[j] ^= st0[1][j] & st0[2][j] & st0[3][j];
            }
        }
        assert_eq!(st[w], want, "wire {w}");
    }
}

#[test]
fn params_validate_bounds() {
    assert!(SynthParams::default().validate().is_ok());
    let mut p = SynthParams {
        time_limit: 0.0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    p = SynthParams::default();
    p.restarts = 0;
    assert!(p.validate().is_err());
    p = SynthParams::default();
    p.w0 = -1;
    assert!(p.validate().is_err());
}
