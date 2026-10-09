use super::*;
use crate::stages::packing::anf::WindowState;
use crate::stages::packing::oracle_io::{parse_oracle_mpmct, read_oracle_mpmct, window_cuts};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::path::Path;

fn fx(p: &str) -> OracleCircuit {
    read_oracle_mpmct(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/stage4")
            .join(p),
    )
    .unwrap()
}

fn params(w_cap: usize, g: usize, b_mono: usize, span_cap: usize) -> PackParams {
    PackParams {
        w_cap,
        g,
        b_mono,
        span_cap,
        ..PackParams::PRODUCTION
    }
}

/// Replay the packer's window ANF over the emitted windows: (compose aborts,
/// aborts that are not their window's last gate).
fn replay_aborts(b: &[XGate], pk: &Packed, b_mono: usize) -> (usize, usize) {
    let cuts = window_cuts(
        &pk.bounds.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        pk.order.len(),
    );
    let (mut aborts, mut early) = (0, 0);
    for w in cuts.windows(2) {
        let mut ws = WindowState::<1>::new(8 * b_mono);
        let mut st = MulStats::default();
        for i in w[0]..w[1] {
            if !ws.compose(&b[pk.order[i] as usize], CapSemantics::CPython39, &mut st) {
                aborts += 1;
                if i != w[1] - 1 {
                    early += 1;
                }
                break;
            }
        }
    }
    (aborts, early)
}

#[test]
fn committed_delete_preserves_order() {
    let mut c = Committed::new(200);
    for iid in [5u32, 2, 9] {
        c.insert(iid, iid + 10);
    }
    c.remove(2);
    c.insert(7, 1);
    assert_eq!(c.entries(), vec![(5, 15), (9, 19), (7, 1)]);
    c.set(9, 3);
    assert_eq!(c.get(9), Some(3));
    assert_eq!(c.get(2), None);
    // compaction (more than half dead, > 64 slots) keeps insertion order
    for iid in 100..180u32 {
        c.insert(iid, 1);
    }
    for iid in (100..180u32).filter(|i| i % 4 != 0) {
        c.remove(iid);
    }
    let live: Vec<u32> = c.entries().iter().map(|e| e.0).collect();
    let want: Vec<u32> = [5, 9, 7]
        .into_iter()
        .chain((100..180).filter(|i| i % 4 == 0))
        .collect();
    assert_eq!(live, want);
    assert!(c.ent.len() < 64);
    c.clear();
    assert!(c.is_empty() && c.get(5).is_none());
}

#[test]
fn compose_failure_boundary_after_gate_and_duplicate_boundary() {
    // dense16 at `512 4 8 3000` (an oracle fixture): one compose abort, whose
    // window closes right after the aborting gate; the pre-emptive close then
    // finds an empty window, so the boundary repeats. The first interval also
    // closes an empty window (boundary 0, filtered out).
    let a = fx("pack/dense16_a.mpmct1");
    let b = fx("pack/dense16_b.mpmct1");
    let p = params(512, 4, 8, 3000);
    let pk = pack(&a, &b, &p, "t", &mut Vec::new()).unwrap();
    let r = &pk.report;
    assert_eq!(r.compose_aborts, 1);
    assert_eq!(r.duplicate_bounds, 1);
    assert_eq!(r.windows, r.bounds + 2);
    assert_eq!(replay_aborts(&b.gates, &pk, 8), (1, 0));
    let dup = pk.bounds.windows(2).position(|w| w[0] == w[1]).unwrap();
    // the aborting gate is the last one before the repeated boundary
    let mut ws = WindowState::<1>::new(64);
    let mut st = MulStats::default();
    let lo = if dup == 0 { 0 } else { pk.bounds[dup - 1] };
    let mut aborted_at = None;
    for i in lo..pk.bounds[dup] {
        if !ws.compose(
            &b.gates[pk.order[i] as usize],
            CapSemantics::CPython39,
            &mut st,
        ) {
            aborted_at = Some(i);
            break;
        }
    }
    assert_eq!(aborted_at, Some(pk.bounds[dup] - 1));
}

#[test]
fn deaths_repr_is_python_dict_order() {
    let r = PackReport {
        deaths: vec![("burst_close", 16), ("sibling", 332), ("drain_close", 71)],
        ..Default::default()
    };
    assert_eq!(
        r.deaths_repr(),
        "{'burst_close': 16, 'sibling': 332, 'drain_close': 71}"
    );
    assert_eq!(PackReport::default().deaths_repr(), "{}");
}

fn random_pair(rng: &mut StdRng, n: u16, m: usize) -> (OracleCircuit, OracleCircuit) {
    let mut a = Vec::new();
    for _ in 0..m {
        let t = rng.random_range(0..n);
        let k = rng.random_range(0..4usize);
        let mut cs: Vec<(u16, bool)> = Vec::new();
        while cs.len() < k {
            let c = rng.random_range(0..n);
            if c != t && !cs.iter().any(|x| x.0 == c) {
                cs.push((c, rng.random_bool(0.5)));
            }
        }
        let mut gt = XGate::conj(t, cs).unwrap();
        gt.comp = rng.random_bool(0.3);
        a.push(gt);
    }
    let mut b = a.clone();
    for _ in 0..m / 10 {
        let j = rng.random_range(0..=b.len());
        let x = a[rng.random_range(0..m)].clone();
        b.insert(j, x.clone());
        b.insert(j, x);
    }
    let commute = |x: &XGate, y: &XGate| {
        !x.ctrls.iter().any(|c| c.0 == y.target) && !y.ctrls.iter().any(|c| c.0 == x.target)
    };
    for _ in 0..20 * m {
        let i = rng.random_range(0..b.len() - 1);
        if commute(&b[i], &b[i + 1]) {
            b.swap(i, i + 1);
        }
    }
    let text = |gs: &[XGate]| {
        let mut v = Vec::new();
        crate::stages::packing::oracle_io::write_pack_mpmct(&mut v, n as usize, gs).unwrap();
        parse_oracle_mpmct(std::str::from_utf8(&v).unwrap()).unwrap()
    };
    (text(&a), text(&b))
}

#[test]
fn random_pairs_pack_certify_and_abort_only_at_window_ends() {
    // Release-mode checks (lowest-ready lemma, unreachable paths) never fire,
    // VERIFY and the reorder certificate pass, and every compose abort is a
    // window's last gate.
    let mut rng = StdRng::seed_from_u64(20_261_008);
    let mut aborts = 0;
    for case in 0..40 {
        let (a, b) = random_pair(&mut rng, 6 + (case % 5) as u16, 300);
        let p = params(
            [8, 64, 512][case % 3],
            [4, 16, 64][(case / 3) % 3],
            [2, 16, 128, 3072][case % 4],
            [20, 300][case % 2],
        );
        let pk = pack(&a, &b, &p, "r", &mut Vec::new()).unwrap();
        assert!(pk.report.verify_ok());
        let (ab, early) = replay_aborts(&b.gates, &pk, p.b_mono);
        assert_eq!(early, 0);
        assert_eq!(ab, pk.report.compose_aborts);
        aborts += ab;
        let mut sorted = pk.order.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().enumerate().all(|(i, &x)| i == x as usize));
    }
    assert!(
        aborts > 0,
        "the random cases should exercise compose aborts"
    );
}

#[test]
fn empty_b_is_an_input_error() {
    let a = parse_oracle_mpmct("mpmct1 2 1\n1 0 1 0 1\n").unwrap();
    let b = parse_oracle_mpmct("mpmct1 2 0\n").unwrap();
    assert!(matches!(
        pack(&a, &b, &PackParams::PRODUCTION, "e", &mut Vec::new()),
        Err(Stage4Error::Input(_))
    ));
}
