use super::*;
use crate::stages::unpacking::mono::Rng;

/// A random gate over `width` wires: one or two controls of mixed
/// polarity, complemented half of the time.
fn random_gate(width: u16, rng: &mut Rng) -> XGate {
    let t = rng.below(width as u64) as u16;
    let mut a = rng.below(width as u64) as u16;
    while a == t {
        a = rng.below(width as u64) as u16;
    }
    let mut b = rng.below(width as u64) as u16;
    while b == t || b == a {
        b = rng.below(width as u64) as u16;
    }
    let mut lits = vec![(a, rng.below(2) == 1)];
    if rng.below(2) == 1 {
        lits.push((b, rng.below(2) == 1));
    }
    let mut x = XGate::conj(t, lits).unwrap();
    x.comp = rng.below(2) == 1;
    x
}

/// A random block of `m` gates over `width` wires. Dense random circuits
/// compose to ANFs of thousands of monomials (hopeless for a debug-mode
/// unit test), so the block is `real` random gates interleaved with
/// self-cancelling pairs `g g` (what a stage-5 block looks like: a short
/// function wrapped in a long gate list), rejection-sampled on the
/// monomial count of its ANF.
fn random_block(width: u16, m: usize, real: usize, rng: &mut Rng) -> Vec<XGate> {
    loop {
        let mut gates: Vec<XGate> = Vec::with_capacity(m);
        let mut left_real = real.min(m);
        while gates.len() < m {
            if left_real > 0 && (gates.len() + 2 > m || rng.below(3) == 0) {
                gates.push(random_gate(width, rng));
                left_real -= 1;
            } else if gates.len() + 2 <= m {
                let g = random_gate(width, rng);
                gates.push(g.clone());
                gates.push(g);
            } else {
                gates.push(random_gate(width, rng));
            }
        }
        assert_eq!(gates.len(), m);
        if let Ok((_, changed, monomials, _)) = compose_stats(&gates, width as usize, 1 << 16, 10.0)
            && changed > 0
            && monomials <= 96
        {
            return gates;
        }
    }
}

fn quick(restarts: usize) -> BlockParams {
    let mut p = BlockParams::default();
    p.synth.restarts = restarts;
    p.verify_batches = 1;
    p
}

#[test]
fn random_blocks_synthesize_and_verify() {
    let mut rng = Rng::new(7);
    let cases: [(u16, usize, usize); 6] = [
        (8, 20, 8),
        (10, 40, 10),
        (12, 60, 10),
        (14, 100, 12),
        (16, 150, 12),
        (16, 200, 14),
    ];
    for (i, &(w, m, real)) in cases.iter().enumerate() {
        let block = random_block(w, m, real, &mut rng);
        let r = synthesize_block(&block, w as usize, &quick(3), 60.0, 11, i);
        assert_eq!(
            r.status,
            BlockStatus::Ok,
            "case {i} ({w} wires, {m} gates): {} {}",
            r.synth_status,
            r.note
        );
        assert_eq!(r.synth_status, "converged");
        let g = r.gates.as_ref().unwrap();
        assert!(
            same_function(&block, g, w as usize, 2, 4242),
            "case {i}: the synthesized block differs from the original"
        );
        assert!(g.iter().all(|x| (x.max_wire() as usize) < w as usize));
        assert!(r.changed > 0 && r.monomials > 0 && r.support > 0);
        assert!(r.universe >= r.support && r.universe >= r.changed);
        assert!(r.universe <= w as usize);
        assert!(r.pre_postopt >= g.len());
        assert!(r.secs >= 0.0);
        assert!(
            g.len() < m,
            "case {i}: {} synthesized gates for {m} (mostly cancelling) gates",
            g.len()
        );
    }
}

#[test]
fn identity_block_synthesizes_to_nothing() {
    let g = XGate::conj(3, [(1u16, true), (2u16, false)]).unwrap();
    let block = vec![g.clone(), g];
    let r = synthesize_block(&block, 8, &quick(1), 10.0, 1, 0);
    assert_eq!(r.status, BlockStatus::Ok);
    assert_eq!(r.changed, 0);
    assert_eq!(r.monomials, 0);
    assert_eq!(r.gates.as_ref().map(|g| g.len()), Some(0));
    assert_eq!(compose_stats(&block, 8, 100, 1.0), Ok((0, 0, 0, 0)));
}

#[test]
fn compose_stats_counts_the_block_anf() {
    // x3 ^= x1 * !x2 ; x3 ^= 1 : P_3 = x3 + x1 + x1 x2 + 1
    let block = vec![
        XGate::conj(3, [(1u16, true), (2u16, false)]).unwrap(),
        XGate::x_gate(3),
    ];
    assert_eq!(compose_stats(&block, 8, 100, 1.0), Ok((3, 1, 4, 2)));
    assert_eq!(compose_stats(&block, 8, 2, 1.0), Err("compose_overflow"));
    assert_eq!(
        compose_stats(&[XGate::cnot(9, 0)], 8, 100, 1.0),
        Err("wire-out-of-range")
    );
}

#[test]
fn compose_overflow_fails_the_block() {
    let mut rng = Rng::new(3);
    let block: Vec<XGate> = (0..200).map(|_| random_gate(16, &mut rng)).collect();
    let mut p = quick(1);
    p.compose_cap = 8;
    let r = synthesize_block(&block, 16, &p, 10.0, 1, 0);
    assert_eq!(r.status, BlockStatus::Failed("compose_overflow"));
    assert_eq!(r.synth_status, "compose_overflow");
    assert!(r.gates.is_none());
    assert_eq!(r.universe, 0);
}

#[test]
fn deterministic_for_equal_inputs_and_seed() {
    let mut rng = Rng::new(99);
    let block = random_block(12, 80, 10, &mut rng);
    let a = synthesize_block(&block, 12, &quick(2), 60.0, 5, 3);
    let b = synthesize_block(&block, 12, &quick(2), 60.0, 5, 3);
    assert_eq!(a.status, BlockStatus::Ok, "{} {}", a.synth_status, a.note);
    assert!(
        !a.deadline_hit,
        "the test budget must not truncate the search"
    );
    assert_eq!(a.gates, b.gates);
    assert_eq!(a.steps, b.steps);
    assert_eq!(a.restarts_used, b.restarts_used);
    assert_eq!(
        (a.support, a.changed, a.monomials, a.max_degree),
        (b.support, b.changed, b.monomials, b.max_degree)
    );
}

#[test]
fn wire_outside_the_state_fails() {
    let block = vec![XGate::cnot(9, 2)];
    let r = synthesize_block(&block, 8, &quick(1), 10.0, 1, 0);
    assert_eq!(r.status, BlockStatus::Failed("wire-out-of-range"));
    assert_eq!(BlockStatus::Failed("x").name(), "x");
    assert_eq!(BlockStatus::Ok.name(), "ok");
    assert!(BlockStatus::Ok.is_ok());
}

#[test]
fn local_universe_is_support_changed_and_lowest_pool() {
    let u = local_universe(&[5, 9], &[9, 2], 16, 3);
    assert_eq!(u.wires, vec![0, 1, 2, 3, 5, 9]);
    let u = local_universe(&[0, 1, 2], &[2], 4, 8);
    assert_eq!(u.wires, vec![0, 1, 2, 3]);
    let u = local_universe(&[], &[], 6, 0);
    assert!(u.wires.is_empty());
}

#[test]
fn same_function_detects_a_difference() {
    let a = vec![XGate::cnot(1, 0), XGate::x_gate(2)];
    let mut b = a.clone();
    assert!(same_function(&a, &b, 4, 1, 1));
    b.pop();
    assert!(!same_function(&a, &b, 4, 1, 1));
    assert!(!same_function(&a, &[XGate::cnot(5, 0)], 4, 1, 1));
}

#[test]
fn params_validate() {
    assert!(BlockParams::default().validate().is_ok());
    let p = BlockParams {
        compose_cap: 0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = BlockParams {
        verify_batches: 0,
        ..Default::default()
    };
    assert!(p.validate().is_err());
    let p = BlockParams {
        ancilla_pool: 2000,
        ..Default::default()
    };
    assert!(p.validate().is_err());
}
