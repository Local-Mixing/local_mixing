// Per-gate conjugation identities behind the twist move: N g N == neg(g)
// and S g S == swap(g), exhaustively verified on random gates (comp gates
// included — fossils relabel like anything else and stay fossils).
#[test]
fn twist_conjugation_units() {
    let mut rng = StdRng::seed_from_u64(31);
    let cnot = |t: u16, c: u16| XGate::conj(t, [(c, true)]).unwrap();
    for _ in 0..5_000 {
        let g = rand_gate(&mut rng, 8, 4, true);
        let w = rng.random_range(0..8u16);
        match conj_by_not(&g, w) {
            Some(g2) => {
                let nw = XGate::x_gate(w);
                assert!(
                    rules::verify_rewrite(&[nw.clone(), g.clone(), nw], std::slice::from_ref(&g2)),
                    "neg conjugation wrong: {g:?} on wire {w}"
                );
                assert_eq!(g2.width(), g.width());
                assert_eq!(g2.comp, g.comp);
            }
            None => assert!(!g.reads(w), "invariant gate must not read the negated wire"),
        }
        let a = rng.random_range(0..8u16);
        let b = rng.random_range(0..8u16);
        if a == b {
            continue;
        }
        if let Some(g2) = conj_by_swap(&g, a, b) {
            let packet = [cnot(b, a), cnot(a, b), cnot(b, a)];
            let mut seq: Vec<XGate> = packet.to_vec();
            seq.push(g.clone());
            seq.extend(packet.to_vec());
            assert!(
                rules::verify_rewrite(&seq, std::slice::from_ref(&g2)),
                "swap conjugation wrong: {g:?} on wires {a},{b}"
            );
            assert_eq!(g2.width(), g.width());
            assert_eq!(g2.comp, g.comp);
            // Involution: conjugating back restores the gate exactly.
            assert_eq!(conj_by_swap(&g2, a, b).unwrap(), g);
        } else {
            assert!(g.target != a && g.target != b && !g.reads(a) && !g.reads(b));
        }
        // Transvection T = cnot(b -> a); gates writing b are excluded by
        // the move's b-selection, so they are out of scope here too.
        if g.target != b {
            let t = cnot(a, b);
            let sandwich = |pieces: &[XGate]| {
                rules::verify_rewrite(&[t.clone(), g.clone(), t.clone()], pieces)
            };
            match conj_by_cnot(&g, a, b) {
                CnotConj::Invariant => {
                    assert!(
                        sandwich(std::slice::from_ref(&g)),
                        "cnot-invariant gate is not invariant: {g:?} a={a} b={b}"
                    );
                }
                CnotConj::Flip(g2) => {
                    assert!(
                        sandwich(std::slice::from_ref(&g2)),
                        "cnot flip wrong: {g:?} a={a} b={b}"
                    );
                    assert_eq!(g2.width(), g.width());
                    assert_eq!(g2.comp, g.comp);
                }
                CnotConj::Split(x, y) => {
                    assert!(!g.comp, "comp gates must be Blocked, not Split");
                    assert!(
                        sandwich(&[x.clone(), y.clone()]),
                        "cnot split wrong: {g:?} a={a} b={b}"
                    );
                    // Disjoint b-slices: the pair commutes.
                    assert!(
                        sandwich(&[y.clone(), x.clone()]),
                        "cnot split pair does not commute"
                    );
                    assert_eq!(x.width(), g.width() + 1);
                    assert_eq!(y.width(), g.width() + 1);
                }
                CnotConj::Blocked => {
                    assert!(
                        g.comp && g.reads(a) && !g.reads(b),
                        "spurious Blocked: {g:?}"
                    );
                }
            }
        }
    }
}

// The chain with twist moves enabled at high weight keeps the function
// (run() global-checks internally), erodes rather than grows fossils, and
// actually relabels interior gates. Also exercises the journal-stamp
// interaction: undo entries over relabeled pieces must die, not fire.
// Target above input: twists add gates without catalogue-invertible bulk,
// so at target the thermostat pegs near-full contraction and would starve
// the expansion channel this test is exercising.
#[test]
fn mixer_twists_preserve_function() {
    let gates = random_mixed_circuit(17, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
        w_twist_neg: 0.10,
        w_twist_swap: 0.10,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    // Rate calibration: twist packets are hard for the catalogue to dig
    // out (brackets are wall-blocked by their own window), so at these
    // weights the thermostat pegs near-full contraction and expansions
    // run at ~2% of moves — expect twist counts near 50, not hundreds.
    let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
    assert!(twists > 30, "swap-family twists barely ran: {twists}");
    assert!(
        mx.counters.twist_relabels > 0,
        "twists never relabeled a gate"
    );
    assert!(mx.remaining_g57() <= comp0, "fossil count increased");
    assert!(mx.counters.merges() > 0, "no merges alongside twists");
    mx.global_check();
}

// The g57 census partitions the width-2 comp population, and the two halves
// measure different things: `shaped` is structure (what the store can
// spell) and `same_pol` measures polarity changes. A negation twist flips
// one control's polarity and moves a gate from opp_pol to same_pol without
// changing its comp, width or count. A swap twist carries polarity with
// the wire and leaves this partition unchanged.
#[test]
fn g57_census_splits_structure_from_twist_polarity() {
    // The swap family flips control polarity only through its negation coins
    // (3/4 of twists carry a negation), so same_pol is a twist odometer:
    // twists ON drive it up, OFF leaves it at zero, while shaped (structure)
    // partitions correctly in both. p_db = 0 here, so nothing injects fresh
    // opposite-polarity material -- the only mover is the twist.
    let base = |p_twist: f64| MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let run = |p_twist| {
        let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, base(p_twist));
        mx.run();
        let cen = mx.g57_census();
        // The partition identity, and agreement with the aggregate accessor.
        assert_eq!(
            cen.shaped,
            cen.same_pol + cen.opp_pol,
            "census does not partition"
        );
        assert_eq!(
            cen.opp_pol,
            mx.true_g57(),
            "true_g57 diverged from the census"
        );
        assert!(
            cen.shaped <= mx.remaining_g57(),
            "shaped exceeds the comp population"
        );
        (cen, mx.counters.twist_relabels)
    };

    let (off, off_rel) = run(0.0);
    let (on, on_rel) = run(0.2);
    assert_eq!(off_rel, 0, "twists-off relabeled a gate: {off_rel}");
    assert!(on_rel > 0, "twists-on never relabeled a gate: {on_rel}");

    // Twists off: no polarity flips and no fresh material, so every shaped
    // gate stays a true g57. This is the load-bearing half -- it proves
    // same_pol tracks the twist, not mixing in general.
    assert_eq!(off.same_pol, 0, "twists-off flipped polarity: {off:?}");
    assert_eq!(off.pol_flipped(), 0.0);

    // Twists on: the negation coins flip it, on the same circuit. No upper
    // bound -- with p_db = 0 the small width-2 population can saturate at
    // 1.0; the sub-1/2 equilibrium seen in production comes from DB splices
    // this test deliberately does not have.
    assert!(
        on.shaped > 0 && off.shaped > 0,
        "no width-2 population: {on:?} {off:?}"
    );
    assert!(
        on.same_pol > 0,
        "swap-family twists flipped no polarity: {on:?}"
    );
    assert!(on.pol_flipped() > off.pol_flipped(), "{on:?} vs {off:?}");
}

#[test]
fn mixer_g57_twists_preserve_function_and_absorb() {
    let gates = random_mixed_circuit(29, 16, 300);
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        twist_g57: true,
        local_verify: true,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    assert!(
        mx.counters.twist_swaps > 100,
        "g57 twists barely ran: {}",
        mx.counters.twist_swaps
    );
    assert!(mx.counters.tg_emitted > 0, "brackets emitted nothing");
    assert!(
        mx.counters.tg_consumed > 0,
        "no seam ever absorbed a neighbor -- the adaptive placement is dead"
    );
    // The solver minimizes NET (word len minus context consumed), so a
    // seam may emit up to 7 gates while consuming 3; what can never
    // happen is a twist NET above the 12-gate bare spelling, since k=0
    // always offers the 6-word per seam.
    let net = (mx.counters.tg_emitted as i64 - mx.counters.tg_consumed as i64) as f64
        / mx.counters.twist_swaps as f64;
    assert!(
        net <= 12.0 + 1e-9,
        "twist net cost exceeded the bare-word bound: {net}"
    );
    mx.global_check();
}

// Symmetric truncation: with twist_min_len at circuit scale every draw is
// near-full-length, so ~half the windows left-truncate (virtual start < 0)
// and their opening packets land at the head. Function preservation +
// global_check through thousands of such windows exercises the boundary
// insert path (brackets before the arena head) and the short-window skip
// paths (len as small as 1).
#[test]
fn twist_left_truncated_windows_preserve_function() {
    let gates = random_mixed_circuit(23, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.2, // slot 1 owns twists now; the w_* are type ratios
        w_twist_neg: 0.10,
        w_twist_swap: 0.10,
        twist_min_len: usize::MAX, // clamped to circuit size -> len == n
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 11,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let twists = mx.counters.twist_swaps + mx.counters.twist_negs + mx.counters.twist_cnots;
    assert!(twists > 50, "twists barely ran: {twists}");
    mx.global_check();
}

// The negate-both variant is a genuine involution (T^2 = id) like the pure
// swap, whereas negate-one is not (T^2 = negate-both); this test drives the
// family hard so all three variants -- and the non-involutive closing
// bracket P^-1 -- get exercised, keeps the function through thousands of
// twists, and never grows fossils. All three variant counters must fire.
#[test]
fn mixer_swap_family_twists_preserve_function() {
    let gates = random_mixed_circuit(19, 16, 300);
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    // The (alpha, beta) coins are 1/4 : 1/2 : 1/4, so at this rate all three
    // variants fire many times.
    assert!(
        mx.counters.twist_swaps > 5,
        "pure swaps barely ran: {}",
        mx.counters.twist_swaps
    );
    assert!(
        mx.counters.twist_negs > 5,
        "negate-one twists barely ran: {}",
        mx.counters.twist_negs
    );
    assert!(
        mx.counters.twist_cnots > 5,
        "negate-both twists barely ran: {}",
        mx.counters.twist_cnots
    );
    assert!(
        mx.counters.twist_relabels > 0,
        "twists never relabeled a gate"
    );
    assert!(mx.remaining_g57() <= comp0, "fossil count increased");
    assert!(mx.counters.merges() > 0, "no merges alongside twists");
    mx.global_check();
}

// twist_neg_p = 0 gives PURE positive swaps: no wire is ever negated, so no
// interior polarity flips (same_pol stays 0) and only the pure-swap counter
// moves -- yet the 3-CNOT brackets are still inserted (comp=0 material
// present). This is the control that separates "foreign CNOTs" from
// "polarity scrambling".
#[test]
fn twist_neg_p_zero_is_pure_swap() {
    let params = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_neg_p: 0.0,
        twist_min_len: 4,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 7,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(random_g57_circuit(17, 16, 400), 16, params);
    mx.run();
    assert!(
        mx.counters.twist_swaps > 20,
        "no pure swaps ran: {}",
        mx.counters.twist_swaps
    );
    assert_eq!(
        mx.counters.twist_negs, 0,
        "negate-one fired at twist_neg_p=0"
    );
    assert_eq!(
        mx.counters.twist_cnots, 0,
        "negate-both fired at twist_neg_p=0"
    );
    assert_eq!(mx.g57_census().same_pol, 0, "pure swap flipped polarity");
    assert!(mx.counters.twist_relabels > 0, "pure swap never relabeled");
    mx.global_check();
}

// With twist weights at zero no twist path may ever be taken — not even a
// skipped attempt from floating-point dust in the weight subtractions —
// so per-move RNG consumption (and hence every seed's trajectory) matches
// the pre-twist chain exactly.
#[test]
fn twist_weights_zero_is_inert() {
    let gates = random_mixed_circuit(3, 16, 300);
    let params = MixParams {
        k_max: 5,
        moves: 10_000,
        target_size: 300,
        temp: 20.0,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut a = Mixer::new(gates, 16, params);
    a.run();
    assert_eq!(
        a.counters.twist_negs
            + a.counters.twist_swaps
            + a.counters.twist_cnots
            + a.counters.twist_skips,
        0
    );
}
