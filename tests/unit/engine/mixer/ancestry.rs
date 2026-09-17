// The backward ancestor-span statistic must be populated under SAMPLED
// ancestry (where the aggregate min/max ancspan= is switched off) and must stay
// inside its definitional bounds. It is bucket occupancy / entropy over
// the INPUT circuit, so all three live in [0,1].
#[test]
fn sampled_ancestor_span_is_populated_and_bounded() {
    let gates = random_mixed_circuit(53, 16, 400);
    let params = MixParams {
        k_max: 6,
        moves: 4_000,
        target_size: 400,
        temp: 20.0,
        anc_samples: 64,
        p_twist: 0.0,
        shuffle_rate: 0.0,
        report_every: u64::MAX,
        seed: 17,
        ..MixParams::default()
    };
    let mut mx = Mixer::new_with_db(gates, 16, params, FrozenDb::empty());
    mx.run();
    let line = mx.tracer_report();
    assert!(
        line.contains("ancspan cov="),
        "ancspan block missing: {line}"
    );
    let grab = |k: &str| -> f64 {
        let i = line.find(k).unwrap_or_else(|| panic!("no {k} in {line}")) + k.len();
        line[i..]
            .split(|c: char| c == ' ' || c == '|')
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (cov, ent, sd) = (grab("ancspan cov="), grab("ent="), grab("sd="));
    for (n, v) in [("cov", cov), ("ent", ent), ("sd", sd)] {
        assert!((0.0..=1.0).contains(&v), "ancspan {n}={v} out of [0,1]");
    }
    // The aggregate min/max form stays off under sampling. The sampled
    // span appears in the tracers line; the mv= aggregates remain zero.
    assert_eq!(
        mx.anc_stats(),
        (0.0, 0.0),
        "sampled mode must still leave anc=/ancspan= at 0"
    );
}

// Seam consumption must propagate ancestry: a bracket word that consumes
// context takes the union of the consumed litters' ancestor sets, matching
// the ancestry semantics of a database splice.
#[test]
fn g57_twist_consumption_inherits_ancestry() {
    let gates = random_mixed_circuit(31, 16, 300);
    let params = MixParams {
        k_max: 6,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.3,
        twist_min_len: 4,
        twist_g57: true,
        local_verify: true,
        ancestors: true,
        verify_every: 1_000,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    assert!(mx.counters.tg_consumed > 0, "no consumption to test");
    let inherited = mx.arena.ids_in_order().iter().any(|&id| {
        let m = mx.meta_of(id);
        m.origin == ORIGIN_SYNTH
            && mx
                .anc
                .get(&m.litter)
                .is_some_and(|bits| bits.iter().any(|&w| w != 0))
    });
    assert!(inherited, "no synthetic gate carries inherited ancestry");
    mx.global_check();
}

// Sampled ancestry must agree with exact ancestry on the quantity they both
// measure. Tracer choice comes from a dedicated rng, so the two runs follow
// the IDENTICAL chain (asserted gate-for-gate) and the only difference is
// the instrument -- which makes this a real calibration rather than two
// independent samples that happen to be close.
#[test]
fn sampled_ancestry_calibrates_to_exact() {
    let gates = random_mixed_circuit(31, 16, 400);
    let base = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.05,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 9,
        ..MixParams::default()
    };
    let mut ex = Mixer::new(
        gates.clone(),
        16,
        MixParams {
            ancestors: true,
            ..base.clone()
        },
    );
    ex.run();
    let mut sa = Mixer::new(
        gates.clone(),
        16,
        MixParams {
            anc_samples: 128,
            ..base.clone()
        },
    );
    sa.run();

    // Same chain: the instrument may not perturb the walk.
    assert_eq!(
        ex.arena.to_vec(),
        sa.arena.to_vec(),
        "tracer selection changed the trajectory"
    );

    // Exact mode still reports anc/span; sampled mode deliberately does not.
    assert!(ex.anc_stats().0 > 0.0, "exact mode lost its anc reading");
    assert_eq!(
        sa.anc_stats(),
        (0.0, 0.0),
        "sampled mode must not fill anc=/ancspan="
    );
    assert!(
        sa.tracer_report().contains("tracers: K=128"),
        "{}",
        sa.tracer_report()
    );

    let exact = ex.anc_incidence();
    let est = sa.anc_incidence();
    assert!(exact > 0.0, "no incidence to compare");
    let rel = (est - exact).abs() / exact;
    assert!(
        rel < 0.25,
        "sampled incidence off by {rel:.3} (exact {exact:.0}, est {est:.0})"
    );
}

// The joint gen x anc census must partition the circuit exactly (every gate
// lands in exactly one band, including the GEN_FRESH sentinel band) and must
// work in BOTH ancestry modes.
#[test]
fn gen_anc_census_partitions_the_circuit() {
    let gates = random_mixed_circuit(23, 16, 400);
    let base = MixParams {
        k_max: 5,
        moves: 20_000,
        target_size: 600,
        temp: 20.0,
        p_twist: 0.1, // mint some GEN_FRESH bracket material
        gen_target: 5,
        verify_every: 5_000,
        report_every: u64::MAX,
        seed: 4,
        ..MixParams::default()
    };
    for (label, p) in [
        (
            "exact",
            MixParams {
                ancestors: true,
                ..base.clone()
            },
        ),
        (
            "sampled",
            MixParams {
                anc_samples: 64,
                ..base.clone()
            },
        ),
    ] {
        let mut mx = Mixer::new(gates.clone(), 16, p);
        mx.run();
        let line = mx.gen_anc_report();
        assert!(
            line.starts_with("[circuit_mixer] gen-anc: r="),
            "{label}: {line}"
        );
        // Band counts must sum to the circuit size.
        let total: usize = line
            .split('|')
            .filter_map(|s| s.split("n=").nth(1))
            .filter_map(|s| s.split_whitespace().next())
            .filter_map(|s| s.parse::<usize>().ok())
            .sum();
        // The first "n=" is the real-gen count in the header, so subtract it.
        let hdr: usize = line
            .split("(n=")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|f| f as usize)
            .expect("header count");
        assert_eq!(
            total - hdr,
            mx.arena.len(),
            "{label} bands do not partition: {line}"
        );
        assert!(
            hdr <= mx.arena.len(),
            "{label}: more real-gen gates than gates"
        );
        // Exact mode reports span per band, sampled mode must not.
        assert_eq!(line.contains("span="), label == "exact", "{label}: {line}");
    }
}

// The whole point of sampling: it runs on inputs the exact instrument
// refuses (it asserts n <= 20_000). Cost is K bits per litter regardless of
// input size, so the ancestor map stays far smaller than the circuit.
#[test]
fn sampled_ancestry_runs_past_the_exact_cap() {
    let n = 25_000;
    let gates = random_mixed_circuit(5, 24, n);
    let params = MixParams {
        k_max: 5,
        moves: 3_000,
        target_size: n + 200,
        temp: 50.0,
        anc_samples: 64,
        verify_every: 1_500,
        report_every: u64::MAX,
        seed: 3,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 24, params);
    mx.run();
    assert!(
        mx.anc_incidence() > 0.0,
        "sampled ancestry recorded nothing"
    );
    let rep = mx.tracer_report();
    assert!(rep.contains("K=64 of m=25000"), "{rep}");
    // Only litters that actually carry a tracer are stored, so the map is a
    // small fraction of the circuit -- this is what makes it scale.
    assert!(
        mx.anc.len() < mx.arena.len(),
        "ancestor map ({}) is not smaller than the circuit ({})",
        mx.anc.len(),
        mx.arena.len()
    );
    mx.global_check();
}

// Ancestry treats a cross as a DB splice over the window {g, h}: every
// output of a crossing — the intact pivot included — carries the UNION of
// both parents' ancestor sets. Verified through the journal, which
// records the pre-cross litters: every piece of a live entry must read
// exactly union(set(litters[0]), set(litters[1])).
#[test]
fn cross_outputs_carry_union_ancestry() {
    let gates = random_mixed_circuit(43, 16, 200);
    let n = gates.len();
    let params = MixParams {
        k_max: 6,
        moves: 6_000,
        target_size: 4 * n,
        temp: 20.0,
        ancestors: true,
        report_every: u64::MAX,
        seed: 5,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let mut checked = 0usize;
    let mut expected = vec![0u64; mx.anc_words];
    let mut got = vec![0u64; mx.anc_words];
    for e in mx.journal.iter() {
        let live = e
            .after
            .iter()
            .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st);
        if !live {
            continue;
        }
        expected.iter_mut().for_each(|w| *w = 0);
        mx.anc_or_into(e.litters[0], &mut expected);
        mx.anc_or_into(e.litters[1], &mut expected);
        for &(id, _) in &e.after {
            got.iter_mut().for_each(|w| *w = 0);
            mx.anc_or_into(mx.meta_of(id).litter, &mut got);
            assert_eq!(
                got, expected,
                "a cross output (intact pivot included) must carry the parents' union"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "the run must leave live journal entries to check"
    );
    mx.global_check();
}

// The pre-cross litters a live journal entry will restore must survive
// anc_prune: the cross relabels EVERY output to the union litter, so the
// parents' litters can go extinct among live gates, and pruning their
// sets would make a later undo restore ancestry-less litters silently.
#[test]
fn anc_prune_keeps_litters_live_journal_entries_restore() {
    let gates = random_mixed_circuit(47, 16, 200);
    let n = gates.len();
    let params = MixParams {
        k_max: 6,
        moves: 6_000,
        target_size: 4 * n,
        temp: 20.0,
        ancestors: true,
        report_every: u64::MAX,
        seed: 6,
        ..MixParams::default()
    };
    let mut mx = Mixer::new(gates, 16, params);
    mx.run();
    let live_entries: Vec<[u64; 2]> = mx
        .journal
        .iter()
        .filter(|e| {
            e.after
                .iter()
                .all(|&(id, st)| mx.arena.is_linked(id) && mx.arena.stamp(id) == st)
        })
        .map(|e| e.litters)
        .collect();
    assert!(
        !live_entries.is_empty(),
        "need live journal entries to make the test bite"
    );
    let resolve = |mx: &Mixer, l: u64| {
        let mut bits = vec![0u64; mx.anc_words];
        mx.anc_or_into(l, &mut bits);
        bits
    };
    let before: Vec<[Vec<u64>; 2]> = live_entries
        .iter()
        .map(|ls| [resolve(&mx, ls[0]), resolve(&mx, ls[1])])
        .collect();
    mx.anc_prune();
    for (ls, want) in live_entries.iter().zip(before.iter()) {
        assert_eq!(
            resolve(&mx, ls[0]),
            want[0],
            "prune dropped a restorable litter's set"
        );
        assert_eq!(
            resolve(&mx, ls[1]),
            want[1],
            "prune dropped a restorable litter's set"
        );
    }
}

// Sidecar round trip in both universes: write -> read must reproduce the
// per-gate resolved sets verbatim, and importing into a FRESH mixer over
// the same circuit must resolve identically (the phase-boundary use).
#[test]
fn anc_sidecar_round_trips_exact_and_sampled() {
    for sampled in [false, true] {
        let gates = random_mixed_circuit(51, 16, 150);
        let params = MixParams {
            k_max: 6,
            moves: 4_000,
            target_size: 3 * gates.len(),
            temp: 20.0,
            ancestors: !sampled,
            anc_samples: if sampled { 32 } else { 0 },
            report_every: u64::MAX,
            seed: 7,
            ..MixParams::default()
        };
        let mut mx = Mixer::new(gates, 16, params);
        mx.run();
        let path = std::env::temp_dir().join(format!("circuit_mixer_anc_test_{sampled}.anc"));
        let path = path.to_str().unwrap().to_string();
        mx.write_anc_sidecar(&path).expect("write sidecar");
        let sc = Mixer::read_anc_sidecar(&path).expect("read sidecar");
        assert_eq!(sc.sampled, sampled);
        assert_eq!(sc.m, mx.anc_m, "universe size must survive");
        assert_eq!(sc.tracers, mx.anc_tracers, "tracer list must survive");
        let resolved: Vec<Vec<u64>> = {
            let mut v = Vec::new();
            let mut bits = vec![0u64; mx.anc_words];
            let mut cur = mx.arena.head();
            while cur != NIL {
                bits.iter_mut().for_each(|w| *w = 0);
                mx.anc_or_into(mx.meta_of(cur).litter, &mut bits);
                v.push(bits.clone());
                cur = mx.arena.neighbor(cur, Dir::R);
            }
            v
        };
        assert_eq!(
            sc.sets, resolved,
            "sidecar rows must equal the resolved per-gate sets"
        );

        // Check the writer's header as part of the sidecar roundtrip.
        let serialized = std::fs::read_to_string(&path).expect("read sidecar text");
        assert!(serialized.starts_with("mixer-anc 1 "));

        // Import into a fresh run over the SAME circuit (ancestry off at
        // construction; the sidecar defines the universe).
        let out_gates = mx.arena.to_vec();
        let params2 = MixParams {
            k_max: 6,
            moves: 2_000,
            temp: 20.0,
            report_every: u64::MAX,
            seed: 8,
            ..MixParams::default()
        };
        let mut mx2 = Mixer::new(out_gates, 16, params2);
        let sc2 = Mixer::read_anc_sidecar(&path).expect("re-read sidecar");
        mx2.import_ancestry(sc2);
        assert_eq!(
            mx2.anc_m, mx.anc_m,
            "imported universe must be the ORIGINAL m"
        );
        assert_eq!(mx2.anc_tracers, mx.anc_tracers);
        let resolved2: Vec<Vec<u64>> = {
            let mut v = Vec::new();
            let mut bits = vec![0u64; mx2.anc_words];
            let mut cur = mx2.arena.head();
            while cur != NIL {
                bits.iter_mut().for_each(|w| *w = 0);
                mx2.anc_or_into(mx2.meta_of(cur).litter, &mut bits);
                v.push(bits.clone());
                cur = mx2.arena.neighbor(cur, Dir::R);
            }
            v
        };
        assert_eq!(
            resolved2, resolved,
            "imported ancestry must resolve identically"
        );
        // The imported run must keep walking and unioning without issue.
        mx2.run();
        mx2.global_check();
        assert_eq!(
            mx2.anc_m, mx.anc_m,
            "the universe must not drift during the run"
        );
        let _ = std::fs::remove_file(&path);
    }
}
