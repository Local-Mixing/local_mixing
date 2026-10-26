use super::*;
use crate::circuit::xgate::XGate;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn tof(t: u16, a: u16, b: u16) -> XGate {
    XGate::conj(t, [(a, true), (b, true)]).unwrap()
}

fn random_gates(seed: u64, width: u16, m: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
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

fn blank_source() -> Source {
    Source::new(Vec::new(), 4, BTreeSet::new())
}

/// A census of `m` gates with the given affine (gid, start, end) intervals.
fn census_with(m: usize, affine: Vec<(usize, usize, usize)>) -> Census {
    Census {
        gates: m,
        width: 16,
        samples: 0,
        affine: affine
            .into_iter()
            .map(|(gid, s, e)| Interval {
                source_gate: gid,
                start_cut: s,
                end_cut: e,
                constant: false,
                support: vec![0],
                is_output: false,
                validated: true,
            })
            .collect(),
        direct: Vec::new(),
        audit: Default::default(),
        seconds: 0.0,
    }
}

fn params(term_cap: usize, segments: usize, avoid_radius: usize) -> PlanParams {
    PlanParams {
        term_cap,
        max_block_gates: 400,
        lookahead: 16,
        // Small: the lookahead past an eligible end otherwise grows the
        // polynomials to the 4x per-wire cap (slow in debug builds).
        product_pairs: 100_000,
        segments,
        avoid_radius,
    }
}

/// From-scratch composition (the prototype's `compose_window`, no limits).
fn reference<const W: usize>(gates: &[XGate], width: usize) -> Vec<Poly<W>> {
    let mut polys: Vec<Poly<W>> = (0..width)
        .map(|w| [Mono::<W>::bit(w)].into_iter().collect())
        .collect();
    for g in gates {
        let mut prod: Poly<W> = [Mono::<W>::ZERO].into_iter().collect();
        for &(w, pos) in &g.ctrls {
            let mut cp = polys[w as usize].clone();
            if !pos {
                toggle(&mut cp, Mono::<W>::ZERO);
            }
            let mut next = Poly::default();
            for a in &prod {
                for b in &cp {
                    toggle(&mut next, a.or(b));
                }
            }
            prod = next;
        }
        if g.comp {
            toggle(&mut prod, Mono::<W>::ZERO);
        }
        for m in prod {
            toggle(&mut polys[g.target as usize], m);
        }
    }
    polys
}

#[test]
fn incremental_matches_reference_composition() {
    let mut rng = StdRng::seed_from_u64(91823);
    let lim = Limits {
        total: 1 << 20,
        wire: 1 << 20,
        pairs: 1 << 24,
        gates: 1000,
    };
    for _ in 0..12 {
        let n = 7;
        let mut st = Compose::<1>::new(n);
        let mut gates = Vec::new();
        for _ in 0..30 {
            let t = rng.random_range(0..7u16);
            let a = (t + 1 + rng.random_range(0..6u16)) % 7;
            let mut b = (a + 1) % 7;
            while b == a || b == t {
                b = (b + 1) % 7;
            }
            let mut g =
                XGate::conj(t, [(a, rng.random::<bool>()), (b, rng.random::<bool>())]).unwrap();
            g.comp = rng.random::<bool>();
            gates.push(g.clone());
            st.add(&g, &lim).unwrap();
            assert_eq!(st.p, reference::<1>(&gates, n));
            assert_eq!(st.total, st.p.iter().map(|p| p.len()).sum::<usize>());
        }
        assert_eq!(st.changed, 30);
    }
}

#[test]
fn rejection_does_not_mutate_previous_prefix() {
    let mut st = Compose::<1>::new(3);
    let before = st.p.clone();
    let lim = Limits {
        total: 3,
        wire: 1,
        pairs: 10,
        gates: 100,
    };
    assert_eq!(st.add(&tof(0, 1, 2), &lim), Err("wire_terms"));
    assert_eq!(st.p, before);
    assert_eq!(st.total, 3);
    let lim = Limits { wire: 10, ..lim };
    assert_eq!(st.add(&tof(0, 1, 2), &lim), Err("total_terms"));
    assert_eq!(st.p, before);
    assert_eq!(st.total, 3);
    let lim = Limits { pairs: 0, ..lim };
    assert_eq!(st.add(&tof(0, 1, 2), &lim), Err("product_pairs"));
    assert_eq!(st.p, before);
    assert_eq!(st.changed, 0);
    let lim = Limits {
        total: 10,
        wire: 10,
        pairs: 10,
        gates: 100,
    };
    assert_eq!(st.add(&tof(0, 1, 2), &lim), Ok(()));
    assert_eq!(st.total, 4);
}

#[test]
fn clearance_respects_entire_exposure_lifetimes() {
    assert_eq!(
        signed_clearance(&[0, 0, 1, 1, 1, 0, 0]),
        vec![2, 1, -1, -2, -1, 1, 2]
    );
    assert_eq!(signed_clearance(&[1, 1, 1]), vec![-3, -3, -3]);
    assert_eq!(signed_clearance(&[0, 0, 0]), vec![3, 3, 3]);
}

fn check_partition(p: &Plan, m: usize, params: &PlanParams) {
    assert!(p.bounds.windows(2).all(|w| w[0] < w[1]));
    assert!(p.bounds.iter().all(|&b| b > 0 && b < m));
    let edges = p.edges(m);
    let mut blocks = p.blocks.clone();
    blocks.sort_by_key(|b| b.start);
    assert_eq!(blocks.len(), edges.len() - 1);
    for (k, b) in blocks.iter().enumerate() {
        assert_eq!((b.start, b.end), (edges[k], edges[k + 1]));
        assert!(b.end - b.start <= params.max_block_gates);
        assert!(b.end <= b.max_feasible_end);
        assert!(b.terms <= params.term_cap || b.stop_reason == "single_gate_exception");
    }
    assert!(p.seams.iter().all(|s| p.bounds.contains(s)));
}

#[test]
fn blind_plan_partitions_the_circuit_deterministically() {
    let (width, m) = (16usize, 2000usize);
    let g = random_gates(7, 16, m);
    let census = Census::empty(m, width);
    let pr = params(1000, 1, 0);
    let p = plan(&g, width, &census, &blank_source(), &pr, &[]);
    check_partition(&p, m, &pr);
    assert!(p.seams.is_empty());
    assert!(p.bounds.len() >= 10, "{} bounds", p.bounds.len());
    // C-blind: every cut scores alike, so the latest eligible cut wins.
    for b in &p.blocks {
        assert_eq!(b.end, b.max_feasible_end);
        assert_eq!(b.deep_affine_at_end, 0);
        assert_eq!(b.signed_clearance, m as i32 + 1);
    }
    let again = plan(&g, width, &census, &blank_source(), &pr, &[]);
    assert_eq!(
        (again.bounds, again.blocks, again.seams),
        (p.bounds, p.blocks, p.seams)
    );
}

#[test]
fn gate_cap_bounds_linear_blocks() {
    let mut rng = StdRng::seed_from_u64(3);
    let m = 500;
    let g: Vec<XGate> = (0..m)
        .map(|_| {
            let t = rng.random_range(0..16u16);
            let mut c = rng.random_range(0..16u16);
            while c == t {
                c = rng.random_range(0..16u16);
            }
            XGate::cnot(t, c)
        })
        .collect();
    let pr = PlanParams {
        term_cap: 1000,
        max_block_gates: 50,
        lookahead: 8,
        product_pairs: 1000,
        segments: 1,
        avoid_radius: 0,
    };
    let p = plan(&g, 16, &Census::empty(m, 16), &blank_source(), &pr, &[]);
    assert_eq!(p.bounds, (1..10).map(|k| 50 * k).collect::<Vec<_>>());
    assert!(p.blocks[..9].iter().all(|b| b.stop_reason == "gate_cap"));
    assert_eq!(p.blocks[9].stop_reason, "end");
    assert_eq!(p.single_gate_exceptions, 0);
}

#[test]
fn oversized_gate_becomes_its_own_block() {
    let g = vec![tof(0, 1, 2), tof(1, 2, 3), XGate::cnot(2, 3)];
    let pr = PlanParams {
        term_cap: 100,
        max_block_gates: 10,
        lookahead: 4,
        product_pairs: 0,
        segments: 1,
        avoid_radius: 0,
    };
    let p = plan(&g, 4, &Census::empty(3, 4), &blank_source(), &pr, &[]);
    assert_eq!(p.bounds, vec![1, 2]);
    assert_eq!(p.single_gate_exceptions, 3);
    assert!(
        p.blocks
            .iter()
            .all(|b| b.stop_reason == "single_gate_exception" && b.terms == 4)
    );
}

#[test]
fn segment_seams_land_on_unexposed_cuts() {
    let (width, m) = (16usize, 3000usize);
    let g = random_gates(8, 16, m);
    let census = census_with(
        m,
        vec![
            (0, 0, 700),
            (1, 702, 1500),
            (2, 1503, 2250),
            (3, 2253, m + 1),
        ],
    );
    let cov = census.coverage();
    let pr = params(1000, 4, 0);
    let p = plan(&g, width, &census, &blank_source(), &pr, &[]);
    assert_eq!(p.seams, vec![701, 1500, 2250]);
    for &s in &p.seams {
        assert_eq!(cov[s], 0);
    }
    check_partition(&p, m, &pr);
    for b in &p.blocks {
        assert_eq!(b.all_internal_affine_at_end, cov[b.end]);
        assert_eq!(b.deep_affine_at_end, cov[b.end]);
        assert_eq!(b.direct_locations_at_end, 0);
    }
    // Excluding birth 1 removes it from the deep count only.
    let src = Source::new(Vec::new(), 4, [1].into_iter().collect());
    let p2 = plan(&g, width, &census, &src, &pr, &[]);
    assert_eq!(p2.seams, p.seams);
    for b in &p2.blocks {
        if (702..1500).contains(&b.end) {
            assert_eq!((b.deep_affine_at_end, b.all_internal_affine_at_end), (0, 1));
        }
    }
    // Blocks never straddle a seam.
    for b in &p.blocks {
        assert!(p.seams.iter().all(|&s| s <= b.start || s >= b.end));
    }
}

#[test]
fn avoid_radius_moves_bounds_away_from_previous_round() {
    let (width, m) = (16usize, 2000usize);
    let g = random_gates(9, 16, m);
    let census = Census::empty(m, width);
    let base = params(2000, 1, 0);
    let first = plan(&g, width, &census, &blank_source(), &base, &[]);
    let avoid = PlanParams {
        avoid_radius: 3,
        ..base.clone()
    };
    let second = plan(&g, width, &census, &blank_source(), &avoid, &first.bounds);
    check_partition(&second, m, &avoid);
    assert!(second.seams.is_empty());
    for &b in &second.bounds {
        assert!(
            first.bounds.iter().all(|&p| b.abs_diff(p) > 3),
            "bound {b} within the radius of a previous bound"
        );
    }
    assert_ne!(second.bounds, first.bounds);
    // Without previous bounds the radius changes nothing (the one-segment
    // parallel path agrees with the sequential planner).
    let same = plan(&g, width, &census, &blank_source(), &avoid, &[]);
    assert_eq!(same.bounds, first.bounds);
    assert_eq!(same.blocks, first.blocks);
}

#[test]
fn profile_counts_clearance_and_split() {
    let m = 1000;
    let census = census_with(m, vec![(0, 100, 400), (0, 430, 900)]);
    let prof = Profile::new(&census, &blank_source());
    for (cut, counts) in [
        (50, [0, 0, 0]),
        (99, [0, 0, 0]),
        (100, [1, 1, 0]),
        (250, [1, 1, 0]),
        (399, [1, 1, 0]),
        (400, [0, 0, 0]),
        (415, [0, 0, 0]),
        (430, [1, 1, 0]),
        (899, [1, 1, 0]),
        (900, [0, 0, 0]),
        (1000, [0, 0, 0]),
    ] {
        assert_eq!(prof.counts(cut), counts, "cut {cut}");
    }
    for (cut, clearance) in [
        (50, 50),
        (100, -1),
        (250, -150),
        (399, -1),
        (400, 1),
        (414, 15),
        (415, 15),
        (429, 1),
        (430, -1),
        (950, 51),
    ] {
        assert_eq!(prof.clearance(cut), clearance, "cut {cut}");
    }
    assert_eq!(prof.split(200, 800), 415);
    // Excluding the birth drops the deep count; the all-affine count still
    // steers the split into the hole.
    let src = Source::new(Vec::new(), 4, [0].into_iter().collect());
    let prof2 = Profile::new(&census, &src);
    assert_eq!(prof2.counts(250), [0, 1, 0]);
    assert_eq!(prof2.split(200, 800), 415);
    // Direct intervals count in the third slot only.
    let mut with_direct = census_with(m, vec![]);
    with_direct.direct.push(Interval {
        source_gate: 0,
        start_cut: 10,
        end_cut: 20,
        constant: false,
        support: vec![3],
        is_output: false,
        validated: true,
    });
    let prof3 = Profile::new(&with_direct, &blank_source());
    assert_eq!(prof3.counts(15), [0, 0, 1]);
    assert_eq!(prof3.clearance(15), m as i32 + 1);
    // Unexposed blocks split at the midpoint; every split is interior.
    let blind = Profile::new(&Census::empty(m, 16), &blank_source());
    assert_eq!(blind.split(200, 800), 500);
    for (lo, hi) in [(0, 2), (10, 13), (0, m), (997, 1000)] {
        let c = blind.split(lo, hi);
        assert!(lo < c && c < hi, "split({lo}, {hi}) = {c}");
        let c = prof.split(lo, hi);
        assert!(lo < c && c < hi, "split({lo}, {hi}) = {c}");
    }
}

#[test]
fn plan_files_round_trip() {
    let (width, m) = (16usize, 600usize);
    let g = random_gates(11, 16, m);
    let census = census_with(m, vec![(0, 0, 300), (1, 310, m + 1)]);
    let pr = params(1000, 2, 0);
    let p = plan(&g, width, &census, &blank_source(), &pr, &[]);
    check_partition(&p, m, &pr);
    let dir = std::env::temp_dir().join(format!("lm_adaptive_plan_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_plan(&p, &dir).unwrap();
    assert_eq!(read_bounds(&dir.join("bounds")).unwrap(), p.bounds);
    let blocks = std::fs::read_to_string(dir.join("blocks.tsv")).unwrap();
    assert_eq!(blocks.lines().count(), p.blocks.len() + 1);
    assert!(blocks.starts_with("start\tend\tmax_feasible_end\tstop_reason\tterms\t"));
    let summary = std::fs::read_to_string(dir.join("summary.txt")).unwrap();
    assert!(summary.contains(&format!("blocks={}\n", p.blocks.len())));
    assert!(summary.contains(&format!(
        "seams={}\n",
        p.seams.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
    )));
    std::fs::write(dir.join("bad"), "5\n3\n").unwrap();
    assert!(read_bounds(&dir.join("bad")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
