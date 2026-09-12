use super::*;

#[test]
fn cut_points_alternate_lattices_and_keep_old_seams_mid_piece() {
    let mut rng = StdRng::seed_from_u64(7);
    for &(len, p) in &[(1000usize, 4usize), (997, 3), (5000, 8), (64, 2)] {
        for j in [0.0, 0.125] {
            // Round 0: p pieces of about len / p.
            let c0 = cut_points(&[], len, p, 0, j, &mut rng);
            assert_eq!(c0.len(), p - 1, "round 0 cuts (len {len}, p {p})");
            let l = len / p;
            for w in [0]
                .iter()
                .chain(c0.iter())
                .chain([len].iter())
                .collect::<Vec<_>>()
                .windows(2)
            {
                let piece = w[1] - w[0];
                assert!(
                    piece as f64 >= l as f64 * (1.0 - 2.0 * j) - 2.0
                        && piece as f64 <= l as f64 * (1.0 + 2.0 * j) + 2.0,
                    "round-0 piece {piece} vs slice {l} (j {j})"
                );
            }
            // Simulate uneven growth: seams drift by up to +10% of a slice.
            let grow = |cuts: &[usize], len: usize| -> (Vec<usize>, usize) {
                let mut out = Vec::new();
                let mut extra = 0usize;
                for (k, &c) in cuts.iter().enumerate() {
                    extra += (k * 7) % (l / 10 + 1);
                    out.push(c + extra);
                }
                (out, len + extra + 3)
            };
            let (seams1, len1) = grow(&c0, len);
            // Round 1 (shifted): p cuts, p + 1 pieces, each old seam >=
            // (0.5 - j) of its interval away from every new cut.
            let c1 = cut_points(&seams1, len1, p, 1, j, &mut rng);
            assert_eq!(c1.len(), p, "round 1 cuts");
            let bounds1: Vec<usize> = [0]
                .iter()
                .chain(seams1.iter())
                .chain([len1].iter())
                .copied()
                .collect();
            for &s in &seams1 {
                let i = bounds1.iter().position(|&b| b == s).unwrap();
                let interval = (bounds1[i] - bounds1[i - 1]).min(bounds1[i + 1] - bounds1[i]);
                let d = c1.iter().map(|&c| c.abs_diff(s)).min().unwrap();
                assert!(
                    d as f64 >= (0.5 - j) * interval as f64 - 2.0,
                    "seam {s} too close to a cut ({d} < {}), j {j}",
                    (0.5 - j) * interval as f64
                );
            }
            // Round 2 (unshifted): p - 1 cuts, p pieces, same property.
            let (seams2, len2) = grow(&c1, len1);
            let c2 = cut_points(&seams2, len2, p, 2, j, &mut rng);
            assert_eq!(c2.len(), p - 1, "round 2 cuts");
            for &s in &seams2 {
                let d = c2.iter().map(|&c| c.abs_diff(s)).min().unwrap();
                assert!(
                    d as f64 >= (0.5 - j) * (l as f64 * 0.4),
                    "round-2 seam {s} too close ({d})"
                );
            }
        }
    }
    // p = 1: no cuts.
    assert!(cut_points(&[], 1000, 1, 0, 0.125, &mut rng).is_empty());
    assert!(cut_points(&[500], 1000, 1, 1, 0.125, &mut rng).is_empty());
}

#[test]
fn piecewise_auto_recomputes_count_and_keeps_shift_phase() {
    let cfg = PieceCfg {
        min_block_size: Some(100),
        jitter: 0.0,
        ..PieceCfg::default()
    };
    let mut partition = Partition::default();
    let mut rng = StdRng::seed_from_u64(7);
    // P grows, shrinks, falls to one, then grows again. Shifted end
    // blocks are allowed below 100; the divisor is not a hard floor.
    for (round, (len, p, expected)) in [
        (400, 4, vec![100, 200, 300]),
        (600, 6, vec![50, 150, 250, 350, 450, 550]),
        (300, 3, vec![100, 200]),
        (199, 1, vec![]),
        (200, 2, vec![100]),
        (300, 3, vec![50, 150, 250]),
    ]
    .into_iter()
    .enumerate()
    {
        let cuts = partition.cuts(&cfg, len, round, &mut rng);
        assert_eq!(partition.pieces, p, "nominal count at round {round}");
        assert_eq!(cuts, expected, "cuts at round {round}");
        partition.seams = cuts;
    }
    assert_eq!(cfg.pieces_for_len(0), 1);
    assert_eq!(cfg.pieces_for_len(99), 1);
    assert_eq!(cfg.pieces_for_len(299), 2, "integer division rounds down");
}

#[test]
fn piecewise_auto_one_block_transitions_with_jitter() {
    let cfg = PieceCfg {
        min_block_size: Some(100),
        ..PieceCfg::default()
    };
    let mut partition = Partition::default();
    let mut rng = StdRng::seed_from_u64(13);
    // Initial P=1 grows on an odd round; an even round then returns to
    // one block, stays there, and regrows on both shift phases.
    for (round, len) in [99, 400, 199, 150, 450, 80, 50, 500]
        .into_iter()
        .enumerate()
    {
        let cuts = partition.cuts(&cfg, len, round, &mut rng);
        let p = (len / 100).max(1);
        let expected = if p == 1 {
            0
        } else if round % 2 == 1 {
            p
        } else {
            p - 1
        };
        assert_eq!(cuts.len(), expected, "round {round}");
        assert!(cuts.iter().all(|&c| c > 0 && c < len));
        assert!(cuts.windows(2).all(|w| w[0] < w[1]));
        partition.seams = cuts;
    }
}

#[test]
fn piecewise_auto_tracks_actual_seams_while_count_is_unchanged() {
    for jitter in [0.0, 0.125] {
        let cfg = PieceCfg {
            min_block_size: Some(100),
            jitter,
            ..PieceCfg::default()
        };
        let mut partition = Partition {
            pieces: 4,
            seams: vec![110, 215, 350],
        };
        let mut auto_rng = StdRng::seed_from_u64(19);
        let mut fixed_rng = auto_rng.clone();
        for round in 1..5 {
            let expected = cut_points(&partition.seams, 450, 4, round, jitter, &mut fixed_rng);
            let got = partition.cuts(&cfg, 450, round, &mut auto_rng);
            assert_eq!(
                got, expected,
                "must use fixed-P seam logic at round {round}"
            );
            partition.seams = got;
        }
    }
}

#[test]
fn seeds_are_distinct_and_stable() {
    let mut seen = std::collections::HashSet::new();
    for r in 0..50 {
        for i in 0..10 {
            assert!(seen.insert(seed_of(12345, r, i)), "collision at ({r}, {i})");
            assert_eq!(seed_of(12345, r, i), seed_of(12345, r, i));
        }
    }
    assert_ne!(seed_of(1, 0, 0), seed_of(2, 0, 0));
}

#[test]
fn counters_fold_is_field_wise() {
    let mut a = MixCounters::default();
    let mut b = MixCounters::default();
    a.db_comp_hits = 3;
    b.db_comp_hits = 4;
    a.split_span_hist[2] = 5;
    b.split_span_hist[2] = 6;
    b.pair_box_max = 9;
    a.pair_box_max = 4;
    b.len_attempts = vec![1, 2, 3];
    a.len_attempts = vec![10];
    b.splice_sizes = vec![vec![1, 1], vec![2]];
    b.moves = 77;
    a.moves = 5;
    a += &b;
    assert_eq!(a.db_comp_hits, 7);
    assert_eq!(a.split_span_hist[2], 11);
    assert_eq!(a.pair_box_max, 9);
    assert_eq!(a.len_attempts, vec![11, 2, 3]);
    assert_eq!(a.splice_sizes, vec![vec![1, 1], vec![2]]);
    assert_eq!(a.moves, 5, "the move clock is W's, never summed");
}
