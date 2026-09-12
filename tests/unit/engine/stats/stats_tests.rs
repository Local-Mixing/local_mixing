use super::*;
use crate::circuit::xgate::XGate;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn conj(t: u16, lits: &[(u16, bool)]) -> XGate {
    XGate::conj(t, lits.iter().copied()).unwrap()
}

#[test]
fn fanout_counts_reads_until_overwrite() {
    // g0 writes 0; g1 and g2 read 0; g3 overwrites 0; g4 reads 0 again.
    let gates = vec![
        conj(0, &[(1, true)]),
        conj(2, &[(0, true)]),
        conj(3, &[(0, false), (1, true)]),
        conj(0, &[(2, true)]),
        conj(4, &[(0, true)]),
    ];
    let fan = fanouts(gates.iter(), 5);
    assert_eq!(fan, vec![2, 1, 0, 1, 0]);
}

#[test]
fn leeway_counts_commuting_neighbors() {
    // Middle gate collides with nothing (disjoint wires): full box.
    let free = vec![
        conj(0, &[(1, true)]),
        conj(2, &[(3, true)]),
        conj(4, &[(5, true)]),
    ];
    assert_eq!(leeway_at(&free, 1, 100), 2);
    // g1 reads g0's target and g2 overwrites nothing shared... make g2 read g1.
    let wedged = vec![
        conj(0, &[(1, true)]),
        conj(2, &[(0, true)]),
        conj(3, &[(2, true)]),
    ];
    assert_eq!(leeway_at(&wedged, 1, 100), 0);
}

#[test]
fn diffusion_and_autocorr_track_mixing() {
    // Unmixed: origins in blocks, adjacent correlation ~1, diffusion small.
    let blocked: Vec<u32> = (0..1000u32).flat_map(|o| [o, o]).collect();
    assert!(adjacent_origin_autocorr(&blocked) > 0.99);
    assert!(origin_diffusion(&blocked) < 0.01);
    // Well mixed: same multiset, positions scrambled.
    let mut rng = StdRng::seed_from_u64(7);
    let mut mixed = blocked.clone();
    for i in (1..mixed.len()).rev() {
        mixed.swap(i, rng.random_range(0..=i));
    }
    assert!(adjacent_origin_autocorr(&mixed).abs() < 0.1);
    let d = origin_diffusion(&mixed);
    assert!(
        d > 0.15,
        "diffusion {d} should approach uniform after shuffle"
    );
    // Synthetic entries are ignored, not counted as an origin.
    let with_synth: Vec<u32> = [0, ORIGIN_SYNTH, 0, 1, ORIGIN_SYNTH, 1].to_vec();
    assert!(origin_diffusion(&with_synth) > 0.0);
}

#[test]
fn spread_quantiles_split_singles_from_multis() {
    // Origin 0: pieces at 0 and 1 (spread = 2 gates in a 4-gate circuit:
    // std of {0, .25} * 4 = .5). Origin 1: one piece. Origin 2: one piece.
    let origins: Vec<u32> = vec![0, 0, 1, 2];
    let (single_frac, quants, below) = origin_spread_quantiles(&origins, &[0.5], 1.0);
    assert!((single_frac - 0.5).abs() < 1e-12);
    assert!((quants[0] - 0.5).abs() < 1e-12);
    assert!((below - 1.0).abs() < 1e-12);
    // Tightly clumped vs dispersed origins separate in the quantiles.
    let mut blocked: Vec<u32> = (0..500u32).flat_map(|o| [o, o]).collect();
    let (_, q_lo, _) = origin_spread_quantiles(&blocked, &[0.9], 1.0);
    assert!(q_lo[0] < 2.0, "adjacent pieces have sub-2-gate spread");
    let mut rng = StdRng::seed_from_u64(3);
    for i in (1..blocked.len()).rev() {
        blocked.swap(i, rng.random_range(0..=i));
    }
    let (_, q_hi, _) = origin_spread_quantiles(&blocked, &[0.1], 1.0);
    assert!(q_hi[0] > 20.0, "shuffled pieces spread widely even at p10");
}

#[test]
fn entropy_and_span_sanity() {
    assert!((entropy_bits([1u64, 1, 1, 1].into_iter()) - 2.0).abs() < 1e-12);
    assert_eq!(entropy_bits(std::iter::empty()), 0.0);
    let gates = vec![conj(0, &[(1, true)]); 64];
    let mut rng = StdRng::seed_from_u64(1);
    let (mean, mn, mx) = window_wire_span(&gates, 4, 32, 50, &mut rng);
    assert_eq!((mean, mn, mx), (2.0, 2, 2));
    let (bits, distinct) = pair_cooccurrence_entropy(gates.iter(), 4);
    assert_eq!(distinct, 1);
    assert_eq!(bits, 0.0);
}
