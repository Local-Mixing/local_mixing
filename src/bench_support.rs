use crate::circuit::circuit::{CircuitSeq, Polynomial};
use crate::random::random_data::random_circuit;
use std::collections::HashMap;

/// Generate reproducible polynomials for a given circuit configuration.
pub fn gen_polys(seed: u64, n: usize, m: usize) -> Vec<Polynomial> {
    fastrand::seed(seed);
    let ckt = random_circuit(n, m);
    trimmed_polys(&ckt)
}

/// Trim a circuit to its touched wires, remap them densely, then emit polynomials.
///
/// This mirrors `CircuitSeq::canonicalize_polys`: unused wires are removed before
/// polynomial canonicalization, so both benchmarked algorithms work on the same
/// minimal input rather than on trailing identity wires.
pub fn trimmed_polys(ckt: &CircuitSeq) -> Vec<Polynomial> {
    let used = ckt.used_wires();
    let wire_map: HashMap<u16, u16> = used
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as u16))
        .collect();
    let remapped = CircuitSeq {
        gates: ckt
            .gates
            .iter()
            .map(|&[t, c1, c2]| [wire_map[&t], wire_map[&c1], wire_map[&c2]])
            .collect(),
    };
    remapped.to_polynomial(used.len(), 0, remapped.gates.len())
}

/// Default gate count used elsewhere in the repository: 2 * n * ln(n).
pub fn default_m(n: usize) -> usize {
    2 * ((n as f64) * (n as f64).ln()) as usize
}

pub const N_GRID: &[usize] = &[8, 12, 16, 20, 24, 28, 32];
pub const SEEDS: &[u64] = &[1, 2, 3, 4, 5];

/// Iterate over the shared grid, optionally capped for bounded benchmark runs.
pub fn selected_n_grid() -> impl Iterator<Item = usize> {
    let selected = std::env::var("BENCH_N").ok().map(|value| {
        value
            .split(',')
            .filter_map(|item| item.trim().parse().ok())
            .collect::<Vec<usize>>()
    });
    let max_n = std::env::var("BENCH_MAX_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX);
    N_GRID
        .iter()
        .copied()
        .filter(move |&n| n <= max_n && selected.as_ref().is_none_or(|values| values.contains(&n)))
}
