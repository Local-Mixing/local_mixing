// Post-replacement dispersal: re-draw the gate order as a random linear extension of the
// circuit's dependency DAG, so replacement windows stop sitting as dense contiguous blocks.
//
// Semantics: two gates commute exactly when neither's active wire is a control wire of the
// other (Gate::collides_index). Same-active gates also commute (both XOR into the wire), but
// we conservatively chain writes to the same wire (WAW edges) so a reader only needs an edge
// to the LAST writer — transitivity then orders it after every earlier writer. The emitted
// order is a topological order of a DAG that contains every colliding pair, so it is
// reachable from the input order by swaps of adjacent non-colliding gates and computes the
// same function.
//
// This directly targets structural randomness: in matched random circuits the median gate
// leeway is O(wires), while spliced replacement blocks leave median leeway at 1-2. A random
// linear extension respreads commuting gates, raising the median and thinning the too-loose
// p99 tail, without changing gate count, wires, or function.

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::OnceLock;

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim();
            !value.is_empty()
                && !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
        }
        Err(_) => false,
    }
}

pub fn disperse_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("DISPERSE"))
}

pub fn disperse_chunk_size() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("DISPERSE_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000)
            .max(2)
    })
}

pub fn disperse_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::env::var("DISPERSE_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0xd15b_e75e_5eed_0001)
    })
}

/// Tournament size for the wire-cooldown ready-pick. 1 = plain uniform random extension;
/// larger values bias emission toward gates whose wires have been idle longest, which
/// spaces colliding gates apart the way uniform wire usage does in random circuits.
pub fn disperse_tournament() -> usize {
    static TOURNAMENT: OnceLock<usize> = OnceLock::new();
    *TOURNAMENT.get_or_init(|| {
        std::env::var("DISPERSE_TOURNAMENT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
            .max(1)
    })
}

/// Reorder `gates` (and `tags`, kept index-aligned) into a seeded random linear extension of
/// the dependency DAG, processed in contiguous chunks of `chunk` gates to bound memory.
/// Relative order across chunk boundaries is preserved, so correctness holds chunk-wise.
pub fn disperse_random_topo(
    gates: &mut [[u16; 3]],
    tags: Option<&mut [u32]>,
    chunk: usize,
    seed: u64,
) {
    disperse_random_topo_with(gates, tags, chunk, seed, disperse_tournament());
}

pub fn disperse_random_topo_with(
    gates: &mut [[u16; 3]],
    mut tags: Option<&mut [u32]>,
    chunk: usize,
    seed: u64,
    tournament: usize,
) {
    if let Some(tags) = tags.as_deref_mut() {
        assert_eq!(
            tags.len(),
            gates.len(),
            "disperse: tags must stay aligned with gates"
        );
    }
    let m = gates.len();
    let chunk = chunk.max(2);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut start = 0;
    while start < m {
        let end = (start + chunk).min(m);
        let chunk_tags = tags.as_deref_mut().map(|t| &mut t[start..end]);
        disperse_chunk(&mut gates[start..end], chunk_tags, &mut rng, tournament);
        start = end;
    }
}

fn add_edge(succ: &mut [Vec<u32>], indeg: &mut [u32], from: u32, to: u32) {
    if from != u32::MAX && from != to {
        succ[from as usize].push(to);
        indeg[to as usize] += 1;
    }
}

// Age of a wire since it was last emitted; never-emitted wires are infinitely cool.
fn wire_age(last_emit: usize, step: usize) -> usize {
    if last_emit == usize::MAX {
        usize::MAX
    } else {
        step - last_emit
    }
}

// A gate is only as cool as its hottest wire: the most recently emitted wire it touches
// is the one a colliding neighbor would share.
fn gate_coolness(gate: &[u16; 3], step: usize, last_emit: &[usize]) -> usize {
    gate.iter()
        .map(|&wire| wire_age(last_emit[wire as usize], step))
        .min()
        .unwrap()
}

fn disperse_chunk(
    gates: &mut [[u16; 3]],
    tags: Option<&mut [u32]>,
    rng: &mut StdRng,
    tournament: usize,
) {
    let m = gates.len();
    if m < 2 {
        return;
    }
    let max_wire = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .unwrap() as usize
        + 1;

    // Dependency edges via per-wire bookkeeping:
    //   RAW: reader of wire w after the last writer of w,
    //   WAR: writer of wire w after every reader since the previous write of w,
    //   WAW: writers of the same wire kept in order (conservative; makes RAW-to-last-writer
    //        transitively cover all earlier writers).
    let mut last_writer: Vec<u32> = vec![u32::MAX; max_wire];
    let mut readers_since_write: Vec<Vec<u32>> = vec![Vec::new(); max_wire];
    let mut succ: Vec<Vec<u32>> = vec![Vec::new(); m];
    let mut indeg: Vec<u32> = vec![0; m];

    for i in 0..m {
        let [a, c1, c2] = gates[i];
        let (a, c1, c2) = (a as usize, c1 as usize, c2 as usize);
        let iu = i as u32;

        add_edge(&mut succ, &mut indeg, last_writer[c1], iu);
        add_edge(&mut succ, &mut indeg, last_writer[c2], iu);
        for r in 0..readers_since_write[a].len() {
            let reader = readers_since_write[a][r];
            add_edge(&mut succ, &mut indeg, reader, iu);
        }
        add_edge(&mut succ, &mut indeg, last_writer[a], iu);

        readers_since_write[a].clear();
        last_writer[a] = iu;
        readers_since_write[c1].push(iu);
        readers_since_write[c2].push(iu);
    }

    // Randomized Kahn with a wire-cooldown tournament: sample up to `tournament` random
    // ready gates and emit the coolest (wires idle longest). tournament = 1 degenerates to
    // a plain uniform random linear extension. Cost is O(tournament) per gate.
    let mut ready: Vec<u32> = (0..m as u32).filter(|&i| indeg[i as usize] == 0).collect();
    let mut order: Vec<u32> = Vec::with_capacity(m);
    let mut last_emit: Vec<usize> = vec![usize::MAX; max_wire];
    let mut step = 0usize;
    while !ready.is_empty() {
        let mut pick = rng.random_range(0..ready.len());
        if tournament > 1 {
            let samples = tournament.min(ready.len());
            let mut best_cool = gate_coolness(&gates[ready[pick] as usize], step, &last_emit);
            for _ in 1..samples {
                let pos = rng.random_range(0..ready.len());
                let cool = gate_coolness(&gates[ready[pos] as usize], step, &last_emit);
                if cool > best_cool {
                    best_cool = cool;
                    pick = pos;
                }
            }
        }
        let gate = ready.swap_remove(pick);
        order.push(gate);
        for &wire in &gates[gate as usize] {
            last_emit[wire as usize] = step;
        }
        step += 1;
        for s in 0..succ[gate as usize].len() {
            let next = succ[gate as usize][s];
            indeg[next as usize] -= 1;
            if indeg[next as usize] == 0 {
                ready.push(next);
            }
        }
    }
    assert_eq!(
        order.len(),
        m,
        "disperse: dependency graph must be acyclic by construction"
    );

    let permuted: Vec<[u16; 3]> = order.iter().map(|&i| gates[i as usize]).collect();
    gates.copy_from_slice(&permuted);
    if let Some(tags) = tags {
        let permuted_tags: Vec<u32> = order.iter().map(|&i| tags[i as usize]).collect();
        tags.copy_from_slice(&permuted_tags);
    }
}

/// Normalized entropy (0..=1) of per-wire active-write and control-read counts. Random
/// circuits sit near 1.0 on both; a low value means wire usage is structurally uneven, so
/// leeway/fanout cannot match random by rescheduling alone — the deficit is in which wires
/// the gates use, not in what order they appear.
pub fn wire_usage_entropy(gates: &[[u16; 3]], n: usize) -> (f64, f64) {
    let max_wire = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    let size = n.max(max_wire);
    let mut active = vec![0usize; size];
    let mut control = vec![0usize; size];
    for gate in gates {
        active[gate[0] as usize] += 1;
        control[gate[1] as usize] += 1;
        control[gate[2] as usize] += 1;
    }
    (normalized_entropy(&active), normalized_entropy(&control))
}

// Normalized by ln(total wires) — NOT ln(populated wires) — so concentrating all usage on
// a few wires reads as low entropy rather than "evenly spread over the few wires it uses".
fn normalized_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 || counts.len() <= 1 {
        return 0.0;
    }
    let denom = (counts.len() as f64).ln();
    let total = total as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum();
    (entropy / denom).min(1.0)
}

#[derive(Debug, Clone, Copy)]
pub struct StatSummary {
    pub avg: f64,
    pub median: usize,
    pub p95: usize,
    pub p99: usize,
    pub max: usize,
}

fn summarize(mut values: Vec<usize>) -> StatSummary {
    if values.is_empty() {
        return StatSummary {
            avg: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
            max: 0,
        };
    }
    values.sort_unstable();
    let len = values.len();
    let pct = |p: f64| values[(((len - 1) as f64) * p).round() as usize];
    StatSummary {
        avg: values.iter().sum::<usize>() as f64 / len as f64,
        median: pct(0.50),
        p95: pct(0.95),
        p99: pct(0.99),
        max: values[len - 1],
    }
}

fn collides(g1: &[u16; 3], g2: &[u16; 3]) -> bool {
    crate::circuit::circuit::Gate::collides_index(g1, g2)
}

/// Leeway summary over gates sampled every `stride` positions (stride 1 = exhaustive).
/// Leeway of a gate is how many adjacent positions it can commute across on each side
/// before hitting a colliding gate.
pub fn leeway_stats(gates: &[[u16; 3]], stride: usize) -> StatSummary {
    let stride = stride.max(1);
    let mut values = Vec::with_capacity(gates.len() / stride + 1);
    let mut i = 0;
    while i < gates.len() {
        let cur = &gates[i];
        let mut leeway = 0usize;
        let mut j = i;
        while j > 0 && !collides(&gates[j - 1], cur) {
            leeway += 1;
            j -= 1;
        }
        let mut k = i;
        while k + 1 < gates.len() && !collides(&gates[k + 1], cur) {
            leeway += 1;
            k += 1;
        }
        values.push(leeway);
        i += stride;
    }
    summarize(values)
}

/// Fanout summary over gates sampled every `stride` positions: for each sampled gate, the
/// number of later gates using its active wire as a control before that wire is overwritten.
/// Returns the summary plus the fraction of sampled gates with zero fanout.
pub fn fanout_stats(gates: &[[u16; 3]], stride: usize) -> (StatSummary, f64) {
    let stride = stride.max(1);
    let mut values = Vec::with_capacity(gates.len() / stride + 1);
    let mut zero = 0usize;
    let mut i = 0;
    while i < gates.len() {
        let wire = gates[i][0];
        let mut fanout = 0usize;
        for gate in &gates[i + 1..] {
            if gate[0] == wire {
                break;
            }
            if gate[1] == wire || gate[2] == wire {
                fanout += 1;
            }
        }
        zero += usize::from(fanout == 0);
        values.push(fanout);
        i += stride;
    }
    let zero_frac = if values.is_empty() {
        0.0
    } else {
        zero as f64 / values.len() as f64
    };
    (summarize(values), zero_frac)
}

#[cfg(test)]
mod tests {
    use super::{disperse_random_topo, fanout_stats, leeway_stats};
    use crate::circuit::circuit::CircuitSeq;
    use crate::random::random_data::random_circuit;

    // Dense colliding blocks on disjoint wire triples: interior gates are fully pinned
    // (median leeway 0) but every block commutes with every other block.
    fn pinned_block_circuit(blocks: usize, gates_per_block: usize) -> Vec<[u16; 3]> {
        let mut gates = Vec::new();
        for b in 0..blocks {
            let w = (3 * b) as u16;
            for g in 0..gates_per_block {
                if g % 2 == 0 {
                    gates.push([w, w + 1, w + 2]);
                } else {
                    gates.push([w + 1, w, w + 2]);
                }
            }
        }
        gates
    }

    #[test]
    fn disperse_preserves_semantics_on_random_circuits() {
        let n = 8;
        let original = random_circuit(n, 300);
        let mut dispersed = original.gates.clone();
        disperse_random_topo(&mut dispersed, None, 64, 42);

        let dispersed = CircuitSeq { gates: dispersed };
        original
            .probably_equal(&dispersed, n, 2_000)
            .expect("disperse must preserve circuit function");
    }

    #[test]
    fn disperse_preserves_semantics_on_pinned_blocks() {
        let gates = pinned_block_circuit(10, 6);
        let original = CircuitSeq {
            gates: gates.clone(),
        };
        let mut dispersed = gates;
        disperse_random_topo(&mut dispersed, None, 1_000_000, 7);

        let dispersed = CircuitSeq { gates: dispersed };
        original
            .probably_equal(&dispersed, 30, 2_000)
            .expect("disperse must preserve circuit function");
    }

    #[test]
    fn disperse_is_a_permutation_of_the_input_gates() {
        let original = random_circuit(12, 500).gates;
        let mut dispersed = original.clone();
        disperse_random_topo(&mut dispersed, None, 128, 9);

        let mut a = original;
        let mut b = dispersed.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "disperse must not create, drop, or alter gates");
    }

    #[test]
    fn disperse_keeps_tags_aligned_with_gates() {
        let original = random_circuit(10, 400).gates;
        let mut dispersed = original.clone();
        let mut tags: Vec<u32> = (0..original.len() as u32).collect();
        disperse_random_topo(&mut dispersed, Some(&mut tags), 100, 3);

        for (gate, &tag) in dispersed.iter().zip(&tags) {
            assert_eq!(
                *gate, original[tag as usize],
                "tag must still identify the gate it was attached to"
            );
        }
    }

    #[test]
    fn disperse_is_deterministic_for_a_seed() {
        let original = random_circuit(10, 400).gates;
        let mut a = original.clone();
        let mut b = original;
        disperse_random_topo(&mut a, None, 128, 1234);
        disperse_random_topo(&mut b, None, 128, 1234);

        assert_eq!(a, b);
    }

    #[test]
    fn disperse_raises_median_leeway_of_pinned_blocks() {
        let mut gates = pinned_block_circuit(20, 4);
        let before = leeway_stats(&gates, 1);
        disperse_random_topo(&mut gates, None, 1_000_000, 11);
        let after = leeway_stats(&gates, 1);

        assert_eq!(before.median, 0, "pinned fixture should start fully pinned");
        assert!(
            after.median > before.median,
            "dispersal should raise median leeway (got {} -> {})",
            before.median,
            after.median
        );
    }

    #[test]
    fn cooldown_tournament_spaces_at_least_as_well_as_uniform() {
        let gates = pinned_block_circuit(20, 4);

        let mut uniform = gates.clone();
        super::disperse_random_topo_with(&mut uniform, None, 1_000_000, 11, 1);
        let mut cooled = gates.clone();
        super::disperse_random_topo_with(&mut cooled, None, 1_000_000, 11, 8);

        let uni = leeway_stats(&uniform, 1);
        let cool = leeway_stats(&cooled, 1);
        assert!(cool.median > 0, "cooldown must unpin the blocks");
        assert!(
            cool.median >= uni.median,
            "cooldown median {} should not be below uniform median {}",
            cool.median,
            uni.median
        );
        assert!(
            cool.avg >= uni.avg,
            "cooldown avg {:.2} should not be below uniform avg {:.2}",
            cool.avg,
            uni.avg
        );

        // Cooldown must remain a valid, semantics-preserving reorder.
        let original = CircuitSeq {
            gates: gates.clone(),
        };
        let cooled = CircuitSeq { gates: cooled };
        original
            .probably_equal(&cooled, 60, 2_000)
            .expect("cooldown disperse must preserve circuit function");
    }

    #[test]
    fn wire_usage_entropy_separates_uniform_from_concentrated() {
        let concentrated = pinned_block_circuit(2, 20);
        let spread = random_circuit(64, 400).gates;

        let (conc_active, conc_control) = super::wire_usage_entropy(&concentrated, 64);
        let (rand_active, rand_control) = super::wire_usage_entropy(&spread, 64);

        assert!((0.0..=1.0).contains(&conc_active));
        assert!((0.0..=1.0).contains(&conc_control));
        assert!(
            rand_active > 0.9 && rand_control > 0.9,
            "random circuits should be near-uniform (got {:.3}/{:.3})",
            rand_active,
            rand_control
        );
        assert!(
            conc_active < rand_active && conc_control < rand_control,
            "few-wire circuits must read as concentrated ({:.3}/{:.3} vs {:.3}/{:.3})",
            conc_active,
            conc_control,
            rand_active,
            rand_control
        );

        let (empty_active, empty_control) = super::wire_usage_entropy(&[], 8);
        assert_eq!(empty_active, 0.0);
        assert_eq!(empty_control, 0.0);
    }

    #[test]
    fn stats_helpers_handle_degenerate_inputs() {
        let empty = leeway_stats(&[], 1);
        assert_eq!(empty.max, 0);
        let (fan, zero) = fanout_stats(&[], 1);
        assert_eq!(fan.max, 0);
        assert_eq!(zero, 0.0);

        let single = [[0u16, 1, 2]];
        assert_eq!(leeway_stats(&single, 1).median, 0);
        let (fan, zero) = fanout_stats(&single, 1);
        assert_eq!(fan.max, 0);
        assert_eq!(zero, 1.0);
    }
}
