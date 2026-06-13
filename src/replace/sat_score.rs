use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{collections::HashMap, sync::OnceLock};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatHardnessScore {
    pub score: f64,
    pub nonlinear_depth: usize,
    pub dependency_density: f64,
    pub active_balance: f64,
    pub cone_growth: f64,
    pub repeated_template_penalty: f64,
    pub unit_prop_resistance: f64,
    pub gates: usize,
    pub wires: usize,
}

pub fn sat_scoring_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_SCORE") || env_truthy("SAT_HARDEN"))
}

pub fn sat_score_slack() -> usize {
    static SLACK: OnceLock<usize> = OnceLock::new();
    *SLACK.get_or_init(|| {
        std::env::var("SAT_SCORE_SLACK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}

pub fn sat_score_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        std::env::var("SAT_SCORE_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0x5a7a_d00d_cafe_f00d)
    })
}

pub fn score_subcircuit(gates: &[[u16; 3]], n: usize, seed: u64) -> SatHardnessScore {
    let gate_count = gates.len();
    let wire_count = wire_count(gates, n);
    if gate_count == 0 || wire_count == 0 {
        return SatHardnessScore {
            score: 0.0,
            nonlinear_depth: 0,
            dependency_density: 0.0,
            active_balance: 0.0,
            cone_growth: 0.0,
            repeated_template_penalty: 0.0,
            unit_prop_resistance: 0.0,
            gates: gate_count,
            wires: 0,
        };
    }

    let limb_count = (wire_count + 63) / 64;
    let mut deps = vec![vec![0u64; limb_count]; wire_count];
    let mut depths = vec![0usize; wire_count];
    let mut active_counts = vec![0usize; wire_count];
    let mut control_counts = vec![0usize; wire_count];
    let mut touched = vec![false; wire_count];
    let mut active_touched = vec![false; wire_count];
    let mut exact_templates: HashMap<[u16; 3], usize> = HashMap::new();

    for wire in 0..wire_count {
        set_bit(&mut deps[wire], wire);
    }

    let mut total_growth = 0usize;
    for gate in gates {
        let [active, control_a, control_b] = gate_indices(*gate);
        if active >= wire_count || control_a >= wire_count || control_b >= wire_count {
            continue;
        }

        active_counts[active] += 1;
        control_counts[control_a] += 1;
        control_counts[control_b] += 1;
        touched[active] = true;
        touched[control_a] = true;
        touched[control_b] = true;
        active_touched[active] = true;
        *exact_templates.entry(*gate).or_insert(0) += 1;

        let before = bit_count(&deps[active]);
        let mut merged = deps[active].clone();
        or_into(&mut merged, &deps[control_a]);
        or_into(&mut merged, &deps[control_b]);
        let after = bit_count(&merged);
        total_growth += after.saturating_sub(before);

        let control_nonlinear = control_a != control_b
            && !deps[control_a].iter().all(|&x| x == 0)
            && !deps[control_b].iter().all(|&x| x == 0);
        let control_depth =
            depths[control_a].max(depths[control_b]) + usize::from(control_nonlinear);
        depths[active] = depths[active].max(control_depth);
        deps[active] = merged;
    }

    let distinct_wires = touched.iter().filter(|&&x| x).count();
    let output_wires: Vec<usize> = active_touched
        .iter()
        .enumerate()
        .filter_map(|(wire, &is_active)| is_active.then_some(wire))
        .collect();
    let dependency_density = dependency_density(&deps, &output_wires, wire_count);
    let nonlinear_depth = output_wires
        .iter()
        .map(|&wire| depths[wire])
        .max()
        .unwrap_or(0);
    let active_balance =
        0.75 * normalized_entropy(&active_counts) + 0.25 * normalized_entropy(&control_counts);
    let cone_growth = (total_growth as f64 / gate_count as f64 / wire_count as f64).min(1.0);
    let repeated_template_penalty = repeated_template_penalty(&exact_templates, gate_count);
    let unit_prop_resistance = unit_prop_resistance(gates, wire_count, &output_wires, seed);

    let depth_norm = (nonlinear_depth as f64 / gate_count.max(1) as f64).min(1.0);
    let wire_norm = (distinct_wires as f64 / wire_count as f64).min(1.0);
    let gate_norm = (gate_count as f64).ln_1p() / 8.0;
    let score = 3.0 * dependency_density
        + 2.0 * depth_norm
        + 1.5 * active_balance
        + 1.5 * cone_growth
        + 2.0 * unit_prop_resistance
        + 0.5 * wire_norm
        + 0.25 * gate_norm
        - 2.0 * repeated_template_penalty;

    SatHardnessScore {
        score,
        nonlinear_depth,
        dependency_density,
        active_balance,
        cone_growth,
        repeated_template_penalty,
        unit_prop_resistance,
        gates: gate_count,
        wires: distinct_wires,
    }
}

pub fn expansion_selection_score(score: &SatHardnessScore) -> f64 {
    4.0 * score.score
        + score.nonlinear_depth as f64
        + score.dependency_density
        + 0.3 * score.wires as f64
        + 0.1 * score.gates as f64
        - 2.0 * score.repeated_template_penalty
}

pub fn compression_selection_score(score: &SatHardnessScore) -> f64 {
    score.score
        + 0.5 * score.nonlinear_depth as f64
        + score.dependency_density
        + score.unit_prop_resistance
        - 2.0 * score.repeated_template_penalty
}

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

fn wire_count(gates: &[[u16; 3]], n: usize) -> usize {
    let max_wire = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|wire| wire as usize + 1)
        .unwrap_or(0);
    n.max(max_wire)
}

fn gate_indices(gate: [u16; 3]) -> [usize; 3] {
    [gate[0] as usize, gate[1] as usize, gate[2] as usize]
}

fn set_bit(bits: &mut [u64], bit: usize) {
    bits[bit / 64] |= 1u64 << (bit % 64);
}

fn or_into(dst: &mut [u64], src: &[u64]) {
    for (dst_limb, src_limb) in dst.iter_mut().zip(src) {
        *dst_limb |= *src_limb;
    }
}

fn bit_count(bits: &[u64]) -> usize {
    bits.iter().map(|bits| bits.count_ones() as usize).sum()
}

fn dependency_density(deps: &[Vec<u64>], output_wires: &[usize], wire_count: usize) -> f64 {
    if output_wires.is_empty() || wire_count == 0 {
        return 0.0;
    }
    let total_deps: usize = output_wires
        .iter()
        .map(|&wire| bit_count(&deps[wire]))
        .sum();
    (total_deps as f64 / (output_wires.len() * wire_count) as f64).min(1.0)
}

fn normalized_entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    let populated = counts.iter().filter(|&&count| count > 0).count();
    if total == 0 || populated <= 1 {
        return 0.0;
    }

    let total = total as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum();
    (entropy / (populated as f64).ln()).min(1.0)
}

fn repeated_template_penalty(exact_templates: &HashMap<[u16; 3], usize>, gate_count: usize) -> f64 {
    if gate_count == 0 {
        return 0.0;
    }
    let repeats: usize = exact_templates
        .values()
        .map(|&count| count.saturating_sub(1))
        .sum();
    (repeats as f64 / gate_count as f64).min(1.0)
}

#[derive(Clone, Copy)]
enum Ternary {
    Known(bool),
    Unknown,
}

fn unit_prop_resistance(
    gates: &[[u16; 3]],
    wire_count: usize,
    output_wires: &[usize],
    seed: u64,
) -> f64 {
    if gates.is_empty() || wire_count == 0 || output_wires.is_empty() {
        return 0.0;
    }

    const TRIALS: usize = 16;
    let mut rng = StdRng::seed_from_u64(seed ^ gates.len() as u64 ^ ((wire_count as u64) << 32));
    let mut total_unknown_fraction = 0.0;

    for _ in 0..TRIALS {
        let mut values = vec![Ternary::Unknown; wire_count];
        for value in &mut values {
            if rng.random_bool(0.55) {
                *value = Ternary::Known(rng.random());
            }
        }

        for gate in gates {
            let [active, control_a, control_b] = gate_indices(*gate);
            if active >= wire_count || control_a >= wire_count || control_b >= wire_count {
                continue;
            }

            let active_value = values[active];
            let control_value = gate_control_value(values[control_a], values[control_b]);
            values[active] = match (active_value, control_value) {
                (Ternary::Known(a), Ternary::Known(b)) => Ternary::Known(a ^ b),
                _ => Ternary::Unknown,
            };
        }

        let unknown_outputs = output_wires
            .iter()
            .filter(|&&wire| matches!(values[wire], Ternary::Unknown))
            .count();
        total_unknown_fraction += unknown_outputs as f64 / output_wires.len() as f64;
    }

    total_unknown_fraction / TRIALS as f64
}

fn gate_control_value(control_a: Ternary, control_b: Ternary) -> Ternary {
    match (control_a, control_b) {
        (Ternary::Known(true), _) | (_, Ternary::Known(false)) => Ternary::Known(true),
        (Ternary::Known(false), Ternary::Known(true)) => Ternary::Known(false),
        _ => Ternary::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::score_subcircuit;

    #[test]
    fn empty_circuit_scores_zero() {
        let score = score_subcircuit(&[], 4, 1);
        assert_eq!(score.score, 0.0);
        assert_eq!(score.gates, 0);
        assert_eq!(score.wires, 0);
    }

    #[test]
    fn mixed_circuit_scores_above_repeated_template() {
        let repeated = [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]];
        let mixed = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];

        let repeated_score = score_subcircuit(&repeated, 8, 7);
        let mixed_score = score_subcircuit(&mixed, 8, 7);

        assert!(mixed_score.score > repeated_score.score);
        assert!(mixed_score.nonlinear_depth > repeated_score.nonlinear_depth);
        assert!(repeated_score.repeated_template_penalty > mixed_score.repeated_template_penalty);
    }
}
