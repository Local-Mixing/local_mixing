use rand::{Rng, SeedableRng, rngs::StdRng, seq::IndexedRandom};
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    process::Command,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatHardnessScore {
    pub score: f64,
    pub nonlinear_depth: usize,
    pub dependency_density: f64,
    pub active_balance: f64,
    pub cone_growth: f64,
    pub output_cone_gates: usize,
    pub output_cone_fraction: f64,
    pub bcp_resistance: f64,
    pub slice_prop_resistance: f64,
    pub sat_probe_score: f64,
    pub repeated_template_penalty: f64,
    pub unit_prop_resistance: f64,
    pub gates: usize,
    pub wires: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputConeStats {
    pub output_start: usize,
    pub output_bits: usize,
    pub cone_gates: usize,
    pub cone_gate_fraction: f64,
    pub cone_wires: usize,
    pub cone_input_wires: usize,
    pub cone_max_live_wires: usize,
}

pub fn output_cone_stats(
    gates: &[[u16; 3]],
    output_start: usize,
    output_bits: usize,
) -> Option<OutputConeStats> {
    if output_bits == 0 {
        return None;
    }

    let output_end = output_start.checked_add(output_bits)?;
    let max_gate_wire = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|wire| wire as usize + 1)
        .unwrap_or(0);
    let wire_count = max_gate_wire.max(output_end);
    let mut live = vec![false; wire_count];
    let mut cone_wires = vec![false; wire_count];
    let mut live_count = 0usize;
    let mut cone_wire_count = 0usize;

    for wire in output_start..output_end {
        mark_wire(&mut live, wire, &mut live_count);
        mark_wire(&mut cone_wires, wire, &mut cone_wire_count);
    }

    let mut cone_gates = 0usize;
    let mut cone_max_live_wires = live_count;
    for gate in gates.iter().rev() {
        let [active, control_a, control_b] = gate_indices(*gate);
        if active >= wire_count || !live[active] {
            continue;
        }
        cone_gates += 1;
        for wire in [active, control_a, control_b] {
            mark_wire(&mut live, wire, &mut live_count);
            mark_wire(&mut cone_wires, wire, &mut cone_wire_count);
        }
        cone_max_live_wires = cone_max_live_wires.max(live_count);
    }

    Some(OutputConeStats {
        output_start,
        output_bits,
        cone_gates,
        cone_gate_fraction: if gates.is_empty() {
            0.0
        } else {
            cone_gates as f64 / gates.len() as f64
        },
        cone_wires: cone_wire_count,
        cone_input_wires: live_count,
        cone_max_live_wires,
    })
}

pub fn output_cone_stats_for_wires(
    gates: &[[u16; 3]],
    output_wires: &[usize],
) -> Option<OutputConeStats> {
    if output_wires.is_empty() {
        return None;
    }

    let max_output = output_wires.iter().copied().max()?.saturating_add(1);
    let max_gate_wire = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|wire| wire as usize + 1)
        .unwrap_or(0);
    let wire_count = max_gate_wire.max(max_output);
    let mut live = vec![false; wire_count];
    let mut cone_wires = vec![false; wire_count];
    let mut live_count = 0usize;
    let mut cone_wire_count = 0usize;

    for &wire in output_wires {
        mark_wire(&mut live, wire, &mut live_count);
        mark_wire(&mut cone_wires, wire, &mut cone_wire_count);
    }

    let mut cone_gates = 0usize;
    let mut cone_max_live_wires = live_count;
    for gate in gates.iter().rev() {
        let [active, control_a, control_b] = gate_indices(*gate);
        if active >= wire_count || !live[active] {
            continue;
        }
        cone_gates += 1;
        for wire in [active, control_a, control_b] {
            mark_wire(&mut live, wire, &mut live_count);
            mark_wire(&mut cone_wires, wire, &mut cone_wire_count);
        }
        cone_max_live_wires = cone_max_live_wires.max(live_count);
    }

    Some(OutputConeStats {
        output_start: output_wires.iter().copied().min().unwrap_or(0),
        output_bits: output_wires.len(),
        cone_gates,
        cone_gate_fraction: if gates.is_empty() {
            0.0
        } else {
            cone_gates as f64 / gates.len() as f64
        },
        cone_wires: cone_wire_count,
        cone_input_wires: live_count,
        cone_max_live_wires,
    })
}

pub fn sat_scoring_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_truthy("SAT_SCORE") || env_truthy("SAT_HARDEN") || sat_cone_aware_enabled()
    })
}

pub fn sat_cone_aware_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_CONE_AWARE") || env_truthy("SAT_HARDEN"))
}

pub fn sat_cone_min_fraction() -> f64 {
    static MIN: OnceLock<f64> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("SAT_CONE_MIN_FRACTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

pub fn sat_hidden_samf_candidates() -> usize {
    static CANDIDATES: OnceLock<usize> = OnceLock::new();
    *CANDIDATES.get_or_init(|| {
        std::env::var("SAT_HIDDEN_SAMF_CANDIDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| if sat_cone_aware_enabled() { 16 } else { 1 })
            .max(1)
    })
}

pub fn sat_expand_loop_candidates() -> usize {
    static CANDIDATES: OnceLock<usize> = OnceLock::new();
    *CANDIDATES.get_or_init(|| {
        std::env::var("SAT_EXPAND_LOOP_CANDIDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                if sat_cone_aware_enabled() {
                    sat_hidden_samf_candidates()
                } else if sat_scoring_enabled() {
                    4
                } else {
                    1
                }
            })
            .max(1)
    })
}

pub fn sat_bcp_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_BCP") || env_truthy("SAT_HARDEN"))
}

pub fn sat_bcp_min_resistance() -> f64 {
    static MIN: OnceLock<f64> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("SAT_BCP_MIN_RESISTANCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

pub fn sat_slice_prop_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_SLICE_PROP"))
}

// Optional exact fixed-wire window (same convention as SAT_CONE_START/SAT_CONE_BITS).
// When unset, each trial fixes a random SAT_SLICE_FIXED_FRACTION subset of wires instead.
fn sat_slice_fixed_range() -> Option<(usize, usize)> {
    static RANGE: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let start = std::env::var("SAT_SLICE_FIXED_START")
            .ok()?
            .parse()
            .ok()?;
        let bits = std::env::var("SAT_SLICE_FIXED_BITS").ok()?.parse().ok()?;
        Some((start, bits))
    })
}

pub fn sat_compress_protect_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_COMPRESS_PROTECT") || env_truthy("SAT_HARDEN"))
}

pub fn sat_compress_preserve_delta() -> f64 {
    static DELTA: OnceLock<f64> = OnceLock::new();
    *DELTA.get_or_init(|| {
        std::env::var("SAT_COMPRESS_PRESERVE_DELTA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

pub fn sat_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("SAT_PROBE"))
}

pub fn sat_probe_frequency() -> usize {
    static FREQUENCY: OnceLock<usize> = OnceLock::new();
    *FREQUENCY.get_or_init(|| {
        std::env::var("SAT_PROBE_FREQUENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .max(1)
    })
}

pub fn sat_probe_window_gates() -> usize {
    static WINDOW: OnceLock<usize> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("SAT_PROBE_WINDOW_GATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20_000)
    })
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

pub fn sat_expand_min_delta() -> f64 {
    static MIN_DELTA: OnceLock<f64> = OnceLock::new();
    *MIN_DELTA.get_or_init(|| {
        std::env::var("SAT_EXPAND_MIN_DELTA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
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
            output_cone_gates: 0,
            output_cone_fraction: 0.0,
            bcp_resistance: 0.0,
            slice_prop_resistance: 0.0,
            sat_probe_score: 0.0,
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
    let output_cone = output_cone_stats_for_wires(gates, &output_wires);
    let output_cone_gates = output_cone.map(|stats| stats.cone_gates).unwrap_or(0);
    let output_cone_fraction = output_cone
        .map(|stats| stats.cone_gate_fraction)
        .unwrap_or(0.0);
    let active_balance =
        0.75 * normalized_entropy(&active_counts) + 0.25 * normalized_entropy(&control_counts);
    let cone_growth = (total_growth as f64 / gate_count as f64 / wire_count as f64).min(1.0);
    let repeated_template_penalty = repeated_template_penalty(&exact_templates, gate_count);
    let unit_prop_resistance = unit_prop_resistance(gates, wire_count, &output_wires, seed);
    let bcp_resistance = if sat_bcp_enabled() {
        bcp_resistance(gates, wire_count, &output_wires, seed)
    } else {
        0.0
    };
    let slice_prop_resistance = if sat_slice_prop_enabled() {
        slice_prop_resistance(gates, wire_count, &output_wires, seed)
    } else {
        0.0
    };
    let sat_probe_score = if sat_probe_enabled() {
        sat_probe_score(gates, wire_count, &output_wires, seed)
    } else {
        0.0
    };

    let depth_norm = (nonlinear_depth as f64 / gate_count.max(1) as f64).min(1.0);
    let wire_norm = (distinct_wires as f64 / wire_count as f64).min(1.0);
    let gate_norm = (gate_count as f64).ln_1p() / 8.0;
    let score = 3.0 * dependency_density
        + 2.0 * depth_norm
        + 1.5 * active_balance
        + 1.5 * cone_growth
        + 2.0 * unit_prop_resistance
        + 1.5 * output_cone_fraction
        + 0.25 * ((output_cone_gates as f64).ln_1p() / 8.0)
        + 2.5 * bcp_resistance
        + 2.5 * slice_prop_resistance
        + 3.0 * sat_probe_score
        + 0.5 * wire_norm
        + 0.25 * gate_norm
        - 2.0 * repeated_template_penalty;

    SatHardnessScore {
        score,
        nonlinear_depth,
        dependency_density,
        active_balance,
        cone_growth,
        output_cone_gates,
        output_cone_fraction,
        bcp_resistance,
        slice_prop_resistance,
        sat_probe_score,
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
        + 2.0 * score.output_cone_fraction
        + 0.05 * score.output_cone_gates as f64
        + 4.0 * score.bcp_resistance
        + 4.0 * score.slice_prop_resistance
        + 6.0 * score.sat_probe_score
        + 0.3 * score.wires as f64
        + 0.1 * score.gates as f64
        - 2.0 * score.repeated_template_penalty
}

pub fn compression_selection_score(score: &SatHardnessScore) -> f64 {
    score.score
        + 0.5 * score.nonlinear_depth as f64
        + score.dependency_density
        + score.unit_prop_resistance
        + 2.0 * score.output_cone_fraction
        + 0.05 * score.output_cone_gates as f64
        + 4.0 * score.bcp_resistance
        + 4.0 * score.slice_prop_resistance
        + 6.0 * score.sat_probe_score
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

fn mark_wire(bits: &mut [bool], wire: usize, count: &mut usize) {
    if wire < bits.len() && !bits[wire] {
        bits[wire] = true;
        *count += 1;
    }
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

struct LocalCnf {
    clauses: Vec<Vec<i32>>,
    var_count: usize,
    input_vars: Vec<i32>,
    output_vars: Vec<i32>,
    gate_vars: Vec<i32>,
}

fn local_cnf(gates: &[[u16; 3]], wire_count: usize) -> LocalCnf {
    let mut clauses = Vec::with_capacity(gates.len() * 6);
    let mut current: Vec<i32> = (1..=wire_count as i32).collect();
    let input_vars = current.clone();
    let mut gate_vars = Vec::with_capacity(gates.len());
    let mut next_var = wire_count as i32 + 1;

    for &gate in gates {
        let [active, control_a, control_b] = gate_indices(gate);
        if active >= wire_count || control_a >= wire_count || control_b >= wire_count {
            continue;
        }
        let out = next_var;
        next_var += 1;
        gate_vars.push(out);
        clauses.extend(gate_clauses(
            current[active],
            current[control_a],
            current[control_b],
            out,
        ));
        current[active] = out;
    }

    LocalCnf {
        clauses,
        var_count: next_var as usize - 1,
        input_vars,
        output_vars: current,
        gate_vars,
    }
}

fn gate_clauses(active: i32, control_a: i32, control_b: i32, out: i32) -> [Vec<i32>; 6] {
    [
        vec![-control_a, active, out],
        vec![-control_a, -active, -out],
        vec![control_b, active, out],
        vec![control_b, -active, -out],
        vec![control_a, -control_b, -active, out],
        vec![control_a, -control_b, active, -out],
    ]
}

fn bcp_resistance(gates: &[[u16; 3]], wire_count: usize, output_wires: &[usize], seed: u64) -> f64 {
    if gates.is_empty() || wire_count == 0 || output_wires.is_empty() {
        return 0.0;
    }

    let cnf = local_cnf(gates, wire_count);
    if cnf.gate_vars.is_empty() {
        return 0.0;
    }

    let trials = env_usize("SAT_BCP_TRIALS", 8).max(1);
    let assign_prob = env_f64("SAT_BCP_ASSIGN_PROB", 0.35).clamp(0.0, 1.0);
    let output_bits = env_usize("SAT_BCP_OUTPUT_BITS", output_wires.len())
        .max(1)
        .min(output_wires.len());
    let mut rng = StdRng::seed_from_u64(seed ^ 0xbc90_bc90_bc90_bc90);
    let mut total_unassigned = 0.0;
    let mut conflicts = 0usize;

    for _ in 0..trials {
        let mut assumptions = Vec::new();
        for &var in &cnf.input_vars {
            if rng.random_bool(assign_prob) {
                assumptions.push(if rng.random() { var } else { -var });
            }
        }
        for &wire in output_wires.choose_multiple(&mut rng, output_bits) {
            if wire < cnf.output_vars.len() {
                let var = cnf.output_vars[wire];
                assumptions.push(if rng.random() { var } else { -var });
            }
        }

        match unit_propagate(&cnf.clauses, cnf.var_count, &assumptions) {
            Some(assignments) => {
                let unassigned = cnf
                    .gate_vars
                    .iter()
                    .filter(|&&var| assignments[var as usize].is_none())
                    .count();
                total_unassigned += unassigned as f64 / cnf.gate_vars.len() as f64;
            }
            None => conflicts += 1,
        }
    }

    let conflict_penalty = 1.0 - conflicts as f64 / trials as f64;
    (total_unassigned / trials as f64 * conflict_penalty).clamp(0.0, 1.0)
}

// Boundary-conditioned BCP resistance. Unlike `bcp_resistance` (random sparse assumptions
// over all inputs), this models the fixed-slice benchmark: a structured set of whole wires
// is fixed to constants (the y,z slice in the fixed-y/z experiments), output bits are
// constrained, and we measure how much of the circuit unit propagation alone can determine.
// Fixed values are re-randomized per trial so the score reflects robustness across targets,
// not one lucky assignment. 1.0 = BCP determines nothing; 0.0 = fully propagated or refuted.
fn slice_prop_resistance(
    gates: &[[u16; 3]],
    wire_count: usize,
    output_wires: &[usize],
    seed: u64,
) -> f64 {
    let trials = env_usize("SAT_SLICE_PROP_TRIALS", 8).max(1);
    let fixed_fraction = env_f64("SAT_SLICE_FIXED_FRACTION", 2.0 / 3.0).clamp(0.0, 1.0);
    let output_bits = env_usize("SAT_SLICE_OUTPUT_BITS", output_wires.len())
        .max(1)
        .min(output_wires.len().max(1));
    slice_prop_resistance_with(
        gates,
        wire_count,
        output_wires,
        seed,
        trials,
        fixed_fraction,
        sat_slice_fixed_range(),
        output_bits,
    )
}

fn slice_prop_resistance_with(
    gates: &[[u16; 3]],
    wire_count: usize,
    output_wires: &[usize],
    seed: u64,
    trials: usize,
    fixed_fraction: f64,
    fixed_range: Option<(usize, usize)>,
    output_bits: usize,
) -> f64 {
    if gates.is_empty() || wire_count == 0 || output_wires.is_empty() {
        return 0.0;
    }

    let cnf = local_cnf(gates, wire_count);
    if cnf.gate_vars.is_empty() {
        return 0.0;
    }

    let trials = trials.max(1);
    let output_bits = output_bits.min(output_wires.len());
    let all_wires: Vec<usize> = (0..wire_count).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x511c_e0f5_a7b0_0d5e);
    let mut total_unassigned = 0.0;
    let mut conflicts = 0usize;

    for _ in 0..trials {
        let fixed_wires: Vec<usize> = match fixed_range {
            Some((start, bits)) => {
                let start = start.min(wire_count);
                let end = start.saturating_add(bits).min(wire_count);
                (start..end).collect()
            }
            None => {
                let k = ((wire_count as f64 * fixed_fraction).round() as usize).min(wire_count);
                all_wires.choose_multiple(&mut rng, k).copied().collect()
            }
        };

        let mut assumptions = Vec::with_capacity(fixed_wires.len() + output_bits);
        for &wire in &fixed_wires {
            let var = cnf.input_vars[wire];
            assumptions.push(if rng.random() { var } else { -var });
        }
        for &wire in output_wires.choose_multiple(&mut rng, output_bits) {
            if wire < cnf.output_vars.len() {
                let var = cnf.output_vars[wire];
                assumptions.push(if rng.random() { var } else { -var });
            }
        }

        match unit_propagate(&cnf.clauses, cnf.var_count, &assumptions) {
            Some(assignments) => {
                let unassigned = cnf
                    .gate_vars
                    .iter()
                    .filter(|&&var| assignments[var as usize].is_none())
                    .count();
                total_unassigned += unassigned as f64 / cnf.gate_vars.len() as f64;
            }
            None => conflicts += 1,
        }
    }

    let conflict_penalty = 1.0 - conflicts as f64 / trials as f64;
    (total_unassigned / trials as f64 * conflict_penalty).clamp(0.0, 1.0)
}

fn unit_propagate(
    clauses: &[Vec<i32>],
    var_count: usize,
    assumptions: &[i32],
) -> Option<Vec<Option<bool>>> {
    let mut assignments = vec![None; var_count + 1];
    for &lit in assumptions {
        if !assign_lit(&mut assignments, lit) {
            return None;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for clause in clauses {
            let mut satisfied = false;
            let mut unassigned = 0usize;
            let mut last_unassigned = 0i32;

            for &lit in clause {
                match lit_value(&assignments, lit) {
                    Some(true) => {
                        satisfied = true;
                        break;
                    }
                    Some(false) => {}
                    None => {
                        unassigned += 1;
                        last_unassigned = lit;
                    }
                }
            }

            if satisfied {
                continue;
            }
            if unassigned == 0 {
                return None;
            }
            if unassigned == 1 {
                if !assign_lit(&mut assignments, last_unassigned) {
                    return None;
                }
                changed = true;
            }
        }
    }

    Some(assignments)
}

fn assign_lit(assignments: &mut [Option<bool>], lit: i32) -> bool {
    let var = lit.unsigned_abs() as usize;
    let value = lit > 0;
    match assignments[var] {
        Some(existing) => existing == value,
        None => {
            assignments[var] = Some(value);
            true
        }
    }
}

fn lit_value(assignments: &[Option<bool>], lit: i32) -> Option<bool> {
    assignments[lit.unsigned_abs() as usize].map(|value| if lit > 0 { value } else { !value })
}

fn sat_probe_score(
    gates: &[[u16; 3]],
    wire_count: usize,
    output_wires: &[usize],
    seed: u64,
) -> f64 {
    static PROBE_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static PROBE_DISABLED: AtomicBool = AtomicBool::new(false);

    if PROBE_DISABLED.load(Ordering::Relaxed)
        || gates.is_empty()
        || gates.len() > sat_probe_window_gates()
        || output_wires.is_empty()
    {
        return 0.0;
    }

    let probe_index = PROBE_COUNTER.fetch_add(1, Ordering::Relaxed);
    if probe_index % sat_probe_frequency() != 0 {
        return 0.0;
    }

    let cnf = local_cnf(gates, wire_count);
    if cnf.gate_vars.is_empty() {
        return 0.0;
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0x51a7_900b_e551_9eed);
    let mut clauses = cnf.clauses.clone();
    let output_bits = env_usize("SAT_PROBE_TARGET_BITS", output_wires.len())
        .max(1)
        .min(output_wires.len());
    for &wire in output_wires.choose_multiple(&mut rng, output_bits) {
        if wire < cnf.output_vars.len() {
            let var = cnf.output_vars[wire];
            clauses.push(vec![if rng.random() { var } else { -var }]);
        }
    }

    let path = std::env::temp_dir().join(format!(
        "local_mixing_sat_probe_{}_{}.cnf",
        std::process::id(),
        probe_index
    ));
    if write_dimacs(&path, cnf.var_count, &clauses).is_err() {
        return 0.0;
    }

    let solver = std::env::var("SAT_PROBE_SOLVER").unwrap_or_else(|_| "kissat".to_string());
    let timeout_ms = env_usize("SAT_PROBE_TIMEOUT_MS", 250);
    let output = if timeout_ms > 0 {
        Command::new("timeout")
            .arg(format!("{:.3}s", timeout_ms as f64 / 1000.0))
            .arg(&solver)
            .arg(&path)
            .output()
    } else {
        Command::new(&solver).arg(&path).output()
    };
    let _ = std::fs::remove_file(&path);

    let Ok(output) = output else {
        PROBE_DISABLED.store(true, Ordering::Relaxed);
        return 0.0;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}\n{}", text, stderr);
    let conflicts = parse_solver_stat(&combined, "conflicts").unwrap_or(0.0);
    let decisions = parse_solver_stat(&combined, "decisions").unwrap_or(0.0);
    let propagations = parse_solver_stat(&combined, "propagations").unwrap_or(0.0);
    let timed_out = output.status.code() == Some(124);
    let raw =
        conflicts + 0.25 * decisions + 0.001 * propagations + if timed_out { 5000.0 } else { 0.0 };
    (raw.ln_1p() / 10.0).clamp(0.0, 2.0)
}

fn write_dimacs(
    path: &std::path::Path,
    var_count: usize,
    clauses: &[Vec<i32>],
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "p cnf {} {}", var_count, clauses.len())?;
    for clause in clauses {
        for lit in clause {
            write!(file, "{} ", lit)?;
        }
        writeln!(file, "0")?;
    }
    Ok(())
}

fn parse_solver_stat(output: &str, label: &str) -> Option<f64> {
    output.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        if !lower.contains(label) {
            return None;
        }
        lower
            .split(|c: char| !(c.is_ascii_digit() || c == '.'))
            .find(|part| !part.is_empty())
            .and_then(|part| part.parse().ok())
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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
    use super::{output_cone_stats, output_cone_stats_for_wires, score_subcircuit};

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

    #[test]
    fn output_cone_counts_only_relevant_active_writes() {
        let gates = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];
        let stats = output_cone_stats(&gates, 7, 1).unwrap();

        assert_eq!(stats.cone_gates, 4);
        assert_eq!(stats.cone_wires, 8);
        assert_eq!(stats.cone_input_wires, 8);

        let unrelated = output_cone_stats(&gates, 8, 1).unwrap();
        assert_eq!(unrelated.cone_gates, 0);
        assert_eq!(unrelated.cone_wires, 1);
    }

    #[test]
    fn output_cone_counts_noncontiguous_output_wires() {
        let gates = [[0, 1, 2], [3, 4, 5], [6, 0, 3]];
        let stats = output_cone_stats_for_wires(&gates, &[0, 3]).unwrap();

        assert_eq!(stats.output_bits, 2);
        assert_eq!(stats.cone_gates, 2);
        assert_eq!(stats.cone_wires, 6);
    }

    #[test]
    fn bcp_resistance_stays_in_unit_interval() {
        let gates = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];
        let score = super::bcp_resistance(&gates, 8, &[7], 123);

        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn unit_propagation_detects_simple_contradiction() {
        let clauses = vec![vec![1], vec![-1]];
        let result = super::unit_propagate(&clauses, 1, &[]);

        assert!(result.is_none());
    }

    #[test]
    fn slice_prop_resistance_is_deterministic() {
        let gates = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0], [2, 7, 1]];
        let outputs = [0usize, 2, 3, 5, 7];
        let a =
            super::slice_prop_resistance_with(&gates, 8, &outputs, 42, 8, 2.0 / 3.0, None, 3);
        let b =
            super::slice_prop_resistance_with(&gates, 8, &outputs, 42, 8, 2.0 / 3.0, None, 3);

        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a));
    }

    #[test]
    fn slice_prop_resistance_handles_degenerate_inputs() {
        // Empty circuit and empty output set both score zero without panicking.
        assert_eq!(
            super::slice_prop_resistance_with(&[], 8, &[1], 1, 4, 0.5, None, 1),
            0.0
        );
        let gates = [[0, 1, 2]];
        assert_eq!(
            super::slice_prop_resistance_with(&gates, 3, &[], 1, 4, 0.5, None, 1),
            0.0
        );

        // A fixed window extending past the wire count is clamped, not a panic.
        let clamped =
            super::slice_prop_resistance_with(&gates, 3, &[0], 1, 4, 1.0, Some((2, 100)), 1);
        assert!((0.0..=1.0).contains(&clamped));

        // Zero trials is promoted to one trial rather than dividing by zero.
        let no_trials = super::slice_prop_resistance_with(&gates, 3, &[0], 1, 0, 0.5, None, 1);
        assert!((0.0..=1.0).contains(&no_trials));
    }

    #[test]
    fn slice_prop_resistance_fully_fixed_boundary_is_easy() {
        // Fixing every wire lets unit propagation determine every gate variable,
        // so resistance must be exactly zero regardless of the sampled values.
        let gates = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
        let score =
            super::slice_prop_resistance_with(&gates, 3, &[0, 1, 2], 7, 6, 1.0, Some((0, 3)), 1);

        assert_eq!(score, 0.0);
    }
}
