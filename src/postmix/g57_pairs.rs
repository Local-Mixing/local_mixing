//! Deterministic identity-pair seeding for native G57 circuits.
//!
//! A G57 gate is an involution, so two adjacent identical gates implement the
//! identity.  The generator in this module deliberately chooses all insertion
//! gaps against the frozen, pre-insertion circuit.  This makes the placement
//! labels meaningful and, because generation is round-major, makes a run with
//! `p` pairs per target wire an exact prefix of a run with any larger `p` at
//! the same seed.

use crate::circuit::{CircuitSeq, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt;
use std::ops::Range;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum G57PairRegion {
    FirstQuarter,
    MiddleQuarter,
    LastQuarter,
    Uniform,
}

impl G57PairRegion {
    pub const VALUES: [&'static str; 4] =
        ["first-quarter", "middle-quarter", "last-quarter", "uniform"];

    /// Half-open range of admissible frozen gap indices.  A circuit with `m`
    /// gates has `m + 1` gaps: gap 0 is before gate 0 and gap `m` is after the
    /// final gate.  The middle placement is the central quarter.
    pub fn gap_range(self, baseline_gates: usize) -> Range<usize> {
        let gap_count = baseline_gates.saturating_add(1);
        let nonempty = |start: usize, end: usize| {
            let start = start.min(gap_count.saturating_sub(1));
            let end = end.clamp(start.saturating_add(1), gap_count);
            start..end
        };
        match self {
            Self::FirstQuarter => nonempty(0, gap_count.div_ceil(4)),
            Self::MiddleQuarter => {
                // Central quarter [3/8, 5/8), rounded outward so tiny test
                // circuits still have at least one legal frozen gap.
                nonempty((3 * gap_count) / 8, (5 * gap_count).div_ceil(8))
            }
            Self::LastQuarter => nonempty((3 * gap_count) / 4, gap_count),
            Self::Uniform => 0..gap_count,
        }
    }
}

impl fmt::Display for G57PairRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FirstQuarter => "first-quarter",
            Self::MiddleQuarter => "middle-quarter",
            Self::LastQuarter => "last-quarter",
            Self::Uniform => "uniform",
        })
    }
}

impl FromStr for G57PairRegion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "first-quarter" | "first" => Ok(Self::FirstQuarter),
            "middle-quarter" | "middle" => Ok(Self::MiddleQuarter),
            "last-quarter" | "last" => Ok(Self::LastQuarter),
            "uniform" | "random" => Ok(Self::Uniform),
            other => Err(format!(
                "unknown G57 pair region {other:?}; expected one of {}",
                Self::VALUES.join(", ")
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G57PairSeedConfig {
    pub pairs_per_target_wire: usize,
    pub target_wires: usize,
    pub num_wires: usize,
    /// Controls are sampled only from `0..control_wire_limit`.  This is
    /// intentionally separate from `num_wires`: a 384-wire Feistalized
    /// circuit can seed every target while keeping both controls on the 128
    /// functional/slice wires.
    pub control_wire_limit: usize,
    pub region: G57PairRegion,
    pub seed: u64,
}

impl G57PairSeedConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.pairs_per_target_wire == 0 {
            return Err("pairs_per_target_wire must be positive".into());
        }
        if self.num_wires < 3 {
            return Err("native G57 pairs need at least three wires".into());
        }
        if self.num_wires > u16::MAX as usize + 1 {
            return Err(format!(
                "num_wires {} exceeds the u16 wire namespace",
                self.num_wires
            ));
        }
        if self.target_wires == 0 || self.target_wires > self.num_wires {
            return Err(format!(
                "target_wires must be in 1..={} (got {})",
                self.num_wires, self.target_wires
            ));
        }
        if self.control_wire_limit == 0 || self.control_wire_limit > self.num_wires {
            return Err(format!(
                "control_wire_limit must be in 1..={} (got {})",
                self.num_wires, self.control_wire_limit
            ));
        }
        for target in 0..self.target_wires {
            let legal_controls =
                self.control_wire_limit - usize::from(target < self.control_wire_limit);
            if legal_controls < 2 {
                return Err(format!(
                    "target {target} has only {legal_controls} legal controls in 0..{}; two distinct controls are required",
                    self.control_wire_limit
                ));
            }
        }
        Ok(())
    }

    pub fn pair_count(&self) -> Result<usize, String> {
        self.pairs_per_target_wire
            .checked_mul(self.target_wires)
            .ok_or_else(|| "G57 pair count overflows usize".to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G57PairSpec {
    /// Zero-based generation round.  All targets are emitted once per round,
    /// making smaller per-wire counts exact prefixes of larger counts.
    pub round: usize,
    pub target: u16,
    pub first_control: u16,
    pub second_control: u16,
    /// Gap in the frozen baseline circuit, not in the growing output.
    pub gap: usize,
}

impl G57PairSpec {
    pub fn gate(&self) -> [u16; 3] {
        [self.target, self.first_control, self.second_control]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G57PairPlan {
    pub baseline_gates: usize,
    pub gap_range: Range<usize>,
    pub config: G57PairSeedConfig,
    pub specs: Vec<G57PairSpec>,
}

impl G57PairPlan {
    pub fn generate(baseline_gates: usize, config: G57PairSeedConfig) -> Result<Self, String> {
        config.validate()?;
        let pair_count = config.pair_count()?;
        let gap_range = config.region.gap_range(baseline_gates);
        let mut specs = Vec::with_capacity(pair_count);

        // Keep this loop order stable: it is the nesting contract used by the
        // 50/100/200 factorial sweep.
        for round in 0..config.pairs_per_target_wire {
            for target in 0..config.target_wires {
                let target = target as u16;
                // Content and placement have independent per-pair streams.
                // Thus all four regions use byte-for-byte identical G57s even
                // though their `random_range` domains differ.
                let mut content_rng = StdRng::seed_from_u64(pair_seed(
                    config.seed,
                    round,
                    target as usize,
                    0x434f_4e54_454e_545f,
                ));
                let mut anchor_rng = StdRng::seed_from_u64(pair_seed(
                    config.seed,
                    round,
                    target as usize,
                    0x414e_4348_4f52_5f5f,
                ));
                let first_control =
                    draw_other_wire(&mut content_rng, config.control_wire_limit, target, None);
                let second_control = draw_other_wire(
                    &mut content_rng,
                    config.control_wire_limit,
                    target,
                    Some(first_control),
                );
                let gap = anchor_rng.random_range(gap_range.clone());
                specs.push(G57PairSpec {
                    round,
                    target,
                    first_control,
                    second_control,
                    gap,
                });
            }
        }

        Ok(Self {
            baseline_gates,
            gap_range,
            config,
            specs,
        })
    }

    pub fn inserted_gates(&self) -> usize {
        self.specs.len() * 2
    }
}

/// SplitMix64 finalizer used only for deterministic stream derivation.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn pair_seed(root: u64, round: usize, target: usize, domain: u64) -> u64 {
    mix64(
        root ^ domain
            ^ (round as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (target as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
    )
}

fn draw_other_wire(
    rng: &mut StdRng,
    num_wires: usize,
    target: u16,
    other_control: Option<u16>,
) -> u16 {
    loop {
        let wire = rng.random_range(0..num_wires) as u16;
        if wire != target && Some(wire) != other_control {
            return wire;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum G57ShotStop {
    Disabled,
    StepLimit,
    Collision,
    Boundary,
}

impl fmt::Display for G57ShotStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Disabled => "disabled",
            Self::StepLimit => "step-limit",
            Self::Collision => "collision",
            Self::Boundary => "boundary",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G57PairShotRecord {
    pub pair_index: usize,
    pub spec: G57PairSpec,
    pub initial_left_position: usize,
    pub initial_right_position: usize,
    pub final_left_position: usize,
    pub final_right_position: usize,
    pub left_distance: usize,
    pub right_distance: usize,
    pub left_stop: G57ShotStop,
    pub right_stop: G57ShotStop,
    pub adjacent_after_shoot: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct G57PairInsertReport {
    pub baseline_gates: usize,
    pub final_gates: usize,
    pub pairs: usize,
    pub inserted_gates: usize,
    pub target_wires: usize,
    pub control_wire_limit: usize,
    pub pairs_per_target_wire: usize,
    pub region: G57PairRegion,
    pub first_gap: usize,
    pub gap_end_exclusive: usize,
    pub seed: u64,
    pub shoot_steps_per_copy: usize,
    pub total_left_distance: usize,
    pub total_right_distance: usize,
    pub collision_stops: usize,
    pub boundary_stops: usize,
    pub adjacent_pairs_remaining: usize,
    pub records: Vec<G57PairShotRecord>,
}

impl G57PairInsertReport {
    /// Stable, machine-readable acceptance evidence.  One row is emitted per
    /// identity pair with both generated controls, frozen gap, and observed
    /// deterministic shot outcome.
    pub fn manifest_tsv(&self) -> String {
        let mut out = String::new();
        out.push_str("pair_index\tround\ttarget\tcontrol_1\tcontrol_2\tfrozen_gap\tinitial_left\tinitial_right\tfinal_left\tfinal_right\tleft_distance\tright_distance\tleft_stop\tright_stop\tadjacent_after\n");
        for record in &self.records {
            let spec = &record.spec;
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                record.pair_index,
                spec.round,
                spec.target,
                spec.first_control,
                spec.second_control,
                spec.gap,
                record.initial_left_position,
                record.initial_right_position,
                record.final_left_position,
                record.final_right_position,
                record.left_distance,
                record.right_distance,
                record.left_stop,
                record.right_stop,
                u8::from(record.adjacent_after_shoot),
            ));
        }
        out
    }
}

/// Insert the planned native G57 identity pairs into a legacy G57 circuit.
/// Every pair is adjacent at insertion time and all positions refer to the
/// frozen input gaps.  This is the clean pre-old-SSS/pre-fsplit entry point.
pub fn insert_g57_identity_pairs(
    input: &CircuitSeq,
    config: G57PairSeedConfig,
) -> Result<(CircuitSeq, G57PairInsertReport), String> {
    insert_g57_identity_pairs_impl(input, config, None, 0)
}

/// Insert adjacent pairs and then shoot the first copy left and the second
/// copy right through at most `shoot_steps_per_copy` adjacent gates (zero means
/// maximally, to the first collision/boundary).  Every
/// swap is guarded by `Gate::collides_index`; no rewrite axiom is involved,
/// so this remains an exact permutation of commuting gates.  A pair stays
/// adjacent only when shooting is disabled or both escape directions are
/// immediately blocked by a collision/boundary.
pub fn insert_g57_identity_pairs_and_shoot(
    input: &CircuitSeq,
    config: G57PairSeedConfig,
    shoot_steps_per_copy: usize,
) -> Result<(CircuitSeq, G57PairInsertReport), String> {
    let effective_limit = if shoot_steps_per_copy == 0 {
        usize::MAX
    } else {
        shoot_steps_per_copy
    };
    insert_g57_identity_pairs_impl(input, config, Some(effective_limit), shoot_steps_per_copy)
}

fn insert_g57_identity_pairs_impl(
    input: &CircuitSeq,
    config: G57PairSeedConfig,
    effective_shoot_limit: Option<usize>,
    requested_shoot_steps: usize,
) -> Result<(CircuitSeq, G57PairInsertReport), String> {
    let plan = G57PairPlan::generate(input.gates.len(), config)?;
    let mut by_gap: Vec<Vec<usize>> = vec![Vec::new(); input.gates.len() + 1];
    for (pair_index, spec) in plan.specs.iter().enumerate() {
        by_gap[spec.gap].push(pair_index);
    }

    let final_len = input
        .gates
        .len()
        .checked_add(plan.inserted_gates())
        .ok_or_else(|| "output G57 gate count overflows usize".to_string())?;
    let mut gates = Vec::with_capacity(final_len);
    let mut tags: Vec<Option<(usize, usize)>> = Vec::with_capacity(final_len);
    let mut positions = vec![[0usize; 2]; plan.specs.len()];
    for gap in 0..=input.gates.len() {
        for &pair_index in &by_gap[gap] {
            let spec = &plan.specs[pair_index];
            let gate = spec.gate();
            positions[pair_index][0] = gates.len();
            gates.push(gate);
            tags.push(Some((pair_index, 0)));
            positions[pair_index][1] = gates.len();
            gates.push(gate);
            tags.push(Some((pair_index, 1)));
        }
        if gap < input.gates.len() {
            gates.push(input.gates[gap]);
            tags.push(None);
        }
    }
    debug_assert_eq!(gates.len(), final_len);
    debug_assert_eq!(tags.len(), final_len);

    let initial_positions = positions.clone();
    let mut shot_outcomes = Vec::with_capacity(plan.specs.len());
    let mut total_left_distance = 0usize;
    let mut total_right_distance = 0usize;
    let mut collision_stops = 0usize;
    let mut boundary_stops = 0usize;
    for pair_index in 0..plan.specs.len() {
        let (left_distance, left_stop, right_distance, right_stop) =
            if let Some(limit) = effective_shoot_limit {
                let (left_distance, left_stop) = shoot_tagged_copy(
                    &mut gates,
                    &mut tags,
                    &mut positions,
                    pair_index,
                    0,
                    false,
                    limit,
                );
                let (right_distance, right_stop) = shoot_tagged_copy(
                    &mut gates,
                    &mut tags,
                    &mut positions,
                    pair_index,
                    1,
                    true,
                    limit,
                );
                (left_distance, left_stop, right_distance, right_stop)
            } else {
                (0, G57ShotStop::Disabled, 0, G57ShotStop::Disabled)
            };
        total_left_distance += left_distance;
        total_right_distance += right_distance;
        collision_stops += usize::from(left_stop == G57ShotStop::Collision)
            + usize::from(right_stop == G57ShotStop::Collision);
        boundary_stops += usize::from(left_stop == G57ShotStop::Boundary)
            + usize::from(right_stop == G57ShotStop::Boundary);
        shot_outcomes.push((left_distance, right_distance, left_stop, right_stop));
    }

    // Later pairs may commute across earlier tagged copies, so materialize
    // manifest positions only after every shot has completed.
    let mut adjacent_pairs_remaining = 0usize;
    let mut records = Vec::with_capacity(plan.specs.len());
    for pair_index in 0..plan.specs.len() {
        let (left_distance, right_distance, left_stop, right_stop) = shot_outcomes[pair_index];
        let [final_left_position, final_right_position] = positions[pair_index];
        let adjacent_after_shoot = final_left_position.abs_diff(final_right_position) == 1;
        adjacent_pairs_remaining += usize::from(adjacent_after_shoot);
        records.push(G57PairShotRecord {
            pair_index,
            spec: plan.specs[pair_index].clone(),
            initial_left_position: initial_positions[pair_index][0],
            initial_right_position: initial_positions[pair_index][1],
            final_left_position,
            final_right_position,
            left_distance,
            right_distance,
            left_stop,
            right_stop,
            adjacent_after_shoot,
        });
    }

    let report = G57PairInsertReport {
        baseline_gates: input.gates.len(),
        final_gates: gates.len(),
        pairs: plan.specs.len(),
        inserted_gates: plan.inserted_gates(),
        target_wires: plan.config.target_wires,
        control_wire_limit: plan.config.control_wire_limit,
        pairs_per_target_wire: plan.config.pairs_per_target_wire,
        region: plan.config.region,
        first_gap: plan.gap_range.start,
        gap_end_exclusive: plan.gap_range.end,
        seed: plan.config.seed,
        shoot_steps_per_copy: requested_shoot_steps,
        total_left_distance,
        total_right_distance,
        collision_stops,
        boundary_stops,
        adjacent_pairs_remaining,
        records,
    };
    Ok((CircuitSeq { gates }, report))
}

fn shoot_tagged_copy(
    gates: &mut [[u16; 3]],
    tags: &mut [Option<(usize, usize)>],
    positions: &mut [[usize; 2]],
    pair_index: usize,
    copy: usize,
    right: bool,
    max_steps: usize,
) -> (usize, G57ShotStop) {
    if max_steps == 0 {
        return (0, G57ShotStop::Disabled);
    }
    let mut distance = 0usize;
    while distance < max_steps {
        let here = positions[pair_index][copy];
        let neighbor = if right {
            if here + 1 == gates.len() {
                return (distance, G57ShotStop::Boundary);
            }
            here + 1
        } else {
            if here == 0 {
                return (distance, G57ShotStop::Boundary);
            }
            here - 1
        };
        if Gate::collides_index(&gates[here], &gates[neighbor]) {
            return (distance, G57ShotStop::Collision);
        }
        gates.swap(here, neighbor);
        tags.swap(here, neighbor);
        if let Some((other_pair, other_copy)) = tags[here] {
            positions[other_pair][other_copy] = here;
        }
        if let Some((other_pair, other_copy)) = tags[neighbor] {
            positions[other_pair][other_copy] = neighbor;
        }
        distance += 1;
    }
    (distance, G57ShotStop::StepLimit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::xgate::XGate;

    fn fixture() -> CircuitSeq {
        CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]],
        }
    }

    #[test]
    fn count_sweeps_are_exact_nested_generation_prefixes() {
        let make = |count| {
            G57PairPlan::generate(
                101,
                G57PairSeedConfig {
                    pairs_per_target_wire: count,
                    target_wires: 5,
                    num_wires: 9,
                    control_wire_limit: 9,
                    region: G57PairRegion::Uniform,
                    seed: 0x57_57_57,
                },
            )
            .unwrap()
        };
        let p50 = make(50);
        let p100 = make(100);
        let p200 = make(200);
        assert_eq!(p50.specs, p100.specs[..p50.specs.len()]);
        assert_eq!(p100.specs, p200.specs[..p100.specs.len()]);
        assert_eq!(p50.specs.len(), 250);
        assert_eq!(p100.specs.len(), 500);
        assert_eq!(p200.specs.len(), 1_000);
    }

    #[test]
    fn regions_only_use_their_frozen_baseline_gaps() {
        for region in [
            G57PairRegion::FirstQuarter,
            G57PairRegion::MiddleQuarter,
            G57PairRegion::LastQuarter,
            G57PairRegion::Uniform,
        ] {
            let plan = G57PairPlan::generate(
                127,
                G57PairSeedConfig {
                    pairs_per_target_wire: 20,
                    target_wires: 6,
                    num_wires: 8,
                    control_wire_limit: 8,
                    region,
                    seed: 123,
                },
            )
            .unwrap();
            assert!(
                plan.specs
                    .iter()
                    .all(|spec| plan.gap_range.contains(&spec.gap))
            );
        }
    }

    #[test]
    fn pair_contents_are_identical_across_all_regions() {
        let mut plans = Vec::new();
        for region in [
            G57PairRegion::FirstQuarter,
            G57PairRegion::MiddleQuarter,
            G57PairRegion::LastQuarter,
            G57PairRegion::Uniform,
        ] {
            plans.push(
                G57PairPlan::generate(
                    1_001,
                    G57PairSeedConfig {
                        pairs_per_target_wire: 50,
                        target_wires: 32,
                        num_wires: 384,
                        control_wire_limit: 128,
                        region,
                        seed: 20260723,
                    },
                )
                .unwrap(),
            );
        }
        for other in &plans[1..] {
            for (reference, candidate) in plans[0].specs.iter().zip(&other.specs) {
                assert_eq!(reference.round, candidate.round);
                assert_eq!(reference.target, candidate.target);
                assert_eq!(reference.first_control, candidate.first_control);
                assert_eq!(reference.second_control, candidate.second_control);
                // Placement is intentionally the only region-dependent field.
            }
        }
    }

    #[test]
    fn functional_control_slice_supports_targets_outside_it() {
        let plan = G57PairPlan::generate(
            17,
            G57PairSeedConfig {
                pairs_per_target_wire: 2,
                target_wires: 384,
                num_wires: 384,
                control_wire_limit: 128,
                region: G57PairRegion::Uniform,
                seed: 7,
            },
        )
        .unwrap();
        assert_eq!(plan.specs.len(), 768);
        assert!(plan.specs.iter().all(|spec| {
            spec.first_control < 128
                && spec.second_control < 128
                && spec.first_control != spec.second_control
                && spec.target != spec.first_control
                && spec.target != spec.second_control
        }));
    }

    #[test]
    fn every_generated_gate_is_a_true_native_g57() {
        let plan = G57PairPlan::generate(
            7,
            G57PairSeedConfig {
                pairs_per_target_wire: 50,
                target_wires: 7,
                num_wires: 9,
                control_wire_limit: 9,
                region: G57PairRegion::MiddleQuarter,
                seed: 999,
            },
        )
        .unwrap();
        for spec in plan.specs {
            assert_ne!(spec.target, spec.first_control);
            assert_ne!(spec.target, spec.second_control);
            assert_ne!(spec.first_control, spec.second_control);
            let gate = XGate::from_g57(spec.gate());
            assert!(gate.comp);
            assert_eq!(gate.width(), 2);
        }
    }

    #[test]
    fn legacy_insertion_is_adjacent_and_exhaustively_equivalent() {
        let input = fixture();
        for region in [
            G57PairRegion::FirstQuarter,
            G57PairRegion::MiddleQuarter,
            G57PairRegion::LastQuarter,
            G57PairRegion::Uniform,
        ] {
            let config = G57PairSeedConfig {
                pairs_per_target_wire: 3,
                target_wires: 5,
                num_wires: 5,
                control_wire_limit: 5,
                region,
                seed: 42,
            };
            let plan = G57PairPlan::generate(input.gates.len(), config.clone()).unwrap();
            let (output, report) = insert_g57_identity_pairs(&input, config).unwrap();
            assert_eq!(report.pairs, 15);
            assert_eq!(report.inserted_gates, 30);
            assert_eq!(output.gates.len(), input.gates.len() + 30);

            // Reconstruct each frozen gap's expected gate stream, proving the
            // identical copies are adjacent before any shooting stage.
            let mut expected = Vec::new();
            for gap in 0..=input.gates.len() {
                for spec in plan.specs.iter().filter(|spec| spec.gap == gap) {
                    expected.extend([spec.gate(), spec.gate()]);
                }
                if gap < input.gates.len() {
                    expected.push(input.gates[gap]);
                }
            }
            assert_eq!(output.gates, expected);

            for state in 0..(1usize << 5) {
                assert_eq!(
                    input.evaluate(state),
                    output.evaluate(state),
                    "region={region} state={state}"
                );
            }
        }
    }

    #[test]
    fn deterministic_shooting_uses_only_commutations_and_writes_manifest() {
        let input = fixture();
        let config = G57PairSeedConfig {
            pairs_per_target_wire: 2,
            target_wires: 5,
            num_wires: 5,
            control_wire_limit: 5,
            region: G57PairRegion::Uniform,
            seed: 314159,
        };
        let (a, report_a) = insert_g57_identity_pairs_and_shoot(&input, config.clone(), 2).unwrap();
        let (b, report_b) = insert_g57_identity_pairs_and_shoot(&input, config, 2).unwrap();
        assert_eq!(a, b);
        assert_eq!(report_a, report_b);
        assert_eq!(report_a.records.len(), 10);
        assert_eq!(report_a.shoot_steps_per_copy, 2);
        assert!(
            report_a
                .records
                .iter()
                .all(|record| record.left_distance <= 2 && record.right_distance <= 2)
        );
        let manifest = report_a.manifest_tsv();
        assert!(manifest.starts_with("pair_index\tround\ttarget"));
        assert_eq!(manifest.lines().count(), report_a.pairs + 1);
        for state in 0..(1usize << 5) {
            assert_eq!(input.evaluate(state), a.evaluate(state));
        }
    }

    #[test]
    fn zero_shoot_limit_means_maximal_not_disabled() {
        let input = CircuitSeq {
            // These write wire 4 and do not collide with a seeded target/control
            // triple confined to wires 0..=2.
            gates: vec![[4, 3, 2], [4, 3, 2], [4, 3, 2], [4, 3, 2]],
        };
        let (_, report) = insert_g57_identity_pairs_and_shoot(
            &input,
            G57PairSeedConfig {
                pairs_per_target_wire: 1,
                target_wires: 1,
                num_wires: 5,
                control_wire_limit: 3,
                region: G57PairRegion::MiddleQuarter,
                seed: 1,
            },
            0,
        )
        .unwrap();
        assert_eq!(report.shoot_steps_per_copy, 0);
        assert!(report.total_left_distance + report.total_right_distance > 0);
        assert!(report.records.iter().all(|record| {
            record.left_stop != G57ShotStop::Disabled
                && record.right_stop != G57ShotStop::Disabled
                && record.left_stop != G57ShotStop::StepLimit
                && record.right_stop != G57ShotStop::StepLimit
        }));
    }

    #[test]
    fn invalid_control_domains_are_rejected_without_hanging() {
        for control_wire_limit in [1, 2] {
            let result = G57PairPlan::generate(
                1,
                G57PairSeedConfig {
                    pairs_per_target_wire: 1,
                    target_wires: 3,
                    num_wires: 3,
                    control_wire_limit,
                    region: G57PairRegion::Uniform,
                    seed: 0,
                },
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn tiny_baselines_still_have_nonempty_region_ranges() {
        for baseline in 0..4 {
            for region in [
                G57PairRegion::FirstQuarter,
                G57PairRegion::MiddleQuarter,
                G57PairRegion::LastQuarter,
                G57PairRegion::Uniform,
            ] {
                let range = region.gap_range(baseline);
                assert!(range.start < range.end);
                assert!(range.end <= baseline + 1);
            }
        }
    }
}
