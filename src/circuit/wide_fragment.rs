//! Generic fragmentation of wide mixed-polarity conjunction gates.
//!
//! The product gadgetizer normally fragments while it is still constructing
//! semantic blocks and then applies a global commuting shuffle.  This module
//! provides the circuit-level transform used by the optional post-layout pass.
//! Running it after that shuffle keeps each dirty-helper double sweep
//! contiguous, giving the frozen short-window rewriter a coherent local macro.

use super::xgate::XGate;
use rand::Rng;
use rand::seq::SliceRandom;

/// Spelling used for the cap-two dirty ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FragmentStyle {
    /// Every rung is the exact plain conjunction.
    Exact,
    /// Keep the first rung exact, but spell the target reads and every deeper
    /// rung in the g57/CNOT vocabulary.  Their residual constants cancel in
    /// the double sweep; keeping rung zero exact is the load-bearing boundary.
    NativeDeep,
}

impl FragmentStyle {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "exact" | "1" => Some(Self::Exact),
            "native-deep" | "native_deep" | "2" => Some(Self::NativeDeep),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FragmentStats {
    pub input_gates: usize,
    pub output_gates: usize,
    pub fragmented_gates: usize,
    pub native_emissions: usize,
    pub exact_rung_emissions: usize,
    pub max_controls_before: usize,
    pub max_controls_after: usize,
}

/// Replace every gate wider than two controls by an exact cap-two dirty
/// ladder, choosing restored helpers from the existing wire set.
///
/// No helper is assumed clean.  For controls `x0..x(w-1)` and dirty helpers
/// `h0..h(w-3)`, one half-sweep is
///
/// ```text
/// target ^= h_last & x_last
/// h_last ^= h_prev & x...
/// ...
/// h0 ^= x0 & x1
/// ...
/// h_last ^= h_prev & x...
/// ```
///
/// and the half-sweep is repeated.  The unknown helper contributions cancel,
/// every helper is restored, and the target receives exactly the original
/// conjunction.  A complemented gate is `X(target)` plus that conjunction.
pub fn fragment_wide_post_shuffle(
    gates: &mut Vec<XGate>,
    num_wires: usize,
    style: FragmentStyle,
    rng: &mut impl Rng,
) -> Result<FragmentStats, String> {
    let mut stats = FragmentStats {
        input_gates: gates.len(),
        max_controls_before: gates.iter().map(XGate::width).max().unwrap_or(0),
        ..FragmentStats::default()
    };
    let mut output = Vec::with_capacity(gates.len());
    let mut unavailable = vec![false; num_wires];
    let mut available = Vec::with_capacity(num_wires);
    for gate in gates.iter() {
        if gate.width() <= 2 {
            output.push(gate.clone());
            continue;
        }
        let rung_count = gate.width() - 2;
        collect_available_wires(gate, num_wires, &mut unavailable, &mut available);
        if available.len() < rung_count {
            return Err(format!(
                "cannot fragment width-{} gate on {} wires: need {} dirty helpers, only {} are free",
                gate.width(),
                num_wires,
                rung_count,
                available.len()
            ));
        }
        available.shuffle(rng);
        let helpers = &available[..rung_count];

        if gate.comp {
            output.push(XGate::x_gate(gate.target));
        }
        let controls: Vec<(u16, bool)> = gate.ctrls.iter().copied().collect();
        let mut rungs: Vec<XGate> = Vec::with_capacity(rung_count);
        rungs.push(
            XGate::conj(helpers[0], [controls[0], controls[1]])
                .expect("helper is distinct from original controls"),
        );
        for index in 1..rung_count {
            rungs.push(
                XGate::conj(
                    helpers[index],
                    [(helpers[index - 1], true), controls[index + 1]],
                )
                .expect("dirty helpers are distinct"),
            );
        }
        let target_lits = [
            (helpers[rung_count - 1], true),
            controls[controls.len() - 1],
        ];

        for _ in 0..2 {
            emit_ladder_gate(
                gate.target,
                &target_lits,
                style == FragmentStyle::NativeDeep,
                rng,
                &mut output,
                &mut stats,
            );
            for index in (0..rungs.len()).rev() {
                let native = style == FragmentStyle::NativeDeep && index > 0;
                emit_ladder_gate(
                    rungs[index].target,
                    &rungs[index].ctrls,
                    native,
                    rng,
                    &mut output,
                    &mut stats,
                );
            }
            for index in 1..rungs.len() {
                let native = style == FragmentStyle::NativeDeep;
                emit_ladder_gate(
                    rungs[index].target,
                    &rungs[index].ctrls,
                    native,
                    rng,
                    &mut output,
                    &mut stats,
                );
            }
        }
        stats.fragmented_gates += 1;
        clear_unavailable_wires(gate, num_wires, &mut unavailable);
    }
    stats.output_gates = output.len();
    stats.max_controls_after = output.iter().map(XGate::width).max().unwrap_or(0);
    *gates = output;
    Ok(stats)
}

#[inline]
fn collect_available_wires(
    gate: &XGate,
    num_wires: usize,
    unavailable: &mut [bool],
    available: &mut Vec<u16>,
) {
    let target = gate.target as usize;
    if target < num_wires {
        unavailable[target] = true;
    }
    for &(wire, _) in &gate.ctrls {
        let wire = wire as usize;
        if wire < num_wires {
            unavailable[wire] = true;
        }
    }
    available.clear();
    available.extend(
        unavailable
            .iter()
            .enumerate()
            .filter_map(|(wire, &is_unavailable)| (!is_unavailable).then_some(wire as u16)),
    );
}

#[inline]
fn clear_unavailable_wires(gate: &XGate, num_wires: usize, unavailable: &mut [bool]) {
    let target = gate.target as usize;
    if target < num_wires {
        unavailable[target] = false;
    }
    for &(wire, _) in &gate.ctrls {
        let wire = wire as usize;
        if wire < num_wires {
            unavailable[wire] = false;
        }
    }
}

fn emit_ladder_gate(
    target: u16,
    lits: &[(u16, bool)],
    native: bool,
    rng: &mut impl Rng,
    out: &mut Vec<XGate>,
    stats: &mut FragmentStats,
) {
    debug_assert_eq!(lits.len(), 2);
    if native {
        emit_g57_form(target, lits, rng, out);
        stats.native_emissions += 1;
    } else {
        out.push(XGate::conj(target, lits.iter().copied()).expect("ladder literals are distinct"));
        stats.exact_rung_emissions += 1;
    }
}

/// Spell a one- or two-literal conjunction as g57/CNOT plus a possible
/// residual constant. The caller uses it only at positions where the dirty
/// double sweep cancels that residual; deliberately emitting an `X` here
/// would be exact gate-by-gate, but would spend the vocabulary/size gain this
/// mode is meant to test.
fn emit_g57_form(target: u16, lits: &[(u16, bool)], rng: &mut impl Rng, out: &mut Vec<XGate>) {
    match *lits {
        [(wire, true)] => out.push(XGate::cnot(target, wire)),
        [(wire, false)] => {
            out.push(XGate::cnot(target, wire));
        }
        [a, b] => {
            let ((xw, xp), (yw, yp)) = if rng.random_bool(0.5) { (a, b) } else { (b, a) };
            match (xp, yp) {
                (false, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                }
                (true, false) => {
                    out.push(XGate::from_g57([target, yw, xw]));
                }
                (true, true) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, yw));
                }
                (false, false) => {
                    out.push(XGate::from_g57([target, xw, yw]));
                    out.push(XGate::cnot(target, xw));
                }
            }
        }
        _ => unreachable!("native ladder spelling takes one or two literals"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::xgate::eval_lanes;
    use super::*;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn legacy_available_order(gate: &XGate, num_wires: usize, rng: &mut impl Rng) -> Vec<u16> {
        let mut unavailable = vec![gate.target as usize];
        unavailable.extend(gate.ctrls.iter().map(|&(wire, _)| wire as usize));
        let mut available: Vec<u16> = (0..num_wires)
            .filter(|wire| !unavailable.contains(wire))
            .map(|wire| wire as u16)
            .collect();
        available.shuffle(rng);
        available
    }

    #[test]
    fn post_fragment_restores_arbitrary_dirty_helpers_and_matches_wide_gate() {
        for style in [FragmentStyle::Exact, FragmentStyle::NativeDeep] {
            for width in 3..=8usize {
                for comp in [false, true] {
                    for polarity_seed in 0..16u64 {
                        let total = 2 * width + 3;
                        let target = 0u16;
                        let controls: Vec<(u16, bool)> = (0..width)
                            .map(|i| ((i + 1) as u16, (polarity_seed >> (i % 4)) & 1 != 0))
                            .collect();
                        let original = vec![XGate {
                            target,
                            comp,
                            ctrls: controls.iter().copied().collect(),
                        }];
                        let mut fragmented = original.clone();
                        let mut rng = StdRng::seed_from_u64(
                            0xF12A_6E17 ^ polarity_seed ^ ((width as u64) << 16),
                        );
                        let stats =
                            fragment_wide_post_shuffle(&mut fragmented, total, style, &mut rng)
                                .expect("ample dirty helpers");
                        assert_eq!(stats.fragmented_gates, 1);
                        assert!(fragmented.iter().all(|gate| gate.width() <= 2));

                        for sample in 0..8u64 {
                            let mut lanes: Vec<u64> = (0..total).map(|_| rng.next_u64()).collect();
                            // Include structured all-zero/all-one lanes among
                            // the arbitrary dirty helper samples.
                            lanes[0] ^= sample.wrapping_mul(!0u64 / 7);
                            let mut want = lanes.clone();
                            let mut got = lanes.clone();
                            eval_lanes(original.iter(), &mut want);
                            eval_lanes(fragmented.iter(), &mut got);
                            assert_eq!(
                                got, want,
                                "style={style:?} width={width} comp={comp} polarity={polarity_seed}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reused_membership_storage_preserves_legacy_helper_selection() {
        let gates = vec![
            XGate::conj(3, [(0, true), (2, false), (7, true), (11, false)]).unwrap(),
            XGate::conj(
                14,
                [(1, false), (5, true), (8, true), (13, false), (19, true)],
            )
            .unwrap(),
        ];
        let num_wires = 24;
        let seed = 0x6865_6c70_6572_7331;
        let mut legacy_rng = StdRng::seed_from_u64(seed);
        let legacy_orders: Vec<Vec<u16>> = gates
            .iter()
            .map(|gate| legacy_available_order(gate, num_wires, &mut legacy_rng))
            .collect();

        let mut unavailable = vec![false; num_wires];
        let mut available = Vec::with_capacity(num_wires);
        let mut reused_rng = StdRng::seed_from_u64(seed);
        for (gate, legacy) in gates.iter().zip(legacy_orders) {
            collect_available_wires(gate, num_wires, &mut unavailable, &mut available);
            available.shuffle(&mut reused_rng);
            assert_eq!(available, legacy);
            clear_unavailable_wires(gate, num_wires, &mut unavailable);
        }
        assert_eq!(reused_rng.next_u64(), legacy_rng.next_u64());
    }
}
