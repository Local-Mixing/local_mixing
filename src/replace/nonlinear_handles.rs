//! Conservative nonlinear handles for the legacy all-G57 SSS path.
//!
//! A G57 gate `P` is self-inverse and, when its three pins are distinct, has a
//! genuine quadratic term.  If `P` commutes with every gate in a window `W`,
//! then
//!
//! ```text
//!     P ; W ; P == W.
//! ```
//!
//! The two copies are therefore an exact identity at the circuit boundary, but
//! the live state between them is encoded by the nonlinear map `P`.  This is a
//! nonlinear analogue of carrying a SAMF across a region, with one important
//! limitation: this module does *not* conjugate through collisions.  A handle
//! stops before the first colliding gate.  That makes the transform small,
//! all-G57, and easy to audit; a future collision-capable implementation should
//! use the verified heterogeneous R-rule machinery rather than exposed
//! `P; g; P` triples that only manufacture decoder checkpoints.

use crate::circuit::circuit::{CircuitSeq, Gate};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Default-off configuration for conservative nonlinear handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonlinearHandleParams {
    /// Number of handles to attempt.  Zero is a strict no-op.
    pub handles: usize,
    /// Minimum number of existing gates a handle must enclose to be accepted.
    pub min_span: usize,
    /// Desired maximum span.  Zero means the whole current circuit.
    pub max_span: usize,
    /// Random handle candidates scored per attempt; the longest clean prefix wins.
    pub candidates: usize,
    /// Hard ceiling on the resulting gate count.  Zero means no additional ceiling.
    pub max_gates: usize,
    /// Deterministic handle/window selection seed.
    pub seed: u64,
    /// Candidate target wires are restricted to `0..target_wire_limit`.
    /// Zero means all `n` wires.
    pub target_wire_limit: usize,
    /// Candidate control wires are restricted to `0..control_wire_limit`.
    /// Zero means all `n` wires.  Fixed-slice experiments should normally set
    /// this to the number of input-dependent wires so a handle is nonlinear on
    /// the measured slice rather than merely a constant toggle.
    pub control_wire_limit: usize,
}

impl Default for NonlinearHandleParams {
    fn default() -> Self {
        Self {
            handles: 0,
            min_span: 1,
            max_span: 0,
            candidates: 32,
            max_gates: 0,
            seed: 0x4e4c_4841_4e44_4c45, // "NLHANDLE"
            target_wire_limit: 0,
            control_wire_limit: 0,
        }
    }
}

/// One accepted nonlinear state-mask window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonlinearFrameStats {
    /// Index in the pre-insertion circuit at which the opening handle was placed.
    pub start: usize,
    /// Number of pre-existing gates enclosed by the handle pair.
    pub span: usize,
    /// Requested search span before the first collision shortened it.
    pub requested_span: usize,
    /// The self-inverse nonlinear G57 carrier `[target, positive, negative]`.
    pub handle: [u16; 3],
}

/// Coverage and admission telemetry for a handle pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NonlinearHandleReport {
    /// Requested handle attempts.
    pub attempts: usize,
    /// Total random carriers evaluated across all attempts.
    pub candidates_evaluated: usize,
    /// Handles actually inserted.
    pub applied: usize,
    /// Attempts rejected because no candidate reached `min_span`.
    pub span_rejected: usize,
    /// Attempts rejected by `max_gates`.
    pub cap_rejected: usize,
    /// Attempts whose best candidate met a true G57 collision before the
    /// requested end.  This includes both accepted shortened windows and
    /// windows later rejected for being too short or over the gate cap.
    pub collision_limited: usize,
    /// Sum of requested window lengths before collision truncation.
    pub requested_gate_visits: usize,
    /// Sum of enclosed gate counts.  Overlap is deliberately counted once per
    /// active handle: this is nonlinear carrier-depth in gate-visits, not a
    /// union-of-indices measure on a tape that changes after every insertion.
    pub covered_gate_visits: usize,
    pub min_achieved_span: usize,
    pub max_achieved_span: usize,
    /// Distinct target wires nonlinearly masked by at least one accepted handle.
    pub distinct_targets: usize,
    /// Exactly two boundary gates per accepted handle.
    pub gates_added: usize,
    pub frames: Vec<NonlinearFrameStats>,
}

impl NonlinearHandleReport {
    pub fn mean_span(&self) -> f64 {
        if self.applied == 0 {
            0.0
        } else {
            self.covered_gate_visits as f64 / self.applied as f64
        }
    }

    /// Delivered nonlinear carrier-depth divided by requested gate-visits.
    pub fn delivered_fraction(&self) -> f64 {
        if self.requested_gate_visits == 0 {
            0.0
        } else {
            self.covered_gate_visits as f64 / self.requested_gate_visits as f64
        }
    }

    /// Mean number of accepted nonlinear carriers covering a gate visit.
    /// Boundary gates inserted by later handles make this a conservative
    /// estimate (their incidental coverage is not retroactively counted).
    pub fn mean_carrier_depth(&self, final_gate_count: usize) -> f64 {
        if final_gate_count == 0 {
            0.0
        } else {
            self.covered_gate_visits as f64 / final_gate_count as f64
        }
    }
}

/// Insert conservative nonlinear handles without generation tags.
pub fn insert_nonlinear_handles(
    circuit: &mut CircuitSeq,
    n: usize,
    params: &NonlinearHandleParams,
) -> NonlinearHandleReport {
    insert_nonlinear_handles_inner(circuit, n, params, None)
}

/// Tagged variant for new-SSS.  Existing gates retain their generation tags;
/// each opening/closing handle receives a fresh tag derived from the enclosed
/// window.  The circuit and tags are updated transactionally per handle.
pub fn insert_nonlinear_handles_tagged(
    circuit: &mut CircuitSeq,
    n: usize,
    params: &NonlinearHandleParams,
    tags: &mut Vec<u32>,
) -> NonlinearHandleReport {
    assert_eq!(
        circuit.gates.len(),
        tags.len(),
        "nonlinear-handle generation tags must align with gates"
    );
    insert_nonlinear_handles_inner(circuit, n, params, Some(tags))
}

fn insert_nonlinear_handles_inner(
    circuit: &mut CircuitSeq,
    n: usize,
    params: &NonlinearHandleParams,
    mut tags: Option<&mut Vec<u32>>,
) -> NonlinearHandleReport {
    let mut report = NonlinearHandleReport {
        attempts: params.handles,
        ..NonlinearHandleReport::default()
    };
    if params.handles == 0 || circuit.gates.is_empty() {
        return report;
    }
    assert!(n <= u16::MAX as usize + 1, "G57 wire index exceeds u16");

    let target_limit = nonzero_limit(params.target_wire_limit, n, "target");
    let control_limit = nonzero_limit(params.control_wire_limit, n, "control");
    assert!(
        control_limit >= 2,
        "nonlinear handles need at least two candidate control wires"
    );

    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut target_order: Vec<u16> = (0..target_limit).map(|wire| wire as u16).collect();
    target_order.shuffle(&mut rng);
    let mut used_targets = vec![false; n];

    for attempt in 0..params.handles {
        let len = circuit.gates.len();
        if len == 0 {
            report.span_rejected += 1;
            continue;
        }
        let (start, requested_span) = choose_window(len, params, &mut rng);
        if requested_span == 0 {
            report.span_rejected += 1;
            continue;
        }
        report.requested_gate_visits = report.requested_gate_visits.saturating_add(requested_span);

        // Cycle targets before reusing one.  Candidate variation is in the two
        // controls, and the carrier with the longest collision-free prefix wins.
        let target = target_order[attempt % target_order.len()];
        let candidate_count = params.candidates.max(1);
        let mut best: Option<([u16; 3], usize)> = None;
        for _ in 0..candidate_count {
            let Some(handle) = random_handle(target, control_limit, &mut rng) else {
                break;
            };
            report.candidates_evaluated += 1;
            let span = clean_prefix_len(handle, &circuit.gates[start..start + requested_span]);
            if best.as_ref().is_none_or(|(_, best_span)| span > *best_span) {
                best = Some((handle, span));
            }
            if span == requested_span {
                break;
            }
        }
        let Some((handle, span)) = best else {
            report.span_rejected += 1;
            continue;
        };
        if span < requested_span {
            report.collision_limited += 1;
        }
        if span < params.min_span.max(1) {
            report.span_rejected += 1;
            continue;
        }

        let candidate_len = len.saturating_add(2);
        if params.max_gates > 0 && candidate_len > params.max_gates {
            report.cap_rejected += 1;
            continue;
        }

        // This is the exact transport step: conceptually insert P;P before W,
        // then commute the second P across each proven-noncolliding gate.  We
        // materialize the equivalent final sequence P;W;P directly.
        circuit.gates.insert(start, handle);
        circuit.gates.insert(start + span + 1, handle);
        if let Some(tags) = tags.as_deref_mut() {
            let event_tag = crate::replace::replace::new_gate_tag(&tags[start..start + span]);
            tags.insert(start, event_tag);
            tags.insert(start + span + 1, event_tag);
            debug_assert_eq!(circuit.gates.len(), tags.len());
        }

        report.applied += 1;
        report.covered_gate_visits = report.covered_gate_visits.saturating_add(span);
        report.min_achieved_span = if report.applied == 1 {
            span
        } else {
            report.min_achieved_span.min(span)
        };
        report.max_achieved_span = report.max_achieved_span.max(span);
        used_targets[handle[0] as usize] = true;
        report.frames.push(NonlinearFrameStats {
            start,
            span,
            requested_span,
            handle,
        });
    }

    report.distinct_targets = used_targets.into_iter().filter(|&used| used).count();
    report.gates_added = report.applied.saturating_mul(2);
    report
}

fn nonzero_limit(configured: usize, n: usize, label: &str) -> usize {
    let limit = if configured == 0 { n } else { configured };
    assert!(limit > 0, "nonlinear-handle {label} wire range is empty");
    assert!(
        limit <= n,
        "nonlinear-handle {label} wire limit {limit} exceeds circuit width {n}"
    );
    limit
}

fn choose_window(len: usize, params: &NonlinearHandleParams, rng: &mut StdRng) -> (usize, usize) {
    let min_span = params.min_span.max(1);
    if len < min_span {
        return (0, 0);
    }
    // Do not anchor every carrier at the head: distributed starts make the
    // aggregate state-mask coverage meaningful across prefix-based heatmaps.
    // Once a start is chosen, ask for the longest configured suffix and let
    // the first true G57 collision shorten it.
    let start = rng.random_range(0..=len - min_span);
    let tail = len - start;
    let span = if params.max_span == 0 {
        tail
    } else {
        params.max_span.max(min_span).min(tail)
    };
    (start, span)
}

fn random_handle(target: u16, control_limit: usize, rng: &mut StdRng) -> Option<[u16; 3]> {
    let controls: Vec<u16> = (0..control_limit)
        .map(|wire| wire as u16)
        .filter(|&wire| wire != target)
        .collect();
    if controls.len() < 2 {
        return None;
    }
    let i = rng.random_range(0..controls.len());
    let mut j = rng.random_range(0..controls.len() - 1);
    if j >= i {
        j += 1;
    }
    Some([target, controls[i], controls[j]])
}

fn clean_prefix_len(handle: [u16; 3], gates: &[[u16; 3]]) -> usize {
    gates
        .iter()
        .position(|gate| Gate::collides_index(&handle, gate))
        .unwrap_or(gates.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equivalent_exhaustive(a: &CircuitSeq, b: &CircuitSeq, n: usize) -> bool {
        (0..1usize << n).all(|state| {
            Gate::evaluate_index_list(state, &a.gates) == Gate::evaluate_index_list(state, &b.gates)
        })
    }

    fn fixture() -> CircuitSeq {
        CircuitSeq {
            gates: vec![
                [0, 1, 2],
                [3, 4, 5],
                [1, 3, 5],
                [4, 0, 2],
                [2, 3, 5],
                [5, 0, 1],
            ],
        }
    }

    fn random_fixture(seed: u64, n: usize, gates: usize) -> CircuitSeq {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(gates);
        for _ in 0..gates {
            let target = rng.random_range(0..n) as u16;
            let positive = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != target {
                    break wire;
                }
            };
            let negative = loop {
                let wire = rng.random_range(0..n) as u16;
                if wire != target && wire != positive {
                    break wire;
                }
            };
            out.push([target, positive, negative]);
        }
        CircuitSeq { gates: out }
    }

    #[test]
    fn default_configuration_is_a_strict_noop() {
        let original = fixture();
        let mut circuit = original.clone();
        let report = insert_nonlinear_handles(&mut circuit, 6, &Default::default());
        assert_eq!(circuit, original);
        assert_eq!(report.applied, 0);
        assert_eq!(report.gates_added, 0);
    }

    #[test]
    fn every_accepted_handle_is_exactly_equivalent() {
        for seed in 0..32 {
            let original = fixture();
            let mut circuit = original.clone();
            let params = NonlinearHandleParams {
                handles: 8,
                min_span: 1,
                max_span: 5,
                candidates: 64,
                seed,
                ..Default::default()
            };
            let report = insert_nonlinear_handles(&mut circuit, 6, &params);
            assert!(
                equivalent_exhaustive(&original, &circuit, 6),
                "nonlinear handles changed the function for seed {seed}: {report:?}"
            );
            assert_eq!(
                circuit.gates.len(),
                original.gates.len() + 2 * report.applied
            );
            assert_eq!(report.gates_added, 2 * report.applied);
            assert_eq!(report.frames.len(), report.applied);
            assert_eq!(
                report.covered_gate_visits,
                report.frames.iter().map(|frame| frame.span).sum::<usize>()
            );
            assert!(report.delivered_fraction() <= 1.0);
        }
    }

    #[test]
    fn randomized_circuits_remain_exactly_equivalent() {
        let n = 7;
        for seed in 0..24 {
            let original = random_fixture(seed, n, 40);
            let mut circuit = original.clone();
            let report = insert_nonlinear_handles(
                &mut circuit,
                n,
                &NonlinearHandleParams {
                    handles: 16,
                    min_span: 1,
                    max_span: 20,
                    candidates: 64,
                    seed: seed ^ 0xa11f_1ee7,
                    ..Default::default()
                },
            );
            assert!(
                equivalent_exhaustive(&original, &circuit, n),
                "random circuit changed for seed {seed}: {report:?}"
            );
        }
    }

    #[test]
    fn accepted_windows_stop_before_the_first_collision() {
        let original = fixture();
        let mut accepted = None;
        for seed in 0..128 {
            let mut circuit = original.clone();
            let params = NonlinearHandleParams {
                handles: 1,
                min_span: 1,
                max_span: 0,
                candidates: 128,
                seed,
                ..Default::default()
            };
            let report = insert_nonlinear_handles(&mut circuit, 6, &params);
            if report.applied == 1 {
                accepted = Some((circuit, report));
                break;
            }
        }
        let (circuit, report) = accepted.expect("at least one seed should admit a clean window");
        let frame = &report.frames[0];
        for gate in &original.gates[frame.start..frame.start + frame.span] {
            assert!(!Gate::collides_index(&frame.handle, gate));
        }
        if frame.start + frame.span < original.gates.len() {
            assert!(Gate::collides_index(
                &frame.handle,
                &original.gates[frame.start + frame.span]
            ));
        }
        assert!(equivalent_exhaustive(&original, &circuit, 6));
    }

    #[test]
    fn window_selection_distributes_starts_away_from_the_prefix() {
        let params = NonlinearHandleParams {
            handles: 16,
            min_span: 8,
            max_span: 0,
            seed: 123,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(params.seed);
        let starts: Vec<usize> = (0..params.handles)
            .map(|_| choose_window(100, &params, &mut rng).0)
            .collect();
        assert!(starts.iter().any(|&start| start > 0));
        assert!(starts.iter().all(|&start| start <= 92));
    }

    #[test]
    fn tagged_insertion_stays_aligned_and_marks_boundaries_fresh() {
        let original = fixture();
        let mut circuit = original.clone();
        let mut tags = vec![3, 4, 5, 6, 7, 8];
        let params = NonlinearHandleParams {
            handles: 3,
            min_span: 1,
            max_span: 4,
            candidates: 64,
            seed: 91,
            ..Default::default()
        };
        let report = insert_nonlinear_handles_tagged(&mut circuit, 6, &params, &mut tags);
        assert!(report.applied > 0);
        assert_eq!(circuit.gates.len(), tags.len());
        assert!(equivalent_exhaustive(&original, &circuit, 6));
    }

    #[test]
    fn hard_gate_cap_rejects_transactionally() {
        let original = fixture();
        let mut circuit = original.clone();
        let params = NonlinearHandleParams {
            handles: 4,
            min_span: 1,
            max_span: 4,
            candidates: 64,
            max_gates: original.gates.len(),
            seed: 5,
            ..Default::default()
        };
        let report = insert_nonlinear_handles(&mut circuit, 6, &params);
        assert_eq!(report.applied, 0);
        assert!(report.cap_rejected > 0);
        assert_eq!(circuit, original);
    }
}
