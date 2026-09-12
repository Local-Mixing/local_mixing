//! Bounded, empirical gate-wise leakage checks against an immutable reference.
//!
//! A value segment is the interval between consecutive *target writes* on one
//! wire. Reading a wire as a control does not end its value segment. Initial and
//! final I/O segments have no pair of gate endpoints and are not repair targets.
//!
//! Both circuits receive the same fixed, uniformly sampled full-width inputs.
//! A reference may use only a subset of these wires; extra mixed-circuit wires
//! receive independent input bits. Encoded, permuted, or constrained input
//! interfaces need an explicit mapping/sampling adapter before using this API.
//! These tests are experimental evidence, not a security certificate. In
//! particular, candidate selection reuses the held-out bank, so a separate fresh
//! audit is needed to assess a circuit after many adaptive repair attempts.

use crate::circuit::XGate;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct DetectorConfig {
    /// Deterministic feature/segment selection and default sample-bank seed.
    pub seed: u64,
    /// Number of independent 64-sample bit-sliced training batches.
    pub train_batches: usize,
    /// Separate 64-sample batches, never used to fit an affine relation.
    pub heldout_batches: usize,
    /// Reservoir-sampled original internal value features; zero disables affine.
    pub max_original_segments: usize,
    /// Reservoir-sampled original gate predicates; zero disables correlation.
    pub max_original_firings: usize,
    /// Maximum number of mixed internal segments evaluated by one scan.
    pub max_mixed_segments: usize,
    /// Maximum number of detailed hot-segment records retained per scan.
    pub max_hot_segments: usize,
    /// Minimum absolute Pearson/phi coefficient in BOTH sample partitions.
    pub min_abs_correlation: f64,
    /// Required validation accuracy after an exact affine fit on training data.
    pub min_affine_validation_accuracy: f64,
    /// Both zero and one counts must reach this in each partition.
    pub min_minority_count: usize,
    /// Conservative budget for sample/trace/basis storage, measured in u64s.
    pub max_trace_words: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            seed: 0x5143_5f67_6174_6573,
            train_batches: 4,
            heldout_batches: 4,
            max_original_segments: 32,
            max_original_firings: 128,
            max_mixed_segments: 2048,
            max_hot_segments: 128,
            min_abs_correlation: 0.98,
            min_affine_validation_accuracy: 1.0,
            min_minority_count: 8,
            max_trace_words: 8_388_608,
        }
    }
}

impl DetectorConfig {
    /// Cheap preflight for a caller that will run a long mixing stage first.
    /// Checks configuration, shared wire space, and budgets without tracing.
    pub fn validate_input(&self, reference: &[XGate], num_wires: usize) -> Result<(), String> {
        self.validate()?;
        validate_gates(reference, num_wires)?;
        self.check_storage(num_wires)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.train_batches == 0 || self.heldout_batches == 0 {
            return Err("QC requires nonempty training and held-out sample banks".into());
        }
        if self.train_batches > 64 || self.heldout_batches > 64 {
            return Err("QC permits at most 64 batches per sample partition".into());
        }
        if self.max_original_segments == 0 && self.max_original_firings == 0 {
            return Err("QC requires at least one enabled detector".into());
        }
        if self.max_original_segments >= self.train_batches * 64 {
            return Err("QC affine feature cap must be smaller than training sample count".into());
        }
        if self.max_mixed_segments == 0 || self.max_hot_segments == 0 {
            return Err("QC mixed-segment and hot-record budgets must be positive".into());
        }
        if !self.min_abs_correlation.is_finite()
            || !(0.0 < self.min_abs_correlation && self.min_abs_correlation <= 1.0)
        {
            return Err("QC correlation threshold must be finite and in (0, 1]".into());
        }
        if !self.min_affine_validation_accuracy.is_finite()
            || !(0.5 < self.min_affine_validation_accuracy
                && self.min_affine_validation_accuracy <= 1.0)
        {
            return Err("QC affine held-out accuracy must be finite and in (0.5, 1]".into());
        }
        if self.min_minority_count == 0
            || self.min_minority_count > self.train_batches.min(self.heldout_batches) * 32
        {
            return Err("QC minority count must be positive and fit each sample partition".into());
        }
        Ok(())
    }

    fn check_storage(&self, num_wires: usize) -> Result<(), String> {
        // Include input bank, working state, original and mixed captures, basis,
        // temporary vectors, coefficient masks, and returned affine evidence.
        // Segment metadata is O(wires + configured caps), never O(circuit size).
        let batches = self.train_batches + self.heldout_batches;
        let features = self.max_original_segments;
        let words = (num_wires as u128) * ((batches + 1) as u128)
            + 8 * (batches as u128)
                * (features as u128
                    + self.max_original_firings as u128
                    + self.max_mixed_segments as u128)
            + 2 * (features as u128) * ((features + 64) / 64) as u128
            + (self.max_hot_segments as u128) * (features as u128);
        if words > self.max_trace_words as u128 {
            return Err(format!(
                "QC configured sample/trace storage requires at most {words} words, exceeding max_trace_words={}",
                self.max_trace_words
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub wire: u16,
    /// Gate that creates this value, by writing the target wire.
    pub start_gate: usize,
    /// Next gate that writes the same wire (control reads do not count).
    pub end_gate: usize,
}

#[derive(Clone, Debug)]
pub enum Evidence {
    Affine {
        /// Producer gate indices of reference value features in the XOR relation.
        original_features: Vec<usize>,
        complement: bool,
        heldout_accuracy: f64,
    },
    FiringCorrelation {
        /// The producer of the outgoing value segment used as the repair target.
        /// This evidence concerns its firing predicate, not its output value.
        mixed_gate: usize,
        original_gate: usize,
        train_correlation: f64,
        heldout_correlation: f64,
    },
}

#[derive(Clone, Debug)]
pub struct HotSegment {
    pub wire: u16,
    pub start_gate: usize,
    pub end_gate: usize,
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Debug, Default)]
pub struct ScanReport {
    pub hot_segments: Vec<HotSegment>,
    /// Count before the detailed-record cap is applied.
    pub hot_segment_count: usize,
    pub hot_segments_truncated: bool,
    /// Eligible internal segments in the requested scan region.
    pub internal_segments: usize,
    pub scanned_segments: usize,
    /// Gate count in the entire current circuit.
    pub firing_gates: usize,
    pub scanned_firings: usize,
    /// I/O fringe segments in the entire circuit, including untouched wires.
    pub boundary_segments: usize,
    /// Final target-writing gates lack an outgoing internal repair segment.
    /// Their predicates are audited separately without fictitious segment bounds.
    pub unmappable_firing_gates: usize,
    /// Final target-writer predicates eligible in the requested scan region.
    pub boundary_firings: usize,
    pub scanned_boundary_firings: usize,
    /// Hot final-writer predicates cannot be assigned an internal repair segment.
    /// They are nevertheless audited so moving a leak to the fringe is not a
    /// successful repair. Candidate acceptance must also check this count.
    pub hot_boundary_firings: usize,
    /// Bounded detailed firing evidence for the above, capped at max_hot_segments.
    pub boundary_firing_evidence: Vec<Evidence>,
    /// A cap prevented complete internal-segment or boundary-firing coverage.
    pub truncated: bool,
    pub original_internal_segments: usize,
    pub original_segments_sampled: usize,
    pub original_firing_gates: usize,
    pub original_firings_sampled: usize,
    /// Reference caps limit the relations sought, independently of scan coverage.
    pub reference_truncated: bool,
}

/// Immutable reference features and a private input bank. No sampler RNG is used.
pub struct Detector {
    config: DetectorConfig,
    num_wires: usize,
    inputs: Vec<Vec<u64>>,
    affine: AffineBasis,
    original_firings: Vec<(usize, Vec<u64>)>,
    original_internal_segments: usize,
    original_segments_sampled: usize,
    original_firing_gates: usize,
}

impl Detector {
    pub fn new(
        original: &[XGate],
        num_wires: usize,
        config: DetectorConfig,
    ) -> Result<Self, String> {
        let sample_seed = config.seed;
        Self::new_with_sample_seed(original, num_wires, config, sample_seed)
    }

    /// Construct a detector with a separate paired input bank, preserving the
    /// reference-feature and segment selections controlled by `config.seed`.
    /// Reserve such a bank until after adaptive repair selection has finished.
    /// A before/after comparison should use this same fresh detector on both
    /// circuits; compare their complete reports, including reference/scan caps.
    pub fn new_with_sample_seed(
        original: &[XGate],
        num_wires: usize,
        config: DetectorConfig,
        sample_seed: u64,
    ) -> Result<Self, String> {
        config.validate_input(original, num_wires)?;
        let batches = config.train_batches + config.heldout_batches;
        let mut random = SplitMix64(sample_seed);
        let inputs = (0..batches)
            .map(|_| (0..num_wires).map(|_| random.next()).collect())
            .collect::<Vec<Vec<u64>>>();
        let selection = select_segments(
            original,
            num_wires,
            config.max_original_segments,
            config.seed ^ 0x7265_665f_7661_6c75,
            None,
        );
        let firing_indices = select_indices(
            original.len(),
            config.max_original_firings,
            config.seed ^ 0x7265_665f_6669_7265,
        );
        let mut capture_indices = selection
            .segments
            .iter()
            .map(|s| s.start_gate)
            .chain(firing_indices.iter().copied())
            .collect::<Vec<_>>();
        capture_indices.sort_unstable();
        capture_indices.dedup();
        let captures = capture(original, &inputs, &capture_indices);
        let features = selection
            .segments
            .iter()
            .map(|s| {
                let i = capture_indices.binary_search(&s.start_gate).unwrap();
                (s.start_gate, captures[i].value.clone())
            })
            .collect::<Vec<_>>();
        let original_firings = firing_indices
            .iter()
            .map(|&gate| {
                let i = capture_indices.binary_search(&gate).unwrap();
                (gate, captures[i].firing.clone())
            })
            .collect();
        let affine = AffineBasis::new(&features, config.train_batches, batches);
        Ok(Self {
            original_internal_segments: selection.total,
            original_segments_sampled: features.len(),
            original_firing_gates: original.len(),
            num_wires,
            config,
            inputs,
            affine,
            original_firings,
        })
    }

    pub fn scan(&self, current: &[XGate]) -> Result<ScanReport, String> {
        self.scan_impl(current, None)
    }

    /// Screen a candidate's enclosing gate range, including crossing segments.
    /// A segment is in scope if its inclusive writer-to-writer interval overlaps
    /// the half-open range; final-writer predicates inside the range are also
    /// audited. Reject a purported clean candidate when `truncated` or when
    /// `hot_boundary_firings` is nonzero.
    /// A clean result is always relative to the selected original references.
    pub fn scan_region(
        &self,
        current: &[XGate],
        region: Range<usize>,
    ) -> Result<ScanReport, String> {
        if region.start >= region.end || region.end > current.len() {
            return Err("QC scan region must be a nonempty in-bounds gate range".into());
        }
        self.scan_impl(current, Some(region))
    }

    fn scan_impl(
        &self,
        current: &[XGate],
        region: Option<Range<usize>>,
    ) -> Result<ScanReport, String> {
        validate_gates(current, self.num_wires)?;
        let selection = select_segments(
            current,
            self.num_wires,
            self.config.max_mixed_segments,
            self.config.seed ^ 0x6d69_785f_7661_6c75,
            region.as_ref(),
        );
        let boundary_candidates = selection
            .terminal_writers
            .iter()
            .copied()
            .filter(|&gate| region.as_ref().is_none_or(|r| r.contains(&gate)))
            .collect::<Vec<_>>();
        let boundary_indices = if self.original_firings.is_empty() {
            Vec::new()
        } else {
            select_indices(
                boundary_candidates.len(),
                self.config.max_mixed_segments,
                self.config.seed ^ 0x626f_756e_6461_7279,
            )
            .iter()
            .map(|&i| boundary_candidates[i])
            .collect::<Vec<_>>()
        };
        let mut indices = selection
            .segments
            .iter()
            .map(|s| s.start_gate)
            .chain(boundary_indices.iter().copied())
            .collect::<Vec<_>>();
        indices.sort_unstable();
        let captures = capture(current, &self.inputs, &indices);
        let mut report = ScanReport {
            internal_segments: selection.total,
            scanned_segments: selection.segments.len(),
            firing_gates: current.len(),
            scanned_firings: if self.original_firings.is_empty() {
                0
            } else {
                selection.segments.len()
            },
            boundary_segments: self.num_wires + selection.written_wires,
            unmappable_firing_gates: selection.written_wires,
            boundary_firings: boundary_candidates.len(),
            scanned_boundary_firings: boundary_indices.len(),
            truncated: selection.total > selection.segments.len()
                || (!self.original_firings.is_empty()
                    && boundary_candidates.len() > boundary_indices.len()),
            original_internal_segments: self.original_internal_segments,
            original_segments_sampled: self.original_segments_sampled,
            original_firing_gates: self.original_firing_gates,
            original_firings_sampled: self.original_firings.len(),
            reference_truncated: (self.config.max_original_segments > 0
                && self.original_internal_segments > self.original_segments_sampled)
                || (self.config.max_original_firings > 0
                    && self.original_firing_gates > self.original_firings.len()),
            ..ScanReport::default()
        };
        for segment in &selection.segments {
            let sample = &captures[indices.binary_search(&segment.start_gate).unwrap()];
            let mut evidence = Vec::new();
            if self.config.max_original_segments > 0 {
                if let Some(hit) = self.affine.predict(&sample.value, &self.config) {
                    evidence.push(hit);
                }
            }
            if let Some(hit) = self.firing_evidence(segment.start_gate, &sample.firing) {
                evidence.push(hit);
            }
            if !evidence.is_empty() {
                report.hot_segment_count += 1;
                if report.hot_segments.len() < self.config.max_hot_segments {
                    report.hot_segments.push(HotSegment {
                        wire: segment.wire,
                        start_gate: segment.start_gate,
                        end_gate: segment.end_gate,
                        evidence,
                    });
                }
            }
        }
        for gate in boundary_indices {
            let sample = &captures[indices.binary_search(&gate).unwrap()];
            if let Some(hit) = self.firing_evidence(gate, &sample.firing) {
                report.hot_boundary_firings += 1;
                if report.boundary_firing_evidence.len() < self.config.max_hot_segments {
                    report.boundary_firing_evidence.push(hit);
                }
            }
        }
        report.hot_segments_truncated = report.hot_segment_count > report.hot_segments.len();
        Ok(report)
    }

    fn firing_evidence(&self, mixed_gate: usize, firing: &[u64]) -> Option<Evidence> {
        let split = self.config.train_batches;
        let mut best = None;
        let mut best_score = -1.0;
        for (original_gate, reference) in &self.original_firings {
            let Some(train) = phi(
                &firing[..split],
                &reference[..split],
                self.config.min_minority_count,
            ) else {
                continue;
            };
            if train.abs() < self.config.min_abs_correlation {
                continue;
            }
            let Some(heldout) = phi(
                &firing[split..],
                &reference[split..],
                self.config.min_minority_count,
            ) else {
                continue;
            };
            if heldout.abs() < self.config.min_abs_correlation || train * heldout <= 0.0 {
                continue;
            }
            let score = train.abs().min(heldout.abs());
            if score > best_score {
                best_score = score;
                best = Some(Evidence::FiringCorrelation {
                    mixed_gate,
                    original_gate: *original_gate,
                    train_correlation: train,
                    heldout_correlation: heldout,
                });
            }
        }
        best
    }
}

fn validate_gates(gates: &[XGate], num_wires: usize) -> Result<(), String> {
    if num_wires == 0 || num_wires > u16::MAX as usize + 1 {
        return Err("QC wire count must lie in 1..=65536".into());
    }
    for (i, gate) in gates.iter().enumerate() {
        if gate.target as usize >= num_wires
            || gate.ctrls.iter().any(|&(w, _)| w as usize >= num_wires)
        {
            return Err(format!(
                "QC gate {i} references a wire outside the shared input space"
            ));
        }
        if gate.ctrls.iter().any(|&(w, _)| w == gate.target) {
            return Err(format!("QC gate {i} illegally controls its own target"));
        }
    }
    Ok(())
}

struct Selection {
    segments: Vec<Segment>,
    total: usize,
    written_wires: usize,
    terminal_writers: Vec<usize>,
}

fn select_segments(
    gates: &[XGate],
    num_wires: usize,
    cap: usize,
    seed: u64,
    region: Option<&Range<usize>>,
) -> Selection {
    let mut last_write = vec![None; num_wires];
    let mut segments = Vec::new();
    let mut total = 0;
    let mut random = SplitMix64(seed);
    for (end_gate, gate) in gates.iter().enumerate() {
        if let Some(start_gate) = last_write[gate.target as usize] {
            if region.is_none_or(|r| start_gate < r.end && end_gate >= r.start) {
                total += 1;
                let segment = Segment {
                    wire: gate.target,
                    start_gate,
                    end_gate,
                };
                reservoir_push(&mut segments, segment, total, cap, &mut random);
            }
        }
        last_write[gate.target as usize] = Some(end_gate);
    }
    segments.sort_unstable_by_key(|s| s.start_gate);
    let mut terminal_writers = last_write
        .iter()
        .filter_map(|&gate| gate)
        .collect::<Vec<_>>();
    terminal_writers.sort_unstable();
    Selection {
        segments,
        total,
        written_wires: terminal_writers.len(),
        terminal_writers,
    }
}

fn reservoir_push<T>(
    values: &mut Vec<T>,
    item: T,
    seen: usize,
    cap: usize,
    random: &mut SplitMix64,
) {
    if values.len() < cap {
        values.push(item);
    } else if cap > 0 {
        let index = (random.next() % seen as u64) as usize;
        if index < cap {
            values[index] = item;
        }
    }
}

fn select_indices(count: usize, cap: usize, seed: u64) -> Vec<usize> {
    let mut selected = Vec::new();
    let mut random = SplitMix64(seed);
    for i in 0..count {
        reservoir_push(&mut selected, i, i + 1, cap, &mut random);
    }
    selected.sort_unstable();
    selected
}

struct Captured {
    value: Vec<u64>,
    firing: Vec<u64>,
}

/// Stores only selected gates. Each circuit traversal uses one u64 per wire.
fn capture(gates: &[XGate], inputs: &[Vec<u64>], selected: &[usize]) -> Vec<Captured> {
    let mut captures = selected
        .iter()
        .map(|_| Captured {
            value: vec![0; inputs.len()],
            firing: vec![0; inputs.len()],
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return captures;
    }
    for (batch, input) in inputs.iter().enumerate() {
        let mut state = input.clone();
        let mut next = 0;
        for (i, gate) in gates.iter().enumerate() {
            let before = state[gate.target as usize];
            gate.apply_lanes(&mut state);
            if i == selected[next] {
                let after = state[gate.target as usize];
                captures[next].value[batch] = after;
                captures[next].firing[batch] = before ^ after;
                next += 1;
                if next == selected.len() {
                    break;
                }
            }
        }
    }
    captures
}

struct BasisRow {
    /// Train and validation trace reduced by the SAME column operations.
    values: Vec<u64>,
    coefficients: Vec<u64>,
}

struct AffineBasis {
    pivots: Vec<Option<BasisRow>>,
    feature_gates: Vec<usize>,
    train_batches: usize,
}

impl AffineBasis {
    fn new(features: &[(usize, Vec<u64>)], train_batches: usize, batches: usize) -> Self {
        let mut basis = Self {
            pivots: (0..train_batches * 64).map(|_| None).collect(),
            feature_gates: features.iter().map(|(gate, _)| *gate).collect(),
            train_batches,
        };
        // Coefficient zero is the affine offset. Insert it first so constant
        // functions do not masquerade as relations involving reference wires.
        for (column, mut values) in std::iter::once(vec![u64::MAX; batches])
            .chain(features.iter().map(|(_, v)| v.clone()))
            .enumerate()
        {
            let mut coefficients = vec![0; (features.len() + 64) / 64];
            coefficients[column / 64] |= 1 << (column % 64);
            while let Some(pivot) = first_one(&values[..train_batches]) {
                if let Some(row) = &basis.pivots[pivot] {
                    xor_assign(&mut values, &row.values);
                    xor_assign(&mut coefficients, &row.coefficients);
                } else {
                    basis.pivots[pivot] = Some(BasisRow {
                        values,
                        coefficients,
                    });
                    break;
                }
            }
        }
        basis
    }

    fn predict(&self, target: &[u64], config: &DetectorConfig) -> Option<Evidence> {
        if !has_variation(&target[..self.train_batches], config.min_minority_count)
            || !has_variation(&target[self.train_batches..], config.min_minority_count)
        {
            return None;
        }
        let mut residual = target.to_vec();
        let mut coefficients = vec![0; (self.feature_gates.len() + 64) / 64];
        while let Some(pivot) = first_one(&residual[..self.train_batches]) {
            let row = self.pivots[pivot].as_ref()?;
            xor_assign(&mut residual, &row.values);
            xor_assign(&mut coefficients, &row.coefficients);
        }
        let errors: usize = residual[self.train_batches..]
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();
        let heldout_accuracy = 1.0 - errors as f64 / (config.heldout_batches * 64) as f64;
        if heldout_accuracy < config.min_affine_validation_accuracy {
            return None;
        }
        let original_features = self
            .feature_gates
            .iter()
            .enumerate()
            .filter_map(|(i, &gate)| {
                let c = i + 1;
                ((coefficients[c / 64] >> (c % 64)) & 1 != 0).then_some(gate)
            })
            .collect::<Vec<_>>();
        if original_features.is_empty() {
            return None;
        }
        Some(Evidence::Affine {
            original_features,
            complement: coefficients[0] & 1 != 0,
            heldout_accuracy,
        })
    }
}

fn first_one(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .enumerate()
        .find_map(|(i, &word)| (word != 0).then(|| i * 64 + word.trailing_zeros() as usize))
}

fn xor_assign(lhs: &mut [u64], rhs: &[u64]) {
    for (x, y) in lhs.iter_mut().zip(rhs) {
        *x ^= y;
    }
}

fn has_variation(words: &[u64], minimum: usize) -> bool {
    let ones: usize = words.iter().map(|w| w.count_ones() as usize).sum();
    ones >= minimum && words.len() * 64 - ones >= minimum
}

/// Pearson correlation for binary variables is phi. Marginal rates matter:
/// two independent rarely-firing gates can agree on nearly every sample.
fn phi(a: &[u64], b: &[u64], minimum: usize) -> Option<f64> {
    if !has_variation(a, minimum) || !has_variation(b, minimum) {
        return None;
    }
    let n = (a.len() * 64) as f64;
    let a_ones = a.iter().map(|w| w.count_ones() as usize).sum::<usize>() as f64;
    let b_ones = b.iter().map(|w| w.count_ones() as usize).sum::<usize>() as f64;
    let both = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones() as usize)
        .sum::<usize>() as f64;
    let denominator = (a_ones * (n - a_ones) * b_ones * (n - b_ones)).sqrt();
    Some(((n * both - a_ones * b_ones) / denominator).clamp(-1.0, 1.0))
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
#[path = "../../../../tests/stages/db_mixing/leakage_repair/detect/tests.rs"]
mod tests;
