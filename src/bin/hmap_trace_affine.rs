//! Cumulative whole-trace affine reconstruction attack.
//!
//! Unlike `hmap_affine`, which fits each C_i from one isolated G_j snapshot,
//! this tool grows one GF(2) feature span as it advances through G.  In
//! `checkpoint-state` mode it adds every wire at evenly spaced checkpoints;
//! the last column therefore fits from all sampled states across the complete
//! G trajectory.  In `gate-delta` mode it adds the firing bit of every gate:
//! because an XGate changes one target, initial wires plus those deltas span
//! every wire value at every gate boundary.
//!
//! Samples are split three ways.  Fit samples construct the basis, validation
//! samples reject spurious fit dependencies and choose between an affine
//! witness and a constant predictor, and locked test samples supply the
//! reported error.  Two plates are written:
//!
//!   * `<out>.bin`: exact-span semantics (inconsistent => H=0.5), compatible
//!     with `hmap_affine`;
//!   * `<out>.witness.bin`: the feasible affine witness induced by elimination,
//!     selected against a constant on validation and scored only on locked test.
//!
//! The dense all-gate feature set is normally far larger than a feasible fit
//! sample set.  That mode therefore requires `--allow-underdetermined` and
//! reports the first saturation prefix.  A locked-test success is evidence;
//! a negative result after saturation is explicitly inconclusive.

use clap::{Parser, ValueEnum};
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TraceMode {
    /// Add all G wires at evenly spaced checkpoints, cumulatively.
    CheckpointState,
    /// Add every gate firing bit/delta. Exact full trace until fit saturation.
    GateDelta,
}

impl TraceMode {
    fn name(self) -> &'static str {
        match self {
            Self::CheckpointState => "checkpoint-state",
            Self::GateDelta => "gate-delta",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "hmap_trace_affine")]
struct Args {
    /// Original/source circuit C.
    #[arg(long)]
    c: String,
    /// Gadgetized/mixed circuit G.
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "g57")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical width; random x occupies wires 0..n and all other inputs are 0.
    #[arg(long)]
    n: usize,
    #[arg(long, value_enum, default_value_t = TraceMode::CheckpointState)]
    trace_mode: TraceMode,
    /// Source-prefix stride (the final source prefix is always included).
    #[arg(long, default_value_t = 210)]
    c_step: usize,
    /// Number of G columns, including 0 and the complete trace.
    #[arg(long, default_value_t = 12)]
    g_checkpoints: usize,
    /// Analyze only the first N gates of G. Zero means the complete trace.
    /// This is useful for replaying and localizing a discovered frontier.
    #[arg(long, default_value_t = 0)]
    g_limit: usize,
    /// 64-sample lane words used to construct the GF(2) basis.
    #[arg(long, default_value_t = 112)]
    fit_batches: usize,
    /// Independent lane words used to reject/select witnesses.
    #[arg(long, default_value_t = 16)]
    validation_batches: usize,
    /// Locked lane words used only for final reported errors.
    #[arg(long, default_value_t = 32)]
    test_batches: usize,
    #[arg(long, default_value_t = 12345)]
    seed: u64,
    /// Required when candidate features can meet/exceed fit samples.
    #[arg(long, default_value_t = false)]
    allow_underdetermined: bool,
    /// In gate-delta mode, offer every Nth gate delta while still executing
    /// every gate. N=1 is the literal all-gate trace; a larger N gives a
    /// statistically overdetermined trace-wide sketch.
    #[arg(long, default_value_t = 1)]
    delta_stride: usize,
    /// Residue within --delta-stride used for the first selected gate.
    #[arg(long, default_value_t = 0)]
    delta_offset: usize,
    /// In gate-delta mode, also add all current wires at every output
    /// checkpoint. This combines a trace-wide delta sketch with cumulative
    /// whole-state anchors.
    #[arg(long, default_value_t = false)]
    include_checkpoint_states: bool,
    /// Stop reducing new gate deltas after fit rank saturates. The circuit is
    /// still executed to the end and all requested columns are emitted.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    stop_at_saturation: bool,
    /// Output stem. Writes exact and `.witness` plates plus metadata.
    #[arg(long)]
    out: String,
}

fn read_circuit(path: &str, format: &str) -> (Vec<XGate>, usize) {
    match format {
        "g57" => {
            let gates = read_g57_file(path).unwrap_or_else(|e| panic!("read {path} (g57): {e}"));
            let nw = max_wire(&gates) as usize + 1;
            (gates, nw)
        }
        "mpmct1" => read_mpmct(path).unwrap_or_else(|e| panic!("read {path} (mpmct1): {e}")),
        other => panic!("unknown format {other}"),
    }
}

fn step_indices(end: usize, step: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut value = 0usize;
    let step = step.max(1);
    while value < end {
        out.push(value);
        value = value.saturating_add(step);
    }
    out.push(end);
    out
}

fn even_indices(end: usize, count: usize) -> Vec<usize> {
    if end == 0 {
        return vec![0];
    }
    let count = count.max(2).min(end + 1);
    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        out.push(((k as u128 * end as u128) / (count - 1) as u128) as usize);
    }
    out.dedup();
    out
}

#[inline]
fn bit(words: &[u64], index: usize) -> bool {
    ((words[index / 64] >> (index % 64)) & 1) != 0
}

#[inline]
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (left, right) in dst.iter_mut().zip(src) {
        *left ^= *right;
    }
}

fn first_set(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .enumerate()
        .find_map(|(wi, &word)| (word != 0).then_some(wi * 64 + word.trailing_zeros() as usize))
}

fn popcount_xor(left: &[u64], right: &[u64]) -> u64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a ^ b).count_ones() as u64)
        .sum()
}

fn popcount(words: &[u64]) -> u64 {
    words.iter().map(|word| word.count_ones() as u64).sum()
}

fn firing_into(gate: &XGate, state: &[Vec<u64>], out: &mut [u64]) {
    out.fill(!0u64);
    for &(wire, positive) in &gate.ctrls {
        for (dst, &value) in out.iter_mut().zip(&state[wire as usize]) {
            *dst &= if positive { value } else { !value };
        }
    }
    if gate.comp {
        for word in out {
            *word = !*word;
        }
    }
}

fn apply_transposed(gate: &XGate, state: &mut [Vec<u64>], scratch: &mut [u64]) {
    firing_into(gate, state, scratch);
    xor_into(&mut state[gate.target as usize], scratch);
}

struct Target {
    residual_fit: Vec<u64>,
    predicted_validation: Vec<u64>,
    predicted_test: Vec<u64>,
    actual_validation: Vec<u64>,
    actual_test: Vec<u64>,
}

struct BasisRow {
    fit: Vec<u64>,
    validation: Vec<u64>,
    test: Vec<u64>,
}

struct Basis {
    rows: Vec<Option<BasisRow>>,
    rank: usize,
    candidates: usize,
    dependent: usize,
    dependency_validation_disagreements: usize,
    fit_words: usize,
    validation_words: usize,
    test_words: usize,
    scratch_fit: Vec<u64>,
    scratch_validation: Vec<u64>,
    scratch_test: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Offer {
    Independent,
    Dependent,
}

impl Basis {
    fn new(fit_words: usize, validation_words: usize, test_words: usize) -> Self {
        let fit_samples = fit_words * 64;
        Self {
            rows: std::iter::repeat_with(|| None).take(fit_samples).collect(),
            rank: 0,
            candidates: 0,
            dependent: 0,
            dependency_validation_disagreements: 0,
            fit_words,
            validation_words,
            test_words,
            scratch_fit: vec![0; fit_words],
            scratch_validation: vec![0; validation_words],
            scratch_test: vec![0; test_words],
        }
    }

    fn saturated(&self) -> bool {
        self.rank == self.fit_words * 64
    }

    /// Offer one full signature. Locked-test words travel with algebraic row
    /// operations but never influence pivoting, acceptance, or model choice.
    fn offer(&mut self, signature: &[u64], targets: &mut [Target]) -> Offer {
        debug_assert_eq!(
            signature.len(),
            self.fit_words + self.validation_words + self.test_words
        );
        self.candidates += 1;
        let (fit, rest) = signature.split_at(self.fit_words);
        let (validation, test) = rest.split_at(self.validation_words);
        self.scratch_fit.copy_from_slice(fit);
        self.scratch_validation.copy_from_slice(validation);
        self.scratch_test.copy_from_slice(test);

        loop {
            let Some(pivot) = first_set(&self.scratch_fit) else {
                self.dependent += 1;
                if self.scratch_validation.iter().any(|&word| word != 0) {
                    self.dependency_validation_disagreements += 1;
                }
                return Offer::Dependent;
            };
            if let Some(row) = self.rows[pivot].as_ref() {
                xor_into(&mut self.scratch_fit, &row.fit);
                xor_into(&mut self.scratch_validation, &row.validation);
                xor_into(&mut self.scratch_test, &row.test);
                continue;
            }

            // `pivot` is the first free coordinate, but pivots are not
            // necessarily discovered in increasing order.  Eliminate every
            // already-present *later* pivot before storing this row.  Without
            // this reduced form, updating a target with the new row can
            // reintroduce an older pivot that the target had already cleared.
            for later in pivot + 1..self.rows.len() {
                if bit(&self.scratch_fit, later)
                    && let Some(row) = self.rows[later].as_ref()
                {
                    xor_into(&mut self.scratch_fit, &row.fit);
                    xor_into(&mut self.scratch_validation, &row.validation);
                    xor_into(&mut self.scratch_test, &row.test);
                }
            }

            let row = BasisRow {
                fit: std::mem::replace(&mut self.scratch_fit, vec![0; self.fit_words]),
                validation: std::mem::replace(
                    &mut self.scratch_validation,
                    vec![0; self.validation_words],
                ),
                test: std::mem::replace(&mut self.scratch_test, vec![0; self.test_words]),
            };
            self.rows[pivot] = Some(row);
            self.rank += 1;
            let row = self.rows[pivot].as_ref().unwrap();
            for target in targets {
                if bit(&target.residual_fit, pivot) {
                    xor_into(&mut target.residual_fit, &row.fit);
                    xor_into(&mut target.predicted_validation, &row.validation);
                    xor_into(&mut target.predicted_test, &row.test);
                }
            }
            return Offer::Independent;
        }
    }
}

fn collect_source_snapshots(
    gates: &[XGate],
    nw: usize,
    n: usize,
    initial_x: &[Vec<u64>],
    indices: &[usize],
) -> Vec<Vec<Vec<u64>>> {
    let words = initial_x.first().map_or(0, Vec::len);
    let mut state = vec![vec![0u64; words]; nw.max(n)];
    for (dst, src) in state.iter_mut().take(n).zip(initial_x) {
        dst.copy_from_slice(src);
    }
    let mut scratch = vec![0u64; words];
    let mut snapshots = Vec::with_capacity(indices.len());
    let mut next = 0usize;
    for pos in 0..=gates.len() {
        while next < indices.len() && indices[next] == pos {
            snapshots.push(state[..n].to_vec());
            next += 1;
        }
        if pos < gates.len() {
            apply_transposed(&gates[pos], &mut state, &mut scratch);
        }
    }
    snapshots
}

fn make_targets(
    snapshots: &[Vec<Vec<u64>>],
    fit_words: usize,
    validation_words: usize,
) -> Vec<Target> {
    let mut targets = Vec::new();
    for row in snapshots {
        for signature in row {
            let (fit, rest) = signature.split_at(fit_words);
            let (validation, test) = rest.split_at(validation_words);
            targets.push(Target {
                residual_fit: fit.to_vec(),
                predicted_validation: vec![0; validation_words],
                predicted_test: vec![0; test.len()],
                actual_validation: validation.to_vec(),
                actual_test: test.to_vec(),
            });
        }
    }
    targets
}

fn selected_witness_error(target: &Target) -> f64 {
    let validation_bits = (target.actual_validation.len() * 64) as u64;
    let test_bits = (target.actual_test.len() * 64) as u64;
    let direct_validation = popcount_xor(&target.predicted_validation, &target.actual_validation);
    let flip_witness = direct_validation > validation_bits / 2;
    let witness_validation = direct_validation.min(validation_bits - direct_validation);

    let ones_validation = popcount(&target.actual_validation);
    let constant_one = ones_validation > validation_bits / 2;
    let constant_validation = ones_validation.min(validation_bits - ones_validation);

    if witness_validation <= constant_validation {
        let direct_test = popcount_xor(&target.predicted_test, &target.actual_test);
        let errors = if flip_witness {
            test_bits - direct_test
        } else {
            direct_test
        };
        errors as f64 / test_bits as f64
    } else {
        let ones_test = popcount(&target.actual_test);
        let errors = if constant_one {
            test_bits - ones_test
        } else {
            ones_test
        };
        errors as f64 / test_bits as f64
    }
}

fn score_targets(targets: &[Target], rows: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut exact = vec![0f32; rows];
    let mut witness = vec![0f32; rows];
    let mut exact_target_indices = Vec::new();
    for row in 0..rows {
        let mut exact_sum = 0f64;
        let mut witness_sum = 0f64;
        for (bit_index, target) in targets[row * n..(row + 1) * n].iter().enumerate() {
            let consistent_fit = first_set(&target.residual_fit).is_none();
            let validation_errors =
                popcount_xor(&target.predicted_validation, &target.actual_validation);
            let exact_error = if consistent_fit && validation_errors == 0 {
                let test_errors = popcount_xor(&target.predicted_test, &target.actual_test);
                if test_errors == 0 {
                    exact_target_indices.push(row * n + bit_index);
                }
                test_errors as f64 / (target.actual_test.len() * 64) as f64
            } else {
                0.5
            };
            exact_sum += exact_error;
            witness_sum += selected_witness_error(target);
        }
        exact[row] = (exact_sum / n as f64) as f32;
        witness[row] = (witness_sum / n as f64) as f32;
    }
    (exact, witness, exact_target_indices)
}

struct RunResult {
    exact: Vec<f32>,
    witness: Vec<f32>,
    ranks: Vec<usize>,
    candidates: Vec<usize>,
    dependency_disagreements: Vec<usize>,
    exact_bits: Vec<usize>,
    exact_target_indices: Vec<Vec<usize>>,
    saturated_at: Option<usize>,
    skipped_after_saturation: usize,
}

fn record_column(rows: usize, n: usize, basis: &Basis, targets: &[Target], result: &mut RunResult) {
    let (exact, witness, exact_target_indices) = score_targets(targets, rows, n);
    result.exact.extend(exact);
    result.witness.extend(witness);
    result.ranks.push(basis.rank);
    result.candidates.push(basis.candidates);
    result
        .dependency_disagreements
        .push(basis.dependency_validation_disagreements);
    result.exact_bits.push(exact_target_indices.len());
    result.exact_target_indices.push(exact_target_indices);
}

fn run_attack(
    args: &Args,
    g: &[XGate],
    nw_g: usize,
    initial_x: &[Vec<u64>],
    g_indices: &[usize],
    targets: &mut [Target],
    rows: usize,
) -> RunResult {
    let total_words = args.fit_batches + args.validation_batches + args.test_batches;
    let mut state = vec![vec![0u64; total_words]; nw_g.max(args.n)];
    for (dst, src) in state.iter_mut().take(args.n).zip(initial_x) {
        dst.copy_from_slice(src);
    }
    let mut basis = Basis::new(args.fit_batches, args.validation_batches, args.test_batches);
    let mut result = RunResult {
        exact: Vec::with_capacity(rows * g_indices.len()),
        witness: Vec::with_capacity(rows * g_indices.len()),
        ranks: Vec::with_capacity(g_indices.len()),
        candidates: Vec::with_capacity(g_indices.len()),
        dependency_disagreements: Vec::with_capacity(g_indices.len()),
        exact_bits: Vec::with_capacity(g_indices.len()),
        exact_target_indices: Vec::with_capacity(g_indices.len()),
        saturated_at: None,
        skipped_after_saturation: 0,
    };

    let constant = vec![!0u64; total_words];
    basis.offer(&constant, targets);
    let mut scratch = vec![0u64; total_words];

    match args.trace_mode {
        TraceMode::CheckpointState => {
            let mut gate_pos = 0usize;
            for (column, &checkpoint) in g_indices.iter().enumerate() {
                while gate_pos < checkpoint {
                    apply_transposed(&g[gate_pos], &mut state, &mut scratch);
                    gate_pos += 1;
                }
                for wire in &state {
                    basis.offer(wire, targets);
                }
                record_column(rows, args.n, &basis, targets, &mut result);
                println!(
                    "[hmap_trace_affine] col {}/{} G={} candidates={} rank={} exact_bits={}/{} dep_disagree={}",
                    column + 1,
                    g_indices.len(),
                    checkpoint,
                    basis.candidates,
                    basis.rank,
                    result.exact_bits.last().unwrap(),
                    rows * args.n,
                    basis.dependency_validation_disagreements,
                );
            }
        }
        TraceMode::GateDelta => {
            for wire in &state {
                basis.offer(wire, targets);
            }
            let mut next_column = 0usize;
            while next_column < g_indices.len() && g_indices[next_column] == 0 {
                record_column(rows, args.n, &basis, targets, &mut result);
                next_column += 1;
            }
            for (gate_index, gate) in g.iter().enumerate() {
                firing_into(gate, &state, &mut scratch);
                let selected =
                    gate_index % args.delta_stride == args.delta_offset.min(args.delta_stride - 1);
                if selected {
                    if basis.saturated() && args.stop_at_saturation {
                        result.saturated_at.get_or_insert(gate_index);
                        result.skipped_after_saturation += 1;
                    } else {
                        basis.offer(&scratch, targets);
                        if basis.saturated() {
                            result.saturated_at.get_or_insert(gate_index + 1);
                        }
                    }
                }
                xor_into(&mut state[gate.target as usize], &scratch);
                let pos = gate_index + 1;
                while next_column < g_indices.len() && g_indices[next_column] == pos {
                    if args.include_checkpoint_states {
                        for wire in &state {
                            basis.offer(wire, targets);
                        }
                    }
                    record_column(rows, args.n, &basis, targets, &mut result);
                    println!(
                        "[hmap_trace_affine] col {}/{} G={} candidates={} rank={} exact_bits={}/{} saturated_at={:?}",
                        next_column + 1,
                        g_indices.len(),
                        pos,
                        basis.candidates,
                        basis.rank,
                        result.exact_bits.last().unwrap(),
                        rows * args.n,
                        result.saturated_at,
                    );
                    next_column += 1;
                }
            }
        }
    }
    result
}

fn transpose_columns(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut out = vec![0f32; values.len()];
    for col in 0..cols {
        for row in 0..rows {
            out[row * cols + col] = values[col * rows + row];
        }
    }
    out
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_nested_usize(values: &[Vec<usize>]) -> String {
    values
        .iter()
        .map(|column| format!("[{}]", join_usize(column)))
        .collect::<Vec<_>>()
        .join(",")
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&value| value as f64).sum::<f64>() / values.len().max(1) as f64
}

struct Meta<'a> {
    args: &'a Args,
    rows: usize,
    cols: usize,
    i_idx: &'a [usize],
    j_idx: &'a [usize],
    nw_g: usize,
    score: &'a str,
    mu: f64,
    result: &'a RunResult,
    final_candidate_budget: usize,
    overdetermined: bool,
}

fn write_plate(stem: &str, matrix: &[f32], meta: &Meta<'_>) {
    let mut file = std::fs::File::create(format!("{stem}.bin")).expect("create plate bin");
    let bytes: Vec<u8> = matrix
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    file.write_all(&bytes).expect("write plate bin");
    let saturated = meta
        .result
        .saturated_at
        .map_or_else(|| "null".to_string(), |value| value.to_string());
    let json = format!(
        concat!(
            "{{\"rows\":{},\"cols\":{},\"n\":{},\"target_count\":{},",
            "\"mode\":\"cumulative\",\"trace_mode\":\"{}\",\"score\":\"{}\",",
            "\"mu\":{:.8},\"g_wires\":{},\"fit_samples\":{},",
            "\"validation_samples\":{},\"test_samples\":{},",
            "\"feature_budget_final\":{},\"overdetermined\":{},",
            "\"delta_stride\":{},\"delta_offset\":{},\"include_checkpoint_states\":{},",
            "\"saturated_at\":{},\"skipped_after_saturation\":{},",
            "\"i_idx\":[{}],\"j_idx\":[{}],\"rank_by_col\":[{}],",
            "\"candidates_by_col\":[{}],\"dependency_disagreements_by_col\":[{}],",
            "\"exact_bits_by_col\":[{}],\"exact_target_indices_by_col\":[{}]}}"
        ),
        meta.rows,
        meta.cols,
        meta.args.n,
        meta.args.n,
        meta.args.trace_mode.name(),
        meta.score,
        meta.mu,
        meta.nw_g,
        meta.args.fit_batches * 64,
        meta.args.validation_batches * 64,
        meta.args.test_batches * 64,
        meta.final_candidate_budget,
        meta.overdetermined,
        meta.args.delta_stride,
        meta.args.delta_offset,
        meta.args.include_checkpoint_states,
        saturated,
        meta.result.skipped_after_saturation,
        join_usize(meta.i_idx),
        join_usize(meta.j_idx),
        join_usize(&meta.result.ranks),
        join_usize(&meta.result.candidates),
        join_usize(&meta.result.dependency_disagreements),
        join_usize(&meta.result.exact_bits),
        join_nested_usize(&meta.result.exact_target_indices),
    );
    std::fs::write(format!("{stem}.meta.json"), json).expect("write plate metadata");
}

fn main() {
    let args = Args::parse();
    assert!(args.n > 0, "--n must be positive");
    assert!(args.fit_batches > 0, "--fit-batches must be positive");
    assert!(
        args.validation_batches > 0,
        "--validation-batches must be positive"
    );
    assert!(args.test_batches > 0, "--test-batches must be positive");
    assert!(args.delta_stride > 0, "--delta-stride must be positive");
    assert!(
        args.delta_offset < args.delta_stride,
        "--delta-offset must be smaller than --delta-stride"
    );

    let (c, declared_c) = read_circuit(&args.c, &args.c_format);
    let (g_all, declared_g) = read_circuit(&args.g, &args.g_format);
    let g_len = if args.g_limit == 0 {
        g_all.len()
    } else {
        args.g_limit.min(g_all.len())
    };
    let g = &g_all[..g_len];
    let nw_c = declared_c.max(max_wire(&c) as usize + 1).max(args.n);
    let nw_g = declared_g.max(max_wire(g) as usize + 1).max(args.n);
    let i_idx = step_indices(c.len(), args.c_step);
    let j_idx = even_indices(g.len(), args.g_checkpoints);
    let rows = i_idx.len();
    let cols = j_idx.len();
    let final_candidate_budget = match args.trace_mode {
        TraceMode::CheckpointState => 1 + cols * nw_g,
        TraceMode::GateDelta => {
            let selected_deltas = if args.delta_offset >= g.len() {
                0
            } else {
                1 + (g.len() - 1 - args.delta_offset) / args.delta_stride
            };
            let checkpoint_wires = if args.include_checkpoint_states {
                (cols - 1) * nw_g
            } else {
                0
            };
            1 + nw_g + selected_deltas + checkpoint_wires
        }
    };
    let fit_samples = args.fit_batches * 64;
    let overdetermined = fit_samples > final_candidate_budget + 64;
    if !overdetermined && !args.allow_underdetermined {
        panic!(
            "fit samples ({fit_samples}) must exceed candidate features ({final_candidate_budget}) by >64; raise --fit-batches, lower --g-checkpoints, or explicitly pass --allow-underdetermined"
        );
    }
    if !overdetermined {
        eprintln!(
            "[hmap_trace_affine] WARNING: underdetermined feature set (fit samples {fit_samples}, candidates {final_candidate_budget}); positive locked-test witnesses are evidence, saturated negatives are inconclusive"
        );
    }

    let total_words = args.fit_batches + args.validation_batches + args.test_batches;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let initial_x: Vec<Vec<u64>> = (0..args.n)
        .map(|_| (0..total_words).map(|_| rng.random::<u64>()).collect())
        .collect();
    let source = collect_source_snapshots(&c, nw_c, args.n, &initial_x, &i_idx);
    let mut targets = make_targets(&source, args.fit_batches, args.validation_batches);

    println!(
        "[hmap_trace_affine] C={} gates/{} wires rows={}; G={} gates/{} wires cols={}; mode={}; samples fit/validation/test={}/{}/{}; feature budget={} ({})",
        c.len(),
        nw_c,
        rows,
        g.len(),
        nw_g,
        cols,
        args.trace_mode.name(),
        args.fit_batches * 64,
        args.validation_batches * 64,
        args.test_batches * 64,
        final_candidate_budget,
        if overdetermined {
            "overdetermined"
        } else {
            "UNDERDETERMINED"
        },
    );

    let result = run_attack(&args, g, nw_g, &initial_x, &j_idx, &mut targets, rows);
    let exact = transpose_columns(&result.exact, rows, cols);
    let witness = transpose_columns(&result.witness, rows, cols);
    let exact_mu = mean(&exact);
    let witness_mu = mean(&witness);
    let meta = Meta {
        args: &args,
        rows,
        cols,
        i_idx: &i_idx,
        j_idx: &j_idx,
        nw_g,
        score: "exact-span",
        mu: exact_mu,
        result: &result,
        final_candidate_budget,
        overdetermined,
    };
    write_plate(&args.out, &exact, &meta);
    let witness_stem = format!("{}.witness", args.out);
    let witness_meta = Meta {
        score: "affine-witness",
        mu: witness_mu,
        ..meta
    };
    write_plate(&witness_stem, &witness, &witness_meta);
    println!(
        "[hmap_trace_affine] wrote {}{{,.witness}}.bin/.meta.json; mean H exact={exact_mu:.5} witness={witness_mu:.5}; final rank={}/{}; saturation={:?}",
        args.out,
        result.ranks.last().copied().unwrap_or(0),
        fit_samples,
        result.saturated_at,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature_from_truth(n: usize, function: impl Fn(usize) -> bool) -> Vec<u64> {
        let samples = 1usize << n;
        let mut out = vec![0u64; samples.div_ceil(64)];
        for x in 0..samples {
            if function(x) {
                out[x / 64] |= 1u64 << (x % 64);
            }
        }
        out
    }

    fn span_rank(mut features: Vec<Vec<u64>>) -> usize {
        let samples = features.first().map_or(0, |f| f.len() * 64);
        let mut rows: Vec<Option<Vec<u64>>> =
            std::iter::repeat_with(|| None).take(samples).collect();
        let mut rank = 0;
        for feature in &mut features {
            loop {
                let Some(pivot) = first_set(feature) else {
                    break;
                };
                if let Some(row) = rows[pivot].as_ref() {
                    xor_into(feature, row);
                } else {
                    rows[pivot] = Some(feature.clone());
                    rank += 1;
                    break;
                }
            }
        }
        rank
    }

    fn in_span(features: &[Vec<u64>], target: &[u64]) -> bool {
        span_rank(features.to_vec()) == {
            let mut extended = features.to_vec();
            extended.push(target.to_vec());
            span_rank(extended)
        }
    }

    #[test]
    fn initial_plus_deltas_has_same_span_as_every_snapshot() {
        let n = 3;
        let x0 = signature_from_truth(n, |x| x & 1 != 0);
        let x1 = signature_from_truth(n, |x| x & 2 != 0);
        let x2 = signature_from_truth(n, |x| x & 4 != 0);
        let zero = vec![0u64; 1];
        let mut state = vec![x0, x1, x2, zero];
        let gates = vec![
            XGate::conj(3, [(0, true), (1, false)]).unwrap(),
            XGate {
                target: 2,
                comp: true,
                ctrls: [(0, true), (1, true)].into_iter().collect(),
            },
            XGate::x_gate(1),
            XGate::cnot(3, 2),
        ];
        let constant = vec![!0u64; 1];
        let mut explicit = vec![constant.clone()];
        explicit.extend(state.clone());
        let mut delta = vec![constant];
        delta.extend(state.clone());
        let mut scratch = vec![0u64; 1];
        for gate in &gates {
            firing_into(gate, &state, &mut scratch);
            delta.push(scratch.clone());
            xor_into(&mut state[gate.target as usize], &scratch);
            explicit.extend(state.clone());
        }
        assert_eq!(span_rank(explicit), span_rank(delta));
    }

    #[test]
    fn two_trace_times_recover_relation_unavailable_at_either_time() {
        let n = 3;
        let x0 = signature_from_truth(n, |x| x & 1 != 0);
        let x1 = signature_from_truth(n, |x| x & 2 != 0);
        let x2 = signature_from_truth(n, |x| x & 4 != 0);
        let mut state = vec![x0, x1, x2, vec![0u64; 1]];
        let mut scratch = vec![0u64; 1];
        let first = XGate::conj(3, [(0, true), (2, true)]).unwrap();
        apply_transposed(&first, &mut state, &mut scratch);
        let snapshot_one = state.clone();
        let second = XGate::conj(3, [(0, true), (1, true)]).unwrap();
        apply_transposed(&second, &mut state, &mut scratch);
        let snapshot_two = state.clone();
        let target = signature_from_truth(n, |x| (x & 1 != 0) && (x & 2 != 0));
        let constant = vec![!0u64; 1];
        let mut one = vec![constant.clone()];
        one.extend(snapshot_one.clone());
        let mut two = vec![constant.clone()];
        two.extend(snapshot_two.clone());
        assert!(!in_span(&one, &target));
        assert!(!in_span(&two, &target));
        let mut joint = vec![constant];
        joint.extend(snapshot_one);
        joint.extend(snapshot_two);
        assert!(in_span(&joint, &target));
    }

    #[test]
    fn incremental_targets_recover_every_offered_signature() {
        let (fit_words, validation_words, test_words) = (8, 2, 3);
        let total_words = fit_words + validation_words + test_words;
        let mut rng = StdRng::seed_from_u64(7);
        let signatures: Vec<Vec<u64>> = (0..128)
            .map(|_| (0..total_words).map(|_| rng.random::<u64>()).collect())
            .collect();
        let snapshots = vec![signatures.clone()];
        let mut targets = make_targets(&snapshots, fit_words, validation_words);
        let mut basis = Basis::new(fit_words, validation_words, test_words);
        basis.offer(&vec![!0u64; total_words], &mut targets);
        for signature in &signatures {
            basis.offer(signature, &mut targets);
        }
        for target in &targets {
            assert!(first_set(&target.residual_fit).is_none());
            assert_eq!(
                popcount_xor(&target.predicted_validation, &target.actual_validation),
                0
            );
            assert_eq!(popcount_xor(&target.predicted_test, &target.actual_test), 0);
        }
    }
}
