//! Held-out statistical companion to `hmap_affine`.
//!
//! For every source-prefix / gadget-prefix cell and logical target bit, this
//! reader selects the best constant, single-wire, or XOR-of-two-wires predictor
//! on training lanes. The candidate and its complement bit are then frozen and
//! scored on disjoint holdout lanes. The primary matrix is the mean holdout
//! agreement across the selected target bits.
//!
//! A second full matrix repeats the same train/freeze/holdout procedure against
//! independent random targets. It is an honest max-search baseline, not a
//! permutation p-value. The reported interior chronology uses tied-average
//! peak columns and all nonzero circular shifts, preserving ordered-prefix
//! serial structure. Flat rows are excluded, and chronology remains undefined
//! unless a declared minimum count and fraction of core rows are informative.
//!
//! This is the active, fixed-slice-capable continuation of the reader first
//! committed as `src/bin/hmap_stat.rs` at `2a807d28`.

use clap::Parser;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;

const HELPER_RNG_DOMAIN: u64 = 0x4845_4c50_4552_5f34;
const TARGET_RNG_DOMAIN: u64 = 0x5441_5247_4554_5f53;
const NULL_RNG_DOMAIN: u64 = 0x4e55_4c4c_5f53_5441;

#[derive(Parser, Debug)]
#[command(name = "hmap_stat")]
struct Args {
    /// Original/reference circuit C.
    #[arg(long)]
    c: String,
    /// Gadgetized/mixed circuit G.
    #[arg(long)]
    g: String,
    #[arg(long, default_value = "g57")]
    c_format: String,
    #[arg(long, default_value = "mpmct1")]
    g_format: String,
    /// Logical input width. X is shared on wires 0..n of C and G.
    #[arg(long)]
    n: usize,
    /// Prefix strides.
    #[arg(long, default_value_t = 200)]
    c_step: usize,
    #[arg(long, default_value_t = 20_000)]
    g_step: usize,
    /// Interior prefix fractions used for the reported ridge summary.
    #[arg(long, default_value_t = 0.20)]
    core_low: f64,
    #[arg(long, default_value_t = 0.80)]
    core_high: f64,
    /// Minimum informative core rows required before reporting chronology.
    /// Twenty rows are needed for a circular-shift p-value resolution of 0.05.
    #[arg(long, default_value_t = 20)]
    ridge_min_informative_rows: usize,
    /// Minimum fraction of core rows whose peak exceeds their row median by
    /// one selected-target-bit equivalent (0.5 / selected target-bit count).
    #[arg(long, default_value_t = 0.25)]
    ridge_min_coverage: f64,
    /// Absolute tolerance for peak ties and the one-bit prominence threshold.
    #[arg(long, default_value_t = 1e-7)]
    ridge_tie_atol: f64,
    /// Requested total samples. The actual count is rounded up to 64-lane
    /// batches and is recorded in metadata.
    #[arg(long, default_value_t = 4096)]
    samples: usize,
    /// Requested training samples. Zero selects 75% of the actual batches.
    /// The remainder is a disjoint holdout set.
    #[arg(long, default_value_t = 0)]
    train_samples: usize,
    /// Number of deterministically sampled target bits. Zero (the default)
    /// means all n logical target bits.
    #[arg(long, default_value_t = 0)]
    target_bits: usize,
    /// Predictor-wire set, for example "0-511,520". Empty means every G wire.
    #[arg(long, default_value = "")]
    wire_list: String,
    /// Public value for G wires n..2n. Supply together with --fixed-z.
    #[arg(long, value_parser = parse_u128)]
    fixed_y: Option<u128>,
    /// Public value for G wires 2n..3n. Supply together with --fixed-y.
    #[arg(long, value_parser = parse_u128)]
    fixed_z: Option<u128>,
    /// First wire in an independently randomized helper suffix. The suffix is
    /// START..G_NUM_WIRES and uses a separate RNG stream, so changing helper
    /// width cannot change the shared X samples.
    #[arg(long)]
    random_helper_start: Option<usize>,
    /// Sampling seed. X, helper, target-subset, and null streams are
    /// domain-separated from this value.
    #[arg(long, default_value_t = 12_345)]
    seed: u64,
    /// Optional experiment label, such as "matched" or "mismatched-reference".
    #[arg(long, default_value = "")]
    run_label: String,
    /// Output stem. Writes .bin, .null.bin, .winners.jsonl, and .meta.json.
    #[arg(long)]
    out: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    Constant,
    Single(usize),
    Pair(usize, usize),
}

#[derive(Clone, Copy, Debug)]
struct Winner {
    candidate: Candidate,
    complemented: bool,
    train_agreement: f64,
}

struct LoadedCircuit {
    gates: Vec<XGate>,
    num_wires: usize,
    inferred_num_wires: usize,
    declared_num_wires: Option<usize>,
}

fn parse_u128(value: &str) -> Result<u128, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u128::from_str_radix(hex, 16).map_err(|error| format!("invalid hex u128: {error}"))
    } else {
        value
            .parse::<u128>()
            .map_err(|error| format!("invalid decimal u128: {error}"))
    }
}

fn validate_fixed_value(value: u128, n: usize, name: &str) -> Result<(), String> {
    if n < 128 && value >= (1u128 << n) {
        Err(format!(
            "--fixed-{name} value {value:#x} does not fit the declared n={n} bits"
        ))
    } else {
        Ok(())
    }
}

fn indices(end: usize, step: usize) -> Vec<usize> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < end {
        output.push(index);
        index += step;
    }
    output.push(end);
    output
}

fn snapshots(gates: &[XGate], initial: &[u64], indices: &[usize]) -> Vec<Vec<u64>> {
    let mut state = initial.to_vec();
    let mut output = Vec::with_capacity(indices.len());
    let mut next = 0;
    for position in 0..=gates.len() {
        while next < indices.len() && indices[next] == position {
            output.push(state.clone());
            next += 1;
        }
        if position < gates.len() {
            gates[position].apply_lanes(&mut state);
        }
    }
    output
}

fn load_circuit(path: &str, format: &str, n: usize, label: &str) -> LoadedCircuit {
    let (gates, declared_num_wires) = match format {
        "g57" => (
            read_g57_file(path).unwrap_or_else(|error| panic!("read {label} (g57): {error}")),
            None,
        ),
        "mpmct1" => {
            let (gates, declared) =
                read_mpmct(path).unwrap_or_else(|error| panic!("read {label} (mpmct1): {error}"));
            (gates, Some(declared))
        }
        other => panic!("unknown --{label}-format {other}"),
    };
    let inferred_num_wires = (max_wire(&gates) as usize + 1).max(n);
    let num_wires = if let Some(declared) = declared_num_wires {
        assert!(
            declared >= inferred_num_wires,
            "{label} mpmct1 header declares {declared} wires, but its gates/X input use at least \
             {inferred_num_wires}"
        );
        declared
    } else {
        inferred_num_wires
    };
    LoadedCircuit {
        gates,
        num_wires,
        inferred_num_wires,
        declared_num_wires,
    }
}

fn parse_wire_list(specification: &str, num_wires: usize) -> Result<Vec<usize>, String> {
    if num_wires == 0 {
        return Err("cannot select wires from an empty circuit".to_owned());
    }
    if specification.trim().is_empty() {
        return Ok((0..num_wires).collect());
    }

    let mut wires = Vec::new();
    for raw_part in specification.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            return Err("wire list contains an empty entry".to_owned());
        }
        if let Some((low, high)) = part.split_once('-') {
            if high.contains('-') {
                return Err(format!("invalid wire range {part:?}"));
            }
            let low: usize = low
                .trim()
                .parse()
                .map_err(|_| format!("invalid wire range start in {part:?}"))?;
            let high: usize = high
                .trim()
                .parse()
                .map_err(|_| format!("invalid wire range end in {part:?}"))?;
            if low > high {
                return Err(format!("wire range starts after it ends: {part:?}"));
            }
            if high >= num_wires {
                return Err(format!(
                    "wire range {part:?} exceeds valid wires 0..{}",
                    num_wires - 1
                ));
            }
            wires.extend(low..=high);
        } else {
            let wire: usize = part
                .parse()
                .map_err(|_| format!("invalid wire entry {part:?}"))?;
            if wire >= num_wires {
                return Err(format!(
                    "wire {wire} exceeds valid wires 0..{}",
                    num_wires - 1
                ));
            }
            wires.push(wire);
        }
    }
    wires.sort_unstable();
    wires.dedup();
    if wires.is_empty() {
        return Err("wire list selects no wires".to_owned());
    }
    Ok(wires)
}

fn sample_batch_inputs(
    n: usize,
    c_wires: usize,
    g_wires: usize,
    fixed_y: Option<u128>,
    fixed_z: Option<u128>,
    random_helper_start: Option<usize>,
    x_rng: &mut StdRng,
    helper_rng: &mut StdRng,
) -> (Vec<u64>, Vec<u64>) {
    let x: Vec<u64> = (0..n).map(|_| x_rng.random::<u64>()).collect();
    let mut c_input = vec![0u64; c_wires];
    let mut g_input = vec![0u64; g_wires];
    c_input[..n].copy_from_slice(&x);
    g_input[..n].copy_from_slice(&x);

    if let (Some(y), Some(z)) = (fixed_y, fixed_z) {
        for bit in 0..n {
            g_input[n + bit] = if (y >> bit) & 1 == 1 { !0 } else { 0 };
            g_input[2 * n + bit] = if (z >> bit) & 1 == 1 { !0 } else { 0 };
        }
    }
    if let Some(start) = random_helper_start {
        for wire in &mut g_input[start..] {
            *wire = helper_rng.random::<u64>();
        }
    }
    (c_input, g_input)
}

fn choose_target_bits(n: usize, count: usize, rng: &mut StdRng) -> Vec<usize> {
    if count == 0 || count >= n {
        return (0..n).collect();
    }
    let mut targets: Vec<usize> = (0..n).collect();
    for index in 0..count {
        let selected = index + rng.random_range(0..n - index);
        targets.swap(index, selected);
    }
    targets.truncate(count);
    targets.sort_unstable();
    targets
}

fn mismatch_count(
    target: &[u64],
    wires: &[Vec<u64>],
    candidate: Candidate,
    batches: Range<usize>,
) -> u64 {
    batches
        .map(|batch| {
            let prediction = match candidate {
                Candidate::Constant => 0,
                Candidate::Single(wire) => wires[wire][batch],
                Candidate::Pair(first, second) => wires[first][batch] ^ wires[second][batch],
            };
            (prediction ^ target[batch]).count_ones() as u64
        })
        .sum()
}

fn fit_candidate(
    target: &[u64],
    wires: &[Vec<u64>],
    candidate: Candidate,
    batches: Range<usize>,
) -> Winner {
    let samples = 64 * batches.len() as u64;
    let mismatch = mismatch_count(target, wires, candidate, batches);
    let complemented = 2 * mismatch > samples;
    let matches = if complemented {
        mismatch
    } else {
        samples - mismatch
    };
    Winner {
        candidate,
        complemented,
        train_agreement: matches as f64 / samples as f64,
    }
}

fn select_candidate(target: &[u64], wires: &[Vec<u64>], train_batches: Range<usize>) -> Winner {
    let mut best = fit_candidate(target, wires, Candidate::Constant, train_batches.clone());
    for wire in 0..wires.len() {
        let candidate = fit_candidate(
            target,
            wires,
            Candidate::Single(wire),
            train_batches.clone(),
        );
        if candidate.train_agreement > best.train_agreement {
            best = candidate;
        }
    }
    for first in 0..wires.len() {
        for second in first + 1..wires.len() {
            let candidate = fit_candidate(
                target,
                wires,
                Candidate::Pair(first, second),
                train_batches.clone(),
            );
            if candidate.train_agreement > best.train_agreement {
                best = candidate;
            }
        }
    }
    best
}

fn score_frozen_candidate(
    target: &[u64],
    wires: &[Vec<u64>],
    winner: Winner,
    holdout_batches: Range<usize>,
) -> f64 {
    let samples = 64 * holdout_batches.len() as u64;
    let mismatch = mismatch_count(target, wires, winner.candidate, holdout_batches);
    let matches = if winner.complemented {
        mismatch
    } else {
        samples - mismatch
    };
    matches as f64 / samples as f64
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character < ' ' => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn json_usize_array(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| value.to_string())
}

fn json_optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), |value| format!("{value:.9}"))
}

fn write_f32_matrix(path: &str, matrix: &[f32]) {
    let file = File::create(path).unwrap_or_else(|error| panic!("create {path}: {error}"));
    let mut writer = BufWriter::new(file);
    for value in matrix {
        writer
            .write_all(&value.to_le_bytes())
            .unwrap_or_else(|error| panic!("write {path}: {error}"));
    }
    writer
        .flush()
        .unwrap_or_else(|error| panic!("flush {path}: {error}"));
}

fn bytes_xxh3_128(bytes: &[u8]) -> String {
    format!("{:032x}", xxhash_rust::xxh3::xxh3_128(bytes))
}

fn file_xxh3_128(path: &str) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path} for hash: {error}"));
    bytes_xxh3_128(&bytes)
}

fn mean_sigma(matrix: &[f32]) -> (f64, f64) {
    let count = matrix.len() as f64;
    let sum: f64 = matrix.iter().map(|&value| value as f64).sum();
    let sum_squares: f64 = matrix
        .iter()
        .map(|&value| (value as f64) * (value as f64))
        .sum();
    let mean = sum / count;
    let sigma = (sum_squares / count - mean * mean).max(0.0).sqrt();
    (mean, sigma)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = 0.5 * (start + end - 1) as f64;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let left = left - left_mean;
        let right = right - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    (left_variance > 0.0 && right_variance > 0.0)
        .then(|| covariance / (left_variance * right_variance).sqrt())
}

fn spearman_ties(left: &[f64], right: &[f64]) -> Option<f64> {
    pearson(&average_ranks(left), &average_ranks(right))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

#[derive(Debug)]
struct RidgeSummary {
    core_rows: usize,
    core_columns: usize,
    informative_rows: usize,
    informative_coverage: f64,
    required_informative_rows: usize,
    prominence_threshold: f64,
    median_prominence: Option<f64>,
    median_peak_agreement: Option<f64>,
    median_null_peak_agreement: Option<f64>,
    rho: Option<f64>,
    circular_shift_p: Option<f64>,
    circular_shifts: usize,
}

fn ridge_summary(
    matrix: &[f32],
    null_matrix: &[f32],
    rows: usize,
    columns: usize,
    c_prefixes: &[usize],
    g_prefixes: &[usize],
    averaged_target_bits: usize,
    core_low: f64,
    core_high: f64,
    min_informative_rows: usize,
    min_coverage: f64,
    tie_atol: f64,
) -> RidgeSummary {
    let unsupported = || RidgeSummary {
        core_rows: 0,
        core_columns: 0,
        informative_rows: 0,
        informative_coverage: 0.0,
        required_informative_rows: 0,
        prominence_threshold: if averaged_target_bits == 0 {
            0.0
        } else {
            0.5 / averaged_target_bits as f64
        },
        median_prominence: None,
        median_peak_agreement: None,
        median_null_peak_agreement: None,
        rho: None,
        circular_shift_p: None,
        circular_shifts: 0,
    };
    if rows != c_prefixes.len()
        || columns != g_prefixes.len()
        || matrix.len() != rows * columns
        || null_matrix.len() != rows * columns
        || averaged_target_bits == 0
        || c_prefixes.last().copied().unwrap_or(0) == 0
        || g_prefixes.last().copied().unwrap_or(0) == 0
    {
        return unsupported();
    }
    let c_end = *c_prefixes.last().unwrap() as f64;
    let g_end = *g_prefixes.last().unwrap() as f64;
    let core_rows: Vec<usize> = c_prefixes
        .iter()
        .enumerate()
        .filter_map(|(index, &prefix)| {
            let fraction = prefix as f64 / c_end;
            (fraction >= core_low && fraction <= core_high).then_some(index)
        })
        .collect();
    let core_columns: Vec<usize> = g_prefixes
        .iter()
        .enumerate()
        .filter_map(|(index, &prefix)| {
            let fraction = prefix as f64 / g_end;
            (fraction >= core_low && fraction <= core_high).then_some(index)
        })
        .collect();
    let required_informative_rows = min_informative_rows
        .max(3)
        .max((min_coverage * core_rows.len() as f64).ceil() as usize);
    let prominence_threshold = 0.5 / averaged_target_bits as f64;
    if core_rows.len() < 3 || core_columns.is_empty() {
        return RidgeSummary {
            core_rows: core_rows.len(),
            core_columns: core_columns.len(),
            required_informative_rows,
            prominence_threshold,
            ..unsupported()
        };
    }

    let mut informative_row_positions = Vec::with_capacity(core_rows.len());
    let mut informative_peak_locations = Vec::with_capacity(core_rows.len());
    let mut prominences = Vec::with_capacity(core_rows.len());
    let mut peaks = Vec::with_capacity(core_rows.len());
    let mut null_peaks = Vec::with_capacity(core_rows.len());
    for &row in &core_rows {
        let row_values: Vec<f64> = core_columns
            .iter()
            .map(|&column| matrix[row * columns + column] as f64)
            .collect();
        let peak = core_columns
            .iter()
            .map(|&column| matrix[row * columns + column] as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let prominence = peak - median(row_values).unwrap();
        let tied_locations: Vec<f64> = core_columns
            .iter()
            .filter(|&&column| ((matrix[row * columns + column] as f64) - peak).abs() <= tie_atol)
            .map(|&column| g_prefixes[column] as f64 / g_end)
            .collect();
        if prominence + tie_atol >= prominence_threshold {
            informative_row_positions.push(c_prefixes[row] as f64 / c_end);
            informative_peak_locations
                .push(tied_locations.iter().sum::<f64>() / tied_locations.len() as f64);
        }
        prominences.push(prominence);
        peaks.push(peak);
        null_peaks.push(
            core_columns
                .iter()
                .map(|&column| null_matrix[row * columns + column] as f64)
                .fold(f64::NEG_INFINITY, f64::max),
        );
    }

    let informative_rows = informative_row_positions.len();
    let informative_coverage = informative_rows as f64 / core_rows.len() as f64;
    let supported = informative_rows >= required_informative_rows;
    let rho = supported
        .then(|| spearman_ties(&informative_row_positions, &informative_peak_locations))
        .flatten();
    let mut shifted = Vec::new();
    if rho.is_some() {
        for shift in 1..informative_peak_locations.len() {
            let rotated: Vec<f64> = (0..informative_peak_locations.len())
                .map(|index| {
                    informative_peak_locations[(index + shift) % informative_peak_locations.len()]
                })
                .collect();
            if let Some(value) = spearman_ties(&informative_row_positions, &rotated) {
                shifted.push(value);
            }
        }
    }
    let circular_shift_p = rho.map(|observed| {
        (1 + shifted.iter().filter(|&&null| null >= observed).count()) as f64
            / (shifted.len() + 1) as f64
    });

    RidgeSummary {
        core_rows: core_rows.len(),
        core_columns: core_columns.len(),
        informative_rows,
        informative_coverage,
        required_informative_rows,
        prominence_threshold,
        median_prominence: median(prominences),
        median_peak_agreement: median(peaks),
        median_null_peak_agreement: median(null_peaks),
        rho,
        circular_shift_p,
        circular_shifts: shifted.len(),
    }
}

fn winner_json(
    row: usize,
    column: usize,
    c_prefix: usize,
    g_prefix: usize,
    target_bit: usize,
    predictor_wires: &[usize],
    winner: Winner,
    holdout_agreement: f64,
) -> String {
    let (kind, first, second) = match winner.candidate {
        Candidate::Constant => ("constant", "null".to_owned(), "null".to_owned()),
        Candidate::Single(wire) => (
            "single",
            predictor_wires[wire].to_string(),
            "null".to_owned(),
        ),
        Candidate::Pair(first, second) => (
            "xor_pair",
            predictor_wires[first].to_string(),
            predictor_wires[second].to_string(),
        ),
    };
    format!(
        "{{\"row\":{row},\"column\":{column},\"c_prefix\":{c_prefix},\
         \"g_prefix\":{g_prefix},\"target_bit\":{target_bit},\
         \"candidate_kind\":\"{kind}\",\"wire_a\":{first},\"wire_b\":{second},\
         \"complemented\":{},\"train_agreement\":{:.9},\
         \"holdout_agreement\":{holdout_agreement:.9}}}",
        winner.complemented, winner.train_agreement,
    )
}

fn main() {
    let args = Args::parse();
    assert!(args.n > 0, "--n must be positive");
    assert!(args.samples > 0, "--samples must be positive");
    assert!(args.c_step > 0, "--c-step must be positive");
    assert!(args.g_step > 0, "--g-step must be positive");
    assert!(
        0.0 <= args.core_low && args.core_low < args.core_high && args.core_high <= 1.0,
        "ridge core must satisfy 0 <= --core-low < --core-high <= 1"
    );
    assert!(
        args.ridge_min_informative_rows >= 3,
        "--ridge-min-informative-rows must be at least 3"
    );
    assert!(
        0.0 < args.ridge_min_coverage && args.ridge_min_coverage <= 1.0,
        "--ridge-min-coverage must satisfy 0 < value <= 1"
    );
    assert!(
        args.ridge_tie_atol >= 0.0 && args.ridge_tie_atol.is_finite(),
        "--ridge-tie-atol must be finite and nonnegative"
    );

    let c = load_circuit(&args.c, &args.c_format, args.n, "c");
    let g = load_circuit(&args.g, &args.g_format, args.n, "g");
    let c_content_xxh3_128 = file_xxh3_128(&args.c);
    let g_content_xxh3_128 = file_xxh3_128(&args.g);
    let tool_source_xxh3_128 = bytes_xxh3_128(include_bytes!("hmap_stat.rs"));
    let cargo_lock_xxh3_128 = bytes_xxh3_128(include_bytes!("../../Cargo.lock"));
    let tool_executable_path = std::env::current_exe().expect("resolve current executable");
    let tool_executable_bytes =
        std::fs::read(&tool_executable_path).expect("read current executable for provenance hash");
    let tool_executable_xxh3_128 = bytes_xxh3_128(&tool_executable_bytes);
    assert_eq!(
        args.fixed_y.is_some(),
        args.fixed_z.is_some(),
        "--fixed-y and --fixed-z must be supplied together"
    );
    if args.fixed_y.is_some() {
        assert!(args.n <= 128, "fixed Y/Z currently require n <= 128");
        validate_fixed_value(args.fixed_y.unwrap(), args.n, "y")
            .unwrap_or_else(|error| panic!("{error}"));
        validate_fixed_value(args.fixed_z.unwrap(), args.n, "z")
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            g.num_wires >= 3 * args.n,
            "fixed Y/Z require at least 3*n G wires"
        );
    }
    if let Some(start) = args.random_helper_start {
        assert!(
            start >= args.n,
            "--random-helper-start must not overlap shared X wires"
        );
        assert!(
            start < g.num_wires,
            "--random-helper-start must leave at least one helper wire"
        );
        if args.fixed_y.is_some() {
            assert!(
                start >= 3 * args.n,
                "--random-helper-start must be at least 3*n when Y/Z are fixed"
            );
        }
    }

    let predictor_wires = parse_wire_list(&args.wire_list, g.num_wires)
        .unwrap_or_else(|error| panic!("invalid --wire-list: {error}"));
    let candidate_count =
        1 + predictor_wires.len() + predictor_wires.len() * (predictor_wires.len() - 1) / 2;

    let total_batches = args.samples.div_ceil(64).max(2);
    let train_batches = if args.train_samples == 0 {
        (3 * total_batches / 4).clamp(1, total_batches - 1)
    } else {
        args.train_samples.div_ceil(64)
    };
    assert!(
        train_batches > 0 && train_batches < total_batches,
        "training samples must occupy at least one batch and leave at least one holdout batch"
    );
    let actual_samples = total_batches * 64;
    let actual_train_samples = train_batches * 64;
    let actual_holdout_samples = (total_batches - train_batches) * 64;

    let c_indices = indices(c.gates.len(), args.c_step);
    let g_indices = indices(g.gates.len(), args.g_step);
    let rows = c_indices.len();
    let columns = g_indices.len();
    let c_end = c_indices.last().copied().unwrap_or(0) as f64;
    let core_row_count = c_indices
        .iter()
        .filter(|&&prefix| {
            let fraction = prefix as f64 / c_end;
            fraction >= args.core_low && fraction <= args.core_high
        })
        .count();
    if core_row_count < args.ridge_min_informative_rows {
        eprintln!(
            "[hmap_stat] warning: this grid has only {core_row_count} core rows, fewer than \
             --ridge-min-informative-rows={}; chronology will be N/A even if every row is \
             informative; reduce --c-step for an inferential run",
            args.ridge_min_informative_rows
        );
    }

    let mut target_rng = StdRng::seed_from_u64(args.seed ^ TARGET_RNG_DOMAIN);
    let target_bits = choose_target_bits(args.n, args.target_bits, &mut target_rng);

    println!(
        "[hmap_stat] C={}g/{}w G={}g/{}w n={} rows={} cols={} samples={} \
         (train={} holdout={}) targets={} predictor_wires={} candidates={}",
        c.gates.len(),
        c.num_wires,
        g.gates.len(),
        g.num_wires,
        args.n,
        rows,
        columns,
        actual_samples,
        actual_train_samples,
        actual_holdout_samples,
        target_bits.len(),
        predictor_wires.len(),
        candidate_count,
    );

    let mut x_rng = StdRng::seed_from_u64(args.seed);
    let mut helper_rng = StdRng::seed_from_u64(args.seed ^ HELPER_RNG_DOMAIN);
    let mut c_snapshots = Vec::with_capacity(total_batches);
    let mut g_snapshots = Vec::with_capacity(total_batches);
    for _ in 0..total_batches {
        let (c_input, g_input) = sample_batch_inputs(
            args.n,
            c.num_wires,
            g.num_wires,
            args.fixed_y,
            args.fixed_z,
            args.random_helper_start,
            &mut x_rng,
            &mut helper_rng,
        );
        c_snapshots.push(snapshots(&c.gates, &c_input, &c_indices));
        g_snapshots.push(snapshots(&g.gates, &g_input, &g_indices));
    }

    let mut null_rng = StdRng::seed_from_u64(args.seed ^ NULL_RNG_DOMAIN);
    let null_targets: Vec<Vec<Vec<u64>>> = (0..rows)
        .map(|_| {
            (0..target_bits.len())
                .map(|_| {
                    (0..total_batches)
                        .map(|_| null_rng.random::<u64>())
                        .collect()
                })
                .collect()
        })
        .collect();

    let mut holdout_matrix = vec![0f32; rows * columns];
    let mut null_matrix = vec![0f32; rows * columns];
    let winners_path = format!("{}.winners.jsonl", args.out);
    let winners_file = File::create(&winners_path)
        .unwrap_or_else(|error| panic!("create {winners_path}: {error}"));
    let mut winners = BufWriter::new(winners_file);
    let train_range = 0..train_batches;
    let holdout_range = train_batches..total_batches;

    for (column, &g_prefix) in g_indices.iter().enumerate() {
        let wire_columns: Vec<Vec<u64>> = predictor_wires
            .iter()
            .map(|&wire| {
                (0..total_batches)
                    .map(|batch| g_snapshots[batch][column][wire])
                    .collect()
            })
            .collect();

        for (row, &c_prefix) in c_indices.iter().enumerate() {
            let mut holdout_sum = 0.0;
            let mut null_sum = 0.0;
            for (target_index, &target_bit) in target_bits.iter().enumerate() {
                let target: Vec<u64> = (0..total_batches)
                    .map(|batch| c_snapshots[batch][row][target_bit])
                    .collect();
                let winner = select_candidate(&target, &wire_columns, train_range.clone());
                let holdout_agreement =
                    score_frozen_candidate(&target, &wire_columns, winner, holdout_range.clone());
                holdout_sum += holdout_agreement;
                writeln!(
                    winners,
                    "{}",
                    winner_json(
                        row,
                        column,
                        c_prefix,
                        g_prefix,
                        target_bit,
                        &predictor_wires,
                        winner,
                        holdout_agreement,
                    )
                )
                .expect("write winners JSONL");

                let null_target = &null_targets[row][target_index];
                let null_winner = select_candidate(null_target, &wire_columns, train_range.clone());
                null_sum += score_frozen_candidate(
                    null_target,
                    &wire_columns,
                    null_winner,
                    holdout_range.clone(),
                );
            }
            holdout_matrix[row * columns + column] =
                (holdout_sum / target_bits.len() as f64) as f32;
            null_matrix[row * columns + column] = (null_sum / target_bits.len() as f64) as f32;
        }
        println!(
            "[hmap_stat] gadget column {}/{} complete",
            column + 1,
            columns
        );
    }
    winners.flush().expect("flush winners JSONL");

    let (mean, sigma) = mean_sigma(&holdout_matrix);
    let (null_mean, null_sigma) = mean_sigma(&null_matrix);
    let ridge = ridge_summary(
        &holdout_matrix,
        &null_matrix,
        rows,
        columns,
        &c_indices,
        &g_indices,
        target_bits.len(),
        args.core_low,
        args.core_high,
        args.ridge_min_informative_rows,
        args.ridge_min_coverage,
        args.ridge_tie_atol,
    );
    println!(
        "[hmap_stat] holdout mean={mean:.6} sigma={sigma:.6}; honest null \
         mean={null_mean:.6} sigma={null_sigma:.6}"
    );
    match (ridge.rho, ridge.circular_shift_p) {
        (Some(rho), Some(pvalue)) => println!(
            "[hmap_stat] interior rows={} cols={} informative={}/{} ({:.1}%) \
             median_peak={:.6} \
             null_median_peak={:.6} tie_safe_rho={rho:.6} circular_shift_p={pvalue:.6} \
             ({} nonzero shifts)",
            ridge.core_rows,
            ridge.core_columns,
            ridge.informative_rows,
            ridge.required_informative_rows,
            100.0 * ridge.informative_coverage,
            ridge.median_peak_agreement.unwrap(),
            ridge.median_null_peak_agreement.unwrap(),
            ridge.circular_shifts,
        ),
        _ => println!(
            "[hmap_stat] interior rows={} cols={} informative={}/{} ({:.1}%): \
             chronology undefined (flat, tied, or insufficient informative coverage)",
            ridge.core_rows,
            ridge.core_columns,
            ridge.informative_rows,
            ridge.required_informative_rows,
            100.0 * ridge.informative_coverage,
        ),
    }

    let matrix_path = format!("{}.bin", args.out);
    let null_path = format!("{}.null.bin", args.out);
    write_f32_matrix(&matrix_path, &holdout_matrix);
    write_f32_matrix(&null_path, &null_matrix);

    let fixed_y_json = args.fixed_y.map_or_else(
        || "null".to_owned(),
        |value| json_string(&format!("{value:#034x}")),
    );
    let fixed_z_json = args.fixed_z.map_or_else(
        || "null".to_owned(),
        |value| json_string(&format!("{value:#034x}")),
    );
    let fixed_blocks = if args.fixed_y.is_some() {
        format!(
            "[{{\"name\":\"y\",\"start\":{},\"end\":{},\"end_exclusive\":true,\
             \"value\":{fixed_y_json}}},{{\"name\":\"z\",\"start\":{},\"end\":{},\
             \"end_exclusive\":true,\"value\":{fixed_z_json}}}]",
            args.n,
            2 * args.n,
            2 * args.n,
            3 * args.n,
        )
    } else {
        "[]".to_owned()
    };
    let helper_start_json = json_optional_usize(args.random_helper_start);
    let helper_end_json = args
        .random_helper_start
        .map_or_else(|| "null".to_owned(), |_| g.num_wires.to_string());
    let c_declared_json = json_optional_usize(c.declared_num_wires);
    let g_declared_json = json_optional_usize(g.declared_num_wires);
    let x_seed_json = json_string(&format!("{:#018x}", args.seed));
    let helper_seed_json = json_string(&format!("{:#018x}", args.seed ^ HELPER_RNG_DOMAIN));
    let target_seed_json = json_string(&format!("{:#018x}", args.seed ^ TARGET_RNG_DOMAIN));
    let null_seed_json = json_string(&format!("{:#018x}", args.seed ^ NULL_RNG_DOMAIN));
    let helper_policy = if args.random_helper_start.is_some() {
        "independent_random_suffix"
    } else {
        "zero_for_unspecified_inputs"
    };
    let ridge_peak_json = json_optional_f64(ridge.median_peak_agreement);
    let ridge_null_peak_json = json_optional_f64(ridge.median_null_peak_agreement);
    let ridge_prominence_json = json_optional_f64(ridge.median_prominence);
    let ridge_rho_json = json_optional_f64(ridge.rho);
    let ridge_p_json = json_optional_f64(ridge.circular_shift_p);
    let tool_executable_path_json = json_string(tool_executable_path.to_string_lossy().as_ref());
    let ridge_informative_rows = ridge.informative_rows;
    let ridge_informative_coverage = ridge.informative_coverage;
    let ridge_required_informative_rows = ridge.required_informative_rows;
    let ridge_prominence_threshold = ridge.prominence_threshold;
    let metadata = format!(
        "{{\"schema\":\"hmap_stat_v2\",\"measure\":\"mean frozen-candidate holdout \
         agreement\",\"selection\":\"constant/single/xor-pair plus complement selected \
         on training only\",\"candidate_tie_break\":\"constant, then ascending single, \
         then lexicographic pair; uncomplemented wins polarity ties\",\
         \"run_label\":{},\"c\":{},\"g\":{},\"c_format\":{},\"g_format\":{},\
         \"c_content_xxh3_128\":\"{c_content_xxh3_128}\",\
         \"g_content_xxh3_128\":\"{g_content_xxh3_128}\",\
         \"tool_source_xxh3_128\":\"{tool_source_xxh3_128}\",\
         \"cargo_lock_xxh3_128\":\"{cargo_lock_xxh3_128}\",\
         \"tool_executable\":{tool_executable_path_json},\
         \"tool_executable_xxh3_128\":\"{tool_executable_xxh3_128}\",\
         \"n\":{},\"c_gates\":{},\"g_gates\":{},\"c_num_wires\":{},\
         \"g_num_wires\":{},\"c_inferred_num_wires\":{},\
         \"g_inferred_num_wires\":{},\"c_declared_num_wires\":{},\
         \"g_declared_num_wires\":{},\"rows\":{},\"cols\":{},\
         \"requested_samples\":{},\"requested_train_samples\":{},\"samples\":{},\
         \"train_samples\":{},\"holdout_samples\":{},\"batches\":{},\
         \"train_batches\":{},\"seed\":{},\"rng_seeds\":{{\"x\":{},\"helper\":{},\
         \"target_subset\":{},\"null\":{}}},\"x_rng\":\"seed\",\
         \"helper_rng_domain\":\"HELPER_4\",\
         \"target_rng_domain\":\"TARGET_S\",\"null_rng_domain\":\"NULL_STA\",\
         \"target_bits\":{},\"requested_target_bits\":{},\"wire_list\":{},\
         \"predictor_wires\":{},\"candidate_count\":{},\"fixed_y\":{},\
         \"fixed_z\":{},\"fixed_input_blocks\":{},\"random_helper_start\":{},\
         \"random_helper_end\":{},\"random_helper_end_exclusive\":true,\
         \"free_helper\":{},\"helper_policy\":\"{helper_policy}\",\
         \"c_step\":{},\"g_step\":{},\"i_idx\":{},\
         \"j_idx\":{},\"matrix_file\":{},\"null_matrix_file\":{},\
         \"matrix_dtype\":\"f32\",\"matrix_byte_order\":\"little_endian\",\
         \"matrix_layout\":\"row_major[row*cols+column]\",\
         \"winners_file\":{},\"winner_order\":\"g-column,row,target\",\
         \"holdout_mean\":{:.9},\"holdout_sigma\":{:.9},\
         \"null_mean\":{:.9},\"null_sigma\":{:.9},\
         \"null_definition\":\"independent random targets; candidate and complement \
         selected on null training lanes and frozen on null holdout lanes\",\
         \"ridge_core_low\":{},\"ridge_core_high\":{},\
         \"ridge_core_rows\":{},\"ridge_core_columns\":{},\
         \"ridge_min_informative_rows\":{},\"ridge_min_coverage\":{},\
         \"ridge_tie_atol\":{},\
         \"ridge_prominence_definition\":\"row maximum minus row median\",\
         \"ridge_prominence_averaged_target_bits\":{},\
         \"ridge_one_selected_target_bit_prominence_threshold\":{ridge_prominence_threshold:.9},\
         \"ridge_median_prominence\":{ridge_prominence_json},\
         \"ridge_informative_rows\":{ridge_informative_rows},\
         \"ridge_required_informative_rows\":{ridge_required_informative_rows},\
         \"ridge_informative_coverage\":{ridge_informative_coverage:.9},\
         \"ridge_median_peak_agreement\":{ridge_peak_json},\
         \"ridge_median_null_peak_agreement\":{ridge_null_peak_json},\
         \"ridge_tie_safe_rho\":{ridge_rho_json},\
         \"ridge_circular_shift_p\":{ridge_p_json},\
         \"ridge_circular_shifts\":{},\
         \"ridge_null\":\"all nonzero circular shifts of tied-average peak columns; \
         serial order preserved; no free row-permutation p-value\"}}",
        json_string(&args.run_label),
        json_string(&args.c),
        json_string(&args.g),
        json_string(&args.c_format),
        json_string(&args.g_format),
        args.n,
        c.gates.len(),
        g.gates.len(),
        c.num_wires,
        g.num_wires,
        c.inferred_num_wires,
        g.inferred_num_wires,
        c_declared_json,
        g_declared_json,
        rows,
        columns,
        args.samples,
        args.train_samples,
        actual_samples,
        actual_train_samples,
        actual_holdout_samples,
        total_batches,
        train_batches,
        args.seed,
        x_seed_json,
        helper_seed_json,
        target_seed_json,
        null_seed_json,
        json_usize_array(&target_bits),
        args.target_bits,
        json_string(&args.wire_list),
        json_usize_array(&predictor_wires),
        candidate_count,
        fixed_y_json,
        fixed_z_json,
        fixed_blocks,
        helper_start_json,
        helper_end_json,
        args.random_helper_start.is_some(),
        args.c_step,
        args.g_step,
        json_usize_array(&c_indices),
        json_usize_array(&g_indices),
        json_string(&matrix_path),
        json_string(&null_path),
        json_string(&winners_path),
        mean,
        sigma,
        null_mean,
        null_sigma,
        args.core_low,
        args.core_high,
        ridge.core_rows,
        ridge.core_columns,
        args.ridge_min_informative_rows,
        args.ridge_min_coverage,
        args.ridge_tie_atol,
        target_bits.len(),
        ridge.circular_shifts,
    );
    let metadata_path = format!("{}.meta.json", args.out);
    std::fs::write(&metadata_path, metadata)
        .unwrap_or_else(|error| panic!("write {metadata_path}: {error}"));

    println!("[hmap_stat] wrote {matrix_path}, {null_path}, {winners_path}, and {metadata_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_list_parser_accepts_ranges_sorts_and_deduplicates() {
        assert_eq!(
            parse_wire_list("4, 1-3,2,7-8", 10).unwrap(),
            vec![1, 2, 3, 4, 7, 8]
        );
        assert_eq!(parse_wire_list("", 4).unwrap(), vec![0, 1, 2, 3]);
        assert!(parse_wire_list("3-1", 4).is_err());
        assert!(parse_wire_list("0-4", 4).is_err());
        assert!(parse_wire_list("1,,2", 4).is_err());
        assert!(validate_fixed_value(0xffff, 16, "y").is_ok());
        assert!(validate_fixed_value(0x1_0000, 16, "y").is_err());
        assert!(validate_fixed_value(u128::MAX, 128, "z").is_ok());
    }

    #[test]
    fn helper_width_does_not_change_shared_x_stream() {
        let seed = 0x1234_5678_u64;
        let mut x_a = StdRng::seed_from_u64(seed);
        let mut helper_a = StdRng::seed_from_u64(seed ^ HELPER_RNG_DOMAIN);
        let mut x_b = StdRng::seed_from_u64(seed);
        let mut helper_b = StdRng::seed_from_u64(seed ^ HELPER_RNG_DOMAIN);

        for _ in 0..4 {
            let (c_a, g_a) = sample_batch_inputs(
                2,
                2,
                8,
                Some(0b01),
                Some(0b10),
                Some(6),
                &mut x_a,
                &mut helper_a,
            );
            let (c_b, g_b) = sample_batch_inputs(
                2,
                2,
                10,
                Some(0b01),
                Some(0b10),
                Some(6),
                &mut x_b,
                &mut helper_b,
            );
            assert_eq!(c_a, c_b);
            assert_eq!(&g_a[..6], &g_b[..6]);
        }
    }

    #[test]
    fn candidate_and_complement_are_frozen_before_holdout_scoring() {
        let pattern = 0xaaaa_aaaa_aaaa_aaaau64;
        let target = vec![!pattern, pattern];
        let wires = vec![vec![pattern, pattern]];

        let winner = select_candidate(&target, &wires, 0..1);
        assert_eq!(winner.candidate, Candidate::Single(0));
        assert!(winner.complemented);
        assert_eq!(winner.train_agreement, 1.0);

        let holdout = score_frozen_candidate(&target, &wires, winner, 1..2);
        assert_eq!(holdout, 0.0);
    }

    #[test]
    fn ridge_summary_uses_tied_ranks_and_circular_shifts() {
        let rows = 5;
        let columns = 5;
        let mut matrix = vec![0.5f32; rows * columns];
        let null = vec![0.5f32; rows * columns];
        for index in 0..rows {
            matrix[index * columns + index] = 1.0;
        }
        let prefixes = vec![0, 1, 2, 3, 4];
        let summary = ridge_summary(
            &matrix, &null, rows, columns, &prefixes, &prefixes, 1, 0.0, 1.0, 3, 0.25, 1e-7,
        );
        assert_eq!(summary.core_rows, 5);
        assert_eq!(summary.core_columns, 5);
        assert_eq!(summary.informative_rows, 5);
        assert_eq!(summary.prominence_threshold, 0.5);
        assert_eq!(summary.rho, Some(1.0));
        assert_eq!(summary.circular_shifts, 4);
        assert!(summary.circular_shift_p.is_some_and(|value| value > 0.0));

        let flat = ridge_summary(
            &null, &null, rows, columns, &prefixes, &prefixes, 1, 0.0, 1.0, 3, 0.25, 1e-7,
        );
        assert!(flat.rho.is_none());
        assert!(flat.circular_shift_p.is_none());

        let selected_subset = ridge_summary(
            &matrix, &null, rows, columns, &prefixes, &prefixes, 16, 0.0, 1.0, 3, 0.25, 1e-7,
        );
        assert_eq!(selected_subset.prominence_threshold, 0.5 / 16.0);
    }

    #[test]
    fn one_isolated_peak_cannot_define_chronology() {
        let rows = 25;
        let columns = 25;
        let mut matrix = vec![0.5f32; rows * columns];
        let null = matrix.clone();
        matrix[(rows - 1) * columns + (columns - 1)] = 1.0;
        let prefixes: Vec<usize> = (0..rows).collect();

        let summary = ridge_summary(
            &matrix, &null, rows, columns, &prefixes, &prefixes, 1, 0.0, 1.0, 10, 0.25, 1e-7,
        );
        assert_eq!(summary.informative_rows, 1);
        assert_eq!(summary.required_informative_rows, 10);
        assert!(summary.rho.is_none());
        assert!(summary.circular_shift_p.is_none());
    }
}
