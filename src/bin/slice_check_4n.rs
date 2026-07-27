//! Fast fixed-slice semantic checker for 4n two-share TDP circuits, including
//! variants with an appended arbitrary helper band.
//!
//! The scalar research checker evaluates every `(x, W)` case separately.  This
//! binary packs 64 such cases into the bits of each `u64` lane word, so one
//! circuit traversal checks 64 cases.  Its x samples and twelve pseudorandom W
//! words deliberately reproduce CPython's `random.Random(seed).getrandbits(n)`;
//! the other four W words are zero, all-one, alternating, and
//! complement-alternating, matching the existing evidence checker.

use clap::Parser;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, eval_lanes, max_wire};
use std::fmt::Write as _;
use std::time::Instant;

const LANES: usize = 64;
const HELPER_PATTERN_COUNT: usize = 16;
const HELPER_SEED_DOMAIN: u64 = 0x4e5f_4e48;

#[derive(Parser, Debug)]
#[command(name = "slice_check_4n")]
#[command(about = "Bit-sliced fixed-Y/Z semantic checker for strict 4n TDP circuits")]
struct Args {
    /// Candidate strict-4n circuit in mpmct1 format.
    circuit: String,

    /// Original n-wire all-G57 circuit C.
    #[arg(long, alias = "base")]
    base_circuit: String,

    /// Logical block width. The candidate must declare at least 4*n wires.
    #[arg(long)]
    n: usize,

    /// Fixed public Y block value (hex with 0x prefix or decimal).
    #[arg(long, value_parser = parse_u128)]
    fixed_y: u128,

    /// Fixed public Z block value (hex with 0x prefix or decimal).
    #[arg(long, value_parser = parse_u128)]
    fixed_z: u128,

    /// Number of deterministic x samples. Every x is checked with all 16 W patterns.
    #[arg(long, default_value_t = 16)]
    samples: usize,

    /// CPython-compatible deterministic x-sample seed.
    #[arg(long, default_value_t = 20_260_716)]
    seed: u64,

    /// Accepted for command-line compatibility with cnot_painless.py.
    #[arg(long, default_value = "mpmct1")]
    input_format: String,

    /// Accepted for command-line compatibility with cnot_painless.py.
    #[arg(long, default_value = "g57")]
    base_input_format: String,
}

#[derive(Clone, Debug, PartialEq)]
struct CheckReport {
    n: usize,
    circuit_wires: usize,
    circuit_gates: usize,
    base_wires: usize,
    base_gates: usize,
    helper_wires: usize,
    helper_patterns: usize,
    x_samples: usize,
    checked_cases: usize,
    lane_batches: usize,
    fixed_y: u128,
    fixed_z: u128,
    seed: u64,
    elapsed_seconds: f64,
}

impl CheckReport {
    fn evidence(&self) -> String {
        let mut out = String::new();
        writeln!(out, "n={}", self.n).unwrap();
        writeln!(out, "circuit_wires={}", self.circuit_wires).unwrap();
        writeln!(out, "circuit_gates={}", self.circuit_gates).unwrap();
        writeln!(out, "base_wires={}", self.base_wires).unwrap();
        writeln!(out, "base_gates={}", self.base_gates).unwrap();
        writeln!(out, "helper_wires={}", self.helper_wires).unwrap();
        writeln!(out, "helper_patterns={}", self.helper_patterns).unwrap();
        writeln!(out, "x_samples={}", self.x_samples).unwrap();
        writeln!(out, "checked_cases={}", self.checked_cases).unwrap();
        writeln!(out, "lane_width={LANES}").unwrap();
        writeln!(out, "lane_batches={}", self.lane_batches).unwrap();
        writeln!(out, "fixed_y={}", format_hex(self.fixed_y, self.n)).unwrap();
        writeln!(out, "fixed_z={}", format_hex(self.fixed_z, self.n)).unwrap();
        writeln!(out, "seed={}", self.seed).unwrap();
        writeln!(out, "checker=rust_bitslice_64").unwrap();
        writeln!(
            out,
            "helper_pattern_scheme=zero,ones,alternating,complement_alternating,python_mt19937_x12"
        )
        .unwrap();
        writeln!(out, "elapsed_seconds={:.6}", self.elapsed_seconds).unwrap();
        writeln!(out, "base_semantics_ok=true").unwrap();
        writeln!(out, "slice_ok=true").unwrap();
        out
    }
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

fn format_hex(value: u128, n: usize) -> String {
    let digits = n.div_ceil(4).max(1);
    format!("0x{value:0digits$x}")
}

fn low_mask(n: usize) -> u128 {
    if n == 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    }
}

fn inferred_wires(gates: &[XGate]) -> usize {
    if gates.is_empty() {
        0
    } else {
        max_wire(gates.iter()) as usize + 1
    }
}

fn validate_gate_namespace(gates: &[XGate], wires: usize, label: &str) -> Result<(), String> {
    for (index, gate) in gates.iter().enumerate() {
        if gate.target as usize >= wires {
            return Err(format!(
                "{label} gate {index} target {} exceeds wire count {wires}",
                gate.target
            ));
        }
        if gate.ctrls.iter().any(|&(wire, _)| wire as usize >= wires) {
            return Err(format!(
                "{label} gate {index} has a control outside wire count {wires}"
            ));
        }
        if gate.ctrls.iter().any(|&(wire, _)| wire == gate.target) {
            return Err(format!(
                "{label} gate {index} reads its own target {}",
                gate.target
            ));
        }
        if gate.ctrls.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(format!(
                "{label} gate {index} repeats control wire {}",
                gate.ctrls
                    .windows(2)
                    .find(|pair| pair[0].0 == pair[1].0)
                    .unwrap()[0]
                    .0
            ));
        }
    }
    Ok(())
}

/// Minimal CPython-compatible MT19937 for `random.Random(int_seed).getrandbits`.
///
/// CPython seeds the generator with the little-endian 32-bit digits of the
/// absolute integer and the reference `init_by_array` algorithm. Campaign
/// seeds are u64, so at most two key words are needed here.
#[derive(Clone)]
struct PythonRandom {
    mt: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn from_u64(seed: u64) -> Self {
        let mut key = vec![seed as u32];
        let high = (seed >> 32) as u32;
        if high != 0 {
            key.push(high);
        }
        let mut rng = Self {
            mt: [0; 624],
            index: 624,
        };
        rng.init_genrand(19_650_218);
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for index in 1..624 {
            self.mt[index] = 1_812_433_253u32
                .wrapping_mul(self.mt[index - 1] ^ (self.mt[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        debug_assert!(!key.is_empty());
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624usize.max(key.len()) {
            let prior = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prior ^ (prior >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prior = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prior ^ (prior >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.index = 624;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            const MATRIX_A: u32 = 0x9908_b0df;
            const UPPER_MASK: u32 = 0x8000_0000;
            const LOWER_MASK: u32 = 0x7fff_ffff;
            for index in 0..624 {
                let y = (self.mt[index] & UPPER_MASK) | (self.mt[(index + 1) % 624] & LOWER_MASK);
                self.mt[index] =
                    self.mt[(index + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, bits: usize) -> u128 {
        assert!((1..=128).contains(&bits));
        if bits <= 32 {
            return (self.next_u32() >> (32 - bits)) as u128;
        }
        let mut remaining = bits;
        let mut value = 0u128;
        let mut shift = 0usize;
        while remaining > 0 {
            let take = remaining.min(32);
            let mut word = self.next_u32();
            if take < 32 {
                word >>= 32 - take;
            }
            value |= (word as u128) << shift;
            shift += 32;
            remaining -= take;
        }
        value
    }
}

fn helper_patterns(n: usize) -> [u128; HELPER_PATTERN_COUNT] {
    let mask = low_mask(n);
    let alternating = (0..n)
        .step_by(2)
        .fold(0u128, |value, bit| value | (1u128 << bit));
    let mut result = [0u128; HELPER_PATTERN_COUNT];
    result[0] = 0;
    result[1] = mask;
    result[2] = alternating;
    result[3] = mask ^ alternating;
    let mut rng = PythonRandom::from_u64(HELPER_SEED_DOMAIN ^ n as u64);
    for pattern in &mut result[4..] {
        *pattern = rng.getrandbits(n);
    }
    result
}

fn set_lane_bits(state: &mut [u64], start: usize, width: usize, value: u128, lane: usize) {
    let lane_bit = 1u64 << lane;
    for bit in 0..width {
        if (value >> bit) & 1 != 0 {
            state[start + bit] |= lane_bit;
        }
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn extra_helper_bit(pattern: usize, bit: usize) -> bool {
    match pattern {
        0 => false,
        1 => true,
        2 => bit % 2 == 0,
        3 => bit % 2 != 0,
        _ => {
            let word = splitmix64(
                HELPER_SEED_DOMAIN
                    ^ (pattern as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ ((bit / 64) as u64).wrapping_mul(0xa076_1d64_78bd_642f),
            );
            ((word >> (bit % 64)) & 1) != 0
        }
    }
}

fn check_slices(
    base: &[XGate],
    candidate: &[XGate],
    candidate_wires: usize,
    n: usize,
    fixed_y: u128,
    fixed_z: u128,
    samples: usize,
    seed: u64,
) -> Result<CheckReport, String> {
    if n == 0 || n > 128 {
        return Err(format!("n must be in 1..=128 (got {n})"));
    }
    let expected_wires = n
        .checked_mul(4)
        .ok_or_else(|| "4*n overflows usize".to_string())?;
    if candidate_wires < expected_wires {
        return Err(format!(
            "candidate must declare at least 4*n={expected_wires} wires (got {candidate_wires})"
        ));
    }
    if samples == 0 {
        return Err("--samples must be positive".to_string());
    }
    let mask = low_mask(n);
    if fixed_y & !mask != 0 {
        return Err(format!("fixed_y has bits above n={n}"));
    }
    if fixed_z & !mask != 0 {
        return Err(format!("fixed_z has bits above n={n}"));
    }
    let base_wires = inferred_wires(base);
    if base_wires > n {
        return Err(format!(
            "base circuit uses {base_wires} wires, larger than n={n}"
        ));
    }
    validate_gate_namespace(base, n, "base")?;
    validate_gate_namespace(candidate, candidate_wires, "candidate")?;

    let checked_cases = samples
        .checked_mul(HELPER_PATTERN_COUNT)
        .ok_or_else(|| "sample count overflows usize".to_string())?;
    let lane_batches = checked_cases.div_ceil(LANES);
    let patterns = helper_patterns(n);
    let mut x_rng = PythonRandom::from_u64(seed);
    let xs: Vec<u128> = (0..samples).map(|_| x_rng.getrandbits(n)).collect();
    let started = Instant::now();

    for batch in 0..lane_batches {
        let first_case = batch * LANES;
        let batch_cases = (checked_cases - first_case).min(LANES);
        let valid_lanes = if batch_cases == LANES {
            u64::MAX
        } else {
            (1u64 << batch_cases) - 1
        };
        let mut base_state = vec![0u64; n];
        let mut candidate_state = vec![0u64; candidate_wires];
        for lane in 0..batch_cases {
            let case = first_case + lane;
            let x_index = case / HELPER_PATTERN_COUNT;
            let helper_index = case % HELPER_PATTERN_COUNT;
            set_lane_bits(&mut base_state, 0, n, xs[x_index], lane);
            set_lane_bits(&mut candidate_state, 0, n, xs[x_index], lane);
            set_lane_bits(&mut candidate_state, n, n, fixed_y, lane);
            set_lane_bits(&mut candidate_state, 2 * n, n, fixed_z, lane);
            set_lane_bits(&mut candidate_state, 3 * n, n, patterns[helper_index], lane);
            for extra in 0..candidate_wires - expected_wires {
                if extra_helper_bit(helper_index, extra) {
                    candidate_state[expected_wires + extra] |= 1u64 << lane;
                }
            }
        }

        eval_lanes(base, &mut base_state);
        eval_lanes(candidate, &mut candidate_state);
        for wire in 0..n {
            let y_word = if (fixed_y >> wire) & 1 != 0 {
                valid_lanes
            } else {
                0
            };
            let mismatches = (candidate_state[n + wire] ^ base_state[wire] ^ y_word) & valid_lanes;
            if mismatches != 0 {
                let lane = mismatches.trailing_zeros() as usize;
                let case = first_case + lane;
                let x_index = case / HELPER_PATTERN_COUNT;
                let helper_index = case % HELPER_PATTERN_COUNT;
                return Err(format!(
                    "fixed-slice mismatch: case_index={x_index} helper_pattern={helper_index} \
                     wire={wire} x={} helper_value={} expected_bit={} actual_bit={}",
                    format_hex(xs[x_index], n),
                    format_hex(patterns[helper_index], n),
                    ((base_state[wire] ^ y_word) >> lane) & 1,
                    (candidate_state[n + wire] >> lane) & 1,
                ));
            }
        }
    }

    Ok(CheckReport {
        n,
        circuit_wires: candidate_wires,
        circuit_gates: candidate.len(),
        base_wires,
        base_gates: base.len(),
        helper_wires: candidate_wires - 3 * n,
        helper_patterns: HELPER_PATTERN_COUNT,
        x_samples: samples,
        checked_cases,
        lane_batches,
        fixed_y,
        fixed_z,
        seed,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn run(args: Args) -> Result<CheckReport, String> {
    if args.input_format != "mpmct1" {
        return Err(format!(
            "--input-format must be mpmct1 (got {:?})",
            args.input_format
        ));
    }
    if args.base_input_format != "g57" {
        return Err(format!(
            "--base-input-format must be g57 (got {:?})",
            args.base_input_format
        ));
    }
    let base = read_g57_file(&args.base_circuit)
        .map_err(|error| format!("read base G57 circuit: {error}"))?;
    let (candidate, candidate_wires) =
        read_mpmct(&args.circuit).map_err(|error| format!("read candidate mpmct1: {error}"))?;
    check_slices(
        &base,
        &candidate,
        candidate_wires,
        args.n,
        args.fixed_y,
        args.fixed_z,
        args.samples,
        args.seed,
    )
}

fn main() {
    let args = Args::parse();
    match run(args) {
        Ok(report) => print!("{}", report.evidence()),
        Err(error) => {
            println!("base_semantics_ok=false");
            println!("slice_ok=false");
            eprintln!("error={error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic_candidate(n: usize) -> (Vec<XGate>, Vec<XGate>) {
        let base = vec![XGate::from_g57([0, 1, 2])];
        let mut candidate = base.clone();
        candidate.extend((0..n).map(|wire| XGate::cnot((n + wire) as u16, wire as u16)));
        (base, candidate)
    }

    #[test]
    fn python_random_matches_cpython_getrandbits() {
        let mut x_rng = PythonRandom::from_u64(2_026_071_700);
        assert_eq!(
            x_rng.getrandbits(128),
            0x13e8_ff96_26f3_02ab_0980_3864_92af_ece7
        );
        assert_eq!(
            x_rng.getrandbits(128),
            0x9e51_17ca_8707_a1bd_53c6_3300_efec_634b
        );

        let mut rng64 = PythonRandom::from_u64(12_345);
        assert_eq!(rng64.getrandbits(64), 0xbb91_433a_6aa7_9987);
        assert_eq!(rng64.getrandbits(64), 0xd1f6_f86c_029a_7245);
    }

    #[test]
    fn helper_patterns_match_scalar_python_checker() {
        let patterns = helper_patterns(128);
        assert_eq!(patterns[0], 0);
        assert_eq!(patterns[1], u128::MAX);
        assert_eq!(patterns[2], 0x5555_5555_5555_5555_5555_5555_5555_5555);
        assert_eq!(patterns[3], 0xaaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa_aaaa);
        assert_eq!(patterns[4], 0xe4f2_33c3_e38f_d050_544a_844e_a667_eee0);
        assert_eq!(patterns[15], 0x2b6a_3a32_81e5_fb0b_46c4_2a01_b67a_50af);
    }

    #[test]
    fn checks_all_helpers_and_partial_final_lane_batch() {
        let n = 3;
        let (base, candidate) = semantic_candidate(n);
        let report = check_slices(&base, &candidate, 4 * n, n, 0b101, 0b011, 5, 99)
            .expect("valid strict-4n slice");
        assert_eq!(report.checked_cases, 80);
        assert_eq!(report.lane_batches, 2);
        assert_eq!(report.helper_patterns, 16);
        assert!(report.evidence().contains("slice_ok=true\n"));
    }

    #[test]
    fn detects_free_helper_leak_into_middle() {
        let n = 3;
        let (base, mut candidate) = semantic_candidate(n);
        candidate.push(XGate::cnot(n as u16, (3 * n) as u16));
        let error = check_slices(&base, &candidate, 4 * n, n, 0, 0, 1, 7)
            .expect_err("W-dependent middle output must fail");
        assert!(error.contains("fixed-slice mismatch"), "{error}");
        assert!(error.contains("helper_pattern="), "{error}");
    }

    #[test]
    fn accepts_an_appended_helper_band() {
        let n = 3;
        let (base, candidate) = semantic_candidate(n);
        let report = check_slices(&base, &candidate, 4 * n + 1, n, 0, 0, 1, 7)
            .expect("an unused appended helper is valid");
        assert_eq!(report.helper_wires, n + 1);
    }

    #[test]
    fn detects_appended_helper_band_leak_into_middle() {
        let n = 3;
        let (base, mut candidate) = semantic_candidate(n);
        candidate.push(XGate::cnot(n as u16, (4 * n) as u16));
        let error = check_slices(&base, &candidate, 4 * n + 1, n, 0, 0, 1, 7)
            .expect_err("band-dependent middle output must fail");
        assert!(error.contains("fixed-slice mismatch"), "{error}");
    }
}
