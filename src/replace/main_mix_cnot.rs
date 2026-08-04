//! Heterogeneous `sss --cnot` driver.
//!
//! The source file is still the legacy G57/base-83 format. From the requested
//! transformation onward this driver keeps the richer [`XGate`] representation
//! so native CNOTs and conjunction fragments are never coerced back into G57
//! helper sequences.

use std::path::Path;

use rand::{RngCore, SeedableRng, rngs::StdRng};

use crate::{
    circuit::circuit::{CircuitSeq, U1024},
    postmix::{
        compress::{CompressParams, compress},
        format,
        mix::{MixParams, Mixer},
        samf::insert_masked_swap_samfs,
        xgate::{XGate, eval_lanes},
    },
    replace::gadgets::{
        CnotCircuit, ProdConfig, feistalize_cnot, feistalize_with_slice_zero_cnot,
        feistalize_with_slice_zero_hardcoded_cnot, feistalize_with_slice_zero_random_cnot,
        gadgetize_cnot, nonlinear_gadgetize_with_slice_zero_cnot, packed_bit, tdp4n_cnot,
        tdp4n_nonlinear_cnot, tdp4n_nonlinear_with_slice_zero_random_cnot,
        tdp4n_with_slice_zero_random_cnot,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionView {
    Whole,
    GadgetLow,
    GadgetLowZeroNonData { slice_end: usize },
    FeistalMiddle,
    Tdp4nMiddle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SliceLayout {
    Feistal3n,
    Tdp4n,
}

pub struct CnotSssParams<'a> {
    pub rounds: usize,
    pub n: usize,
    pub m: usize,
    pub x: usize,
    pub save: &'a str,
    pub source: &'a str,
    pub do_gadgetize: bool,
    pub do_nonlinear_gadgetize: bool,
    pub do_feistalize: bool,
    pub do_tdp4n: bool,
    pub slice_zero: bool,
    pub slice_zero_random: bool,
    pub slice_zero_random_gates: usize,
    pub slice_zero_hardcoded: bool,
    pub slice_zero_hardcoded_rounds: usize,
    pub gadget_path: Option<&'a str>,
    pub full_shuffle: bool,
    pub full_shuffle_early: bool,
    pub shooting_times: usize,
    pub collision_rounds: usize,
    pub stable_compressions: usize,
    pub expansion_game: bool,
    pub equality_check: bool,
    pub rg_freq: usize,
}

/// Below this width `functionality_check` enumerates EVERY input instead of
/// sampling, turning the check from evidence into a proof.
///
/// 18, not 12: the production nonlinear TDP is 6n wires and `n >= 3` is
/// asserted upstream, so its MINIMUM width is 18. At 12 the exhaustive path
/// was structurally unreachable for exactly the construction that most needs
/// a proof. 2^18 inputs across 64 lanes is ~4k batches — milliseconds.
///
/// The scalar reference checker in the tests reads the same constant; if the
/// two ever disagree the lane/scalar equivalence test finds different
/// first-error inputs and fails confusingly rather than usefully.
const EXHAUSTIVE_WIRE_LIMIT: usize = 18;

fn mask(bits: usize) -> U1024 {
    if bits == 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn packed_words_to_u1024(words: &[u64], bits: usize) -> U1024 {
    let mut output = U1024::zero();
    for bit in 0..bits {
        if packed_bit(words, bit) {
            output |= U1024::one() << bit;
        }
    }
    output
}

fn packed_words_to_hex(words: &[u64], bits: usize) -> String {
    let nibbles = bits.div_ceil(4).max(1);
    let mut output = String::with_capacity(nibbles + 2);
    output.push_str("0x");
    for nibble in (0..nibbles).rev() {
        let mut value = 0u8;
        for offset in 0..4 {
            let bit = 4 * nibble + offset;
            if bit < bits && packed_bit(words, bit) {
                value |= 1 << offset;
            }
        }
        output.push(char::from_digit(value as u32, 16).unwrap());
    }
    output
}

fn write_slice_metadata(
    path: &str,
    n: usize,
    total_wires: usize,
    artifact_gates: usize,
    slice_preblock_gates: usize,
    construction_seed: u64,
    rg_frequency: usize,
    source_gates: usize,
    source_repr_xxh3_128: &str,
    public_y: &[u64],
    public_z: &[u64],
    layout: SliceLayout,
    nonlinear_config: Option<ProdConfig>,
) {
    let y = packed_words_to_hex(public_y, n);
    let z = packed_words_to_hex(public_z, n);
    let meta_path = format!("{path}.slice_zero_random");
    let three_n = 3 * n;
    let (layout_name, w_wires, extra_helper_start) = match layout {
        SliceLayout::Feistal3n => ("feistal_3n", "none".to_owned(), three_n),
        SliceLayout::Tdp4n => {
            assert!(
                total_wires >= 4 * n,
                "4n TDP metadata needs at least 4n wires"
            );
            (
                if nonlinear_config.is_some() {
                    "tdp4n_single_carrier_2223_gray_padded"
                } else {
                    "tdp4n_two_share"
                },
                format!("{}..{}", three_n, 4 * n),
                4 * n,
            )
        }
    };
    let nonlinear_metadata = if let Some(config) = nonlinear_config {
        assert_eq!(
            layout,
            SliceLayout::Tdp4n,
            "nonlinear fixed-slice metadata currently describes the TDP layout"
        );
        let band_start = 4 * n;
        assert!(
            total_wires > band_start,
            "nonlinear TDP metadata needs a nonempty appended band"
        );
        let mut plan = Vec::with_capacity(config.k + config.k_hi);
        plan.extend(std::iter::repeat_n(config.deg.max(2), config.k));
        plan.extend(std::iter::repeat_n(config.deg_hi.max(2), config.k_hi));
        let plan = plan
            .into_iter()
            .map(|degree| degree.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "nonlinear_enabled=true\n\
             nonlinear_config_version=2223-gray-v1\n\
             nonlinear_plan={plan}\n\
             nonlinear_fold=gray\n\
             nonlinear_single_carrier=true\n\
             nonlinear_k={}\n\
             nonlinear_deg={}\n\
             nonlinear_k_hi={}\n\
             nonlinear_deg_hi={}\n\
             nonlinear_band_config={}\n\
             nonlinear_band_size={}\n\
             nonlinear_band_wires={band_start}..{total_wires}\n\
             nonlinear_rsrc={}\n\
             nonlinear_max_width={}\n\
             nonlinear_fill_nl={}\n\
             nonlinear_roll={}\n",
            config.k,
            config.deg,
            config.k_hi,
            config.deg_hi,
            config.band,
            total_wires - band_start,
            config.rsrc,
            config.max_width,
            config.fill_nl,
            config.roll,
        )
    } else {
        "nonlinear_enabled=false\n".to_owned()
    };
    let contents = format!(
        "mode=slice_zero_random\n\
         representation=mpmct1\n\
         layout={layout_name}\n\
         n={n}\n\
         total_wires={total_wires}\n\
         gates={artifact_gates}\n\
         slice_preblock_gates={slice_preblock_gates}\n\
         construction_seed={construction_seed}\n\
         rg_frequency={rg_frequency}\n\
         source_gates={source_gates}\n\
         source_repr_xxh3_128={source_repr_xxh3_128}\n\
         {nonlinear_metadata}\
         y_hex={y}\n\
         z_hex={z}\n\
         x_wires=0..{n}\n\
         y_wires={n}..{}\n\
         z_wires={}..{three_n}\n\
         w_wires={w_wires}\n\
         sat_helper_wires={three_n}..{total_wires}\n\
         extra_helper_wires={extra_helper_start}..{total_wires}\n\
         middle_output_wires={n}..{}\n\
         fixed_input_blocks=y,z\n\
         bit_order=bit i is wire n+i for y and wire 2n+i for z\n",
        2 * n,
        2 * n,
        2 * n,
    );
    std::fs::write(&meta_path, contents).expect("write slice-zero-random metadata");
    println!("[sss:cnot] public slice Y={y} Z={z} ({meta_path})");
}

fn zero_nondata_slice_end(view: FunctionView) -> Option<usize> {
    match view {
        FunctionView::GadgetLowZeroNonData { slice_end } => Some(slice_end),
        _ => None,
    }
}

fn write_zero_nondata_metadata(path: &str, n: usize, total_wires: usize, slice_end: usize) {
    assert!(n <= slice_end && slice_end <= total_wires);
    let meta_path = format!("{path}.slice_zero_ccnot");
    let contents = format!(
        "mode=slice_zero_ccnot\n\
         representation=mpmct1\n\
         nonlinear_config_version=2223-gray-v1\n\
         nonlinear_plan=2,2,2,3\n\
         nonlinear_fold=gray\n\
         nonlinear_single_carrier=true\n\
         n={n}\n\
         total_wires={total_wires}\n\
         data_wires=0..{n}\n\
         fixed_input_wires={n}..{slice_end}\n\
         fixed_input_value=0\n\
         independent_helper_wires={slice_end}..{total_wires}\n\
         band_is_in_fixed_slice=true\n"
    );
    std::fs::write(&meta_path, contents).expect("write slice-zero-ccnot metadata");
    println!("[sss:cnot] production zero slice fixes wires {n}..{slice_end} ({meta_path})");
}

fn random_u1024(rng: &mut impl RngCore) -> U1024 {
    let mut bytes = [0u8; 128];
    rng.fill_bytes(&mut bytes);
    U1024::from_little_endian(&bytes)
}

const VERIFY_LANES: usize = u64::BITS as usize;
const VERIFY_WIRES: usize = 1024;

/// Transpose up to 64 ordinary 1024-bit states into the bit-sliced layout used
/// by `eval_lanes`: `state[wire]` contains that wire's value in every sample.
fn pack_verify_lanes(inputs: &[U1024]) -> Vec<u64> {
    debug_assert!(inputs.len() <= VERIFY_LANES);
    let mut state = vec![0u64; VERIFY_WIRES];
    for (lane, input) in inputs.iter().enumerate() {
        let lane_mask = 1u64 << lane;
        for (word_index, &word) in input.0.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                state[word_index * u64::BITS as usize + bit] |= lane_mask;
                remaining &= remaining - 1;
            }
        }
    }
    state
}

#[inline]
fn eval_circuit_lanes(circuit: &CircuitSeq, state: &mut [u64]) {
    for &[target, positive, negative] in &circuit.gates {
        let fires = state[positive as usize] | !state[negative as usize];
        state[target as usize] ^= fires;
    }
}

#[inline]
fn valid_lane_mask(lanes: usize) -> u64 {
    debug_assert!((1..=VERIFY_LANES).contains(&lanes));
    if lanes == VERIFY_LANES {
        u64::MAX
    } else {
        (1u64 << lanes) - 1
    }
}

fn functionality_check(
    original: &CircuitSeq,
    transformed: &[XGate],
    view: FunctionView,
    n: usize,
    total_wires: usize,
    fixed_slice: Option<(U1024, U1024)>,
    samples: usize,
    seed: u64,
) -> Result<(), String> {
    assert!(
        total_wires <= 1024,
        "XGate equality supports at most 1024 wires"
    );
    let low_mask = mask(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let exhaustive = total_wires <= EXHAUSTIVE_WIRE_LIMIT;
    let count = if exhaustive {
        1usize << total_wires
    } else {
        samples
    };
    let total_mask = mask(total_wires);
    for batch_start in (0..count).step_by(VERIFY_LANES) {
        let batch_len = (count - batch_start).min(VERIFY_LANES);
        let mut inputs = Vec::with_capacity(batch_len);
        for index in batch_start..batch_start + batch_len {
            let mut input = if exhaustive {
                U1024::from(index)
            } else {
                random_u1024(&mut rng) & total_mask
            };
            if let FunctionView::GadgetLowZeroNonData { slice_end } = view {
                let fixed_mask = mask(slice_end) ^ low_mask;
                input &= !fixed_mask;
            }
            if matches!(
                view,
                FunctionView::FeistalMiddle | FunctionView::Tdp4nMiddle
            ) {
                if let Some((public_y, public_z)) = fixed_slice {
                    let fixed_mask = (low_mask << n) | (low_mask << (2 * n));
                    input &= !fixed_mask;
                    input |= public_y << n;
                    input |= public_z << (2 * n);
                }
            }
            inputs.push(input);
        }

        let input_state = pack_verify_lanes(&inputs);
        let mut actual_state = input_state.clone();
        eval_lanes(transformed, &mut actual_state);

        let mut original_state = if view == FunctionView::Whole {
            input_state.clone()
        } else {
            let mut state = vec![0u64; VERIFY_WIRES];
            state[..n].copy_from_slice(&input_state[..n]);
            state
        };
        eval_circuit_lanes(original, &mut original_state);

        let mut mismatched_lanes = 0u64;
        match view {
            FunctionView::Whole => {
                for wire in 0..VERIFY_WIRES {
                    mismatched_lanes |= actual_state[wire] ^ original_state[wire];
                }
            }
            FunctionView::GadgetLow | FunctionView::GadgetLowZeroNonData { .. } => {
                for wire in 0..n {
                    mismatched_lanes |= actual_state[wire] ^ original_state[wire];
                }
            }
            FunctionView::FeistalMiddle | FunctionView::Tdp4nMiddle => {
                for wire in 0..n {
                    mismatched_lanes |=
                        actual_state[n + wire] ^ input_state[n + wire] ^ original_state[wire];
                }
            }
        }
        mismatched_lanes &= valid_lane_mask(batch_len);
        if mismatched_lanes != 0 {
            let lane = mismatched_lanes.trailing_zeros() as usize;
            let input = inputs[lane];
            return Err(format!(
                "functionality mismatch in {:?} view at sampled input 0x{:x}",
                view, input
            ));
        }
    }
    Ok(())
}

fn full_equivalence_check(
    before: &[XGate],
    after: &[XGate],
    total_wires: usize,
    samples: usize,
    seed: u64,
) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let exhaustive = total_wires <= EXHAUSTIVE_WIRE_LIMIT;
    let count = if exhaustive {
        1usize << total_wires
    } else {
        samples
    };
    let total_mask = mask(total_wires);
    for batch_start in (0..count).step_by(VERIFY_LANES) {
        let batch_len = (count - batch_start).min(VERIFY_LANES);
        let mut inputs = Vec::with_capacity(batch_len);
        for index in batch_start..batch_start + batch_len {
            inputs.push(if exhaustive {
                U1024::from(index)
            } else {
                random_u1024(&mut rng) & total_mask
            });
        }

        let input_state = pack_verify_lanes(&inputs);
        let mut before_state = input_state.clone();
        let mut after_state = input_state;
        eval_lanes(before, &mut before_state);
        eval_lanes(after, &mut after_state);

        let mut mismatched_lanes = 0u64;
        for wire in 0..VERIFY_WIRES {
            mismatched_lanes |= before_state[wire] ^ after_state[wire];
        }
        mismatched_lanes &= valid_lane_mask(batch_len);
        if mismatched_lanes != 0 {
            let lane = mismatched_lanes.trailing_zeros() as usize;
            let input = inputs[lane];
            return Err(format!(
                "heterogeneous rewrite mismatch at input 0x{input:x}"
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct GateCounts {
    g57: usize,
    cnot: usize,
    x: usize,
    fragments: usize,
}

fn gate_counts(gates: &[XGate]) -> GateCounts {
    let mut counts = GateCounts::default();
    for gate in gates {
        if gate.comp
            && gate.ctrls.len() == 2
            && gate.ctrls.iter().filter(|&&(_, polarity)| polarity).count() == 1
        {
            counts.g57 += 1;
        } else if !gate.comp && gate.ctrls.len() == 1 && gate.ctrls[0].1 {
            counts.cnot += 1;
        } else if !gate.comp && gate.ctrls.is_empty() {
            counts.x += 1;
        } else {
            counts.fragments += 1;
        }
    }
    counts
}

fn print_counts(label: &str, gates: &[XGate]) {
    let counts = gate_counts(gates);
    println!(
        "[sss:cnot] {label}: total={} | G57={} CNOT={} X={} other-fragments={}",
        gates.len(),
        counts.g57,
        counts.cnot,
        counts.x,
        counts.fragments
    );
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn default_gadget_path(source: &str) -> String {
    let file_name = Path::new(source)
        .file_name()
        .expect("source path has no final component")
        .to_str()
        .expect("source file name is not UTF-8");
    format!("./gadgetized/{file_name}.mpmct1")
}

fn ensure_parent(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }
}

fn assert_tdp_namespace(
    enabled: bool,
    nonlinear: bool,
    gates: &[XGate],
    total_wires: usize,
    n: usize,
    stage: &str,
) {
    if !enabled {
        return;
    }
    let base_wires = n.checked_mul(4).expect("4n TDP wire count overflow");
    if nonlinear {
        // Pin the exact width, not just "more than 4n". The production
        // single-carrier layout is 4n carriers + band_size(2n) = 2n band = 6n
        // (ProdConfig::production has band: 0, so band_size(w) = max(w, 6),
        // and n >= 3 is asserted upstream). A band-size regression — e.g.
        // someone reinstating the retired band: 56 preset — would sail past a
        // `> 4n` check while silently moving every band offset the sidecar and
        // downstream verifiers depend on.
        let expected = base_wires
            .checked_add(n.checked_mul(2).expect("band wire count overflow"))
            .expect("6n TDP wire count overflow");
        assert_eq!(
            total_wires, expected,
            "nonlinear TDP wire-count invariant failed {stage}: expected 6n={expected}, got {total_wires}"
        );
    } else {
        assert_eq!(
            total_wires, base_wires,
            "strict 4n TDP wire-count invariant failed {stage}"
        );
    }
    for (gate_index, gate) in gates.iter().enumerate() {
        assert!(
            (gate.target as usize) < total_wires,
            "strict 4n TDP target wire {} is outside 0..{} at gate {} {stage}",
            gate.target,
            total_wires,
            gate_index
        );
        for &(wire, _) in &gate.ctrls {
            assert!(
                (wire as usize) < total_wires,
                "strict 4n TDP control wire {} is outside 0..{} at gate {} {stage}",
                wire,
                total_wires,
                gate_index
            );
        }
    }
}

/// Run the XGate-native shuffle/shoot/shuffle path selected by `sss --cnot`
/// or implicitly by `sss --nonlinear_gadgetize`.
pub fn main_shuffle_shoot_shuffle_cnot(original: &CircuitSeq, p: &CnotSssParams<'_>) {
    assert!(p.x > 0, "--x must be nonzero");
    assert!(p.n > 0, "--n must be nonzero");
    assert!(p.rg_freq > 0, "--rg-frequency must be nonzero");
    let selected_transforms = [p.do_gadgetize, p.do_feistalize, p.do_tdp4n]
        .into_iter()
        .filter(|selected| *selected)
        .count();
    assert!(
        selected_transforms <= 1,
        "--gadgetize, --feistalize, and --tdp4n are mutually exclusive"
    );
    assert!(
        !(p.do_nonlinear_gadgetize && (p.do_gadgetize || p.do_feistalize)),
        "--nonlinear_gadgetize may be direct or combined only with --tdp4n"
    );
    println!(
        "[sss:cnot] XGate-native backend selected: G57 ingress, heterogeneous mpmct1 thereafter"
    );
    if p.full_shuffle || p.full_shuffle_early {
        println!("[sss:cnot] full-shuffle requests an enlarged final masked-SAMF pass");
    }
    if p.expansion_game {
        println!("[sss:cnot] expansion-game maps to native fresh-wire fragment splits");
    }

    let configured_seed = env_u64("SSS_CNOT_SEED");
    let construction_seed = configured_seed.unwrap_or_else(rand::random);
    let mut rng = StdRng::seed_from_u64(construction_seed);
    let source_repr = original.repr();
    let source_repr_xxh3_128 = format!(
        "{:032x}",
        xxhash_rust::xxh3::xxh3_128(source_repr.as_bytes())
    );
    println!("[sss:cnot] reproducibility seed={construction_seed}");
    let mut public_slice_words: Option<(Vec<u64>, Vec<u64>)> = None;
    let mut public_slice_layout = None;
    let mut fixed_slice = None;
    let (transformed, view, label): (CnotCircuit, FunctionView, &str) = if p.do_tdp4n {
        assert!(
            !p.slice_zero && !p.slice_zero_hardcoded,
            "--tdp4n currently supports only --slice-zero-random"
        );
        if p.slice_zero_random {
            let output = if p.do_nonlinear_gadgetize {
                tdp4n_nonlinear_with_slice_zero_random_cnot(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_random_gates,
                    &mut rng,
                )
            } else {
                tdp4n_with_slice_zero_random_cnot(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_random_gates,
                    &mut rng,
                )
            };
            fixed_slice = Some((
                packed_words_to_u1024(&output.public_y, p.n),
                packed_words_to_u1024(&output.public_z, p.n),
            ));
            public_slice_words = Some((output.public_y, output.public_z));
            public_slice_layout = Some(SliceLayout::Tdp4n);
            (
                output.circuit,
                FunctionView::Tdp4nMiddle,
                if p.do_nonlinear_gadgetize {
                    "slice-zero-random [2,2,2,3] single-carrier Gray-fold TDP"
                } else {
                    "slice-zero-random 4n TDP"
                },
            )
        } else {
            (
                if p.do_nonlinear_gadgetize {
                    tdp4n_nonlinear_cnot(original, p.n, p.rg_freq, &mut rng)
                } else {
                    tdp4n_cnot(original, p.n, p.rg_freq, &mut rng)
                },
                FunctionView::Tdp4nMiddle,
                if p.do_nonlinear_gadgetize {
                    "[2,2,2,3] single-carrier Gray-fold TDP"
                } else {
                    "4n TDP"
                },
            )
        }
    } else if p.do_nonlinear_gadgetize {
        let output = nonlinear_gadgetize_with_slice_zero_cnot(original, p.n, p.rg_freq, &mut rng);
        let slice_end = output.num_wires;
        (
            output,
            FunctionView::GadgetLowZeroNonData { slice_end },
            "slice-zero [2,2,2,3] single-carrier Gray-fold gadgetized",
        )
    } else if p.do_feistalize {
        if p.slice_zero_random {
            let output = feistalize_with_slice_zero_random_cnot(
                original,
                p.n,
                p.rg_freq,
                p.slice_zero_random_gates,
                &mut rng,
            );
            fixed_slice = Some((
                packed_words_to_u1024(&output.public_y, p.n),
                packed_words_to_u1024(&output.public_z, p.n),
            ));
            public_slice_words = Some((output.public_y, output.public_z));
            public_slice_layout = Some(SliceLayout::Feistal3n);
            (
                output.circuit,
                FunctionView::FeistalMiddle,
                "slice-zero-random Feistal",
            )
        } else if p.slice_zero_hardcoded {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                feistalize_with_slice_zero_hardcoded_cnot(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_hardcoded_rounds,
                    &mut rng,
                ),
                FunctionView::FeistalMiddle,
                "slice-zero-hardcoded Feistal",
            )
        } else if p.slice_zero {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                feistalize_with_slice_zero_cnot(original, p.n, p.rg_freq, &mut rng),
                FunctionView::FeistalMiddle,
                "slice-zero Feistal",
            )
        } else {
            (
                feistalize_cnot(original, p.n, p.rg_freq, &mut rng),
                FunctionView::FeistalMiddle,
                "Feistal",
            )
        }
    } else if p.do_gadgetize {
        (
            gadgetize_cnot(original, p.n, p.rg_freq, &mut rng),
            FunctionView::GadgetLow,
            "gadgetized",
        )
    } else {
        (
            CnotCircuit {
                gates: original
                    .gates
                    .iter()
                    .copied()
                    .map(XGate::from_g57)
                    .collect(),
                num_wires: p.n,
            },
            FunctionView::Whole,
            "G57 ingress",
        )
    };
    let mut total_wires = transformed.num_wires;
    assert!(
        total_wires <= 1024,
        "--cnot supports at most 1024 transformed wires"
    );
    let mut gates = transformed.gates;
    assert_tdp_namespace(
        p.do_tdp4n,
        p.do_nonlinear_gadgetize,
        &gates,
        total_wires,
        p.n,
        "after construction",
    );
    println!(
        "[sss:cnot] {label}: source={} gates -> {} gates, {} wires; output format=mpmct1",
        original.gates.len(),
        gates.len(),
        total_wires
    );
    print_counts("after transformation", &gates);

    functionality_check(
        original,
        &gates,
        view,
        p.n,
        total_wires,
        fixed_slice,
        if p.equality_check { 10_000 } else { 256 },
        // Derive from the construction seed rather than a fixed literal: with
        // a constant here every run of every circuit was validated against the
        // SAME sample vectors, so a defect those vectors happen to miss is
        // missed identically forever. The per-round checks already vary their
        // seeds; this one was the outlier.
        construction_seed ^ 0xc001_c0de,
    )
    .expect("CNOT transformation changed required functionality");

    if p.do_gadgetize
        || p.do_nonlinear_gadgetize
        || p.do_feistalize
        || p.do_tdp4n
        || p.gadget_path.is_some()
    {
        let path = p
            .gadget_path
            .map(str::to_owned)
            .unwrap_or_else(|| default_gadget_path(p.source));
        ensure_parent(&path);
        format::write_mpmct(&path, &gates, total_wires).expect("write transformed mpmct1");
        println!("[sss:cnot] transformed circuit written to {path}");
        if let Some((public_y, public_z)) = &public_slice_words {
            write_slice_metadata(
                &path,
                p.n,
                total_wires,
                gates.len(),
                p.slice_zero_random_gates,
                construction_seed,
                p.rg_freq,
                original.gates.len(),
                &source_repr_xxh3_128,
                public_y,
                public_z,
                public_slice_layout.expect("public slice layout"),
                p.do_nonlinear_gadgetize.then_some(ProdConfig::production()),
            );
        }
        if let Some(slice_end) = zero_nondata_slice_end(view) {
            write_zero_nondata_metadata(&path, p.n, total_wires, slice_end);
        }
    }

    // A printed construction seed must replay the whole run even when the
    // original invocation did not receive SSS_CNOT_SEED.  Do not derive this
    // differently based on whether the seed came from the environment.
    let base_seed = construction_seed;
    let explicit_moves = env_u64("SSS_CNOT_MOVES_PER_ROUND");
    for round in 0..p.rounds {
        let before = gates.clone();
        let work_units = p.m.max(1).saturating_mul(gates.len().div_ceil(p.x).max(1));
        let moves = explicit_moves.unwrap_or_else(|| {
            (work_units as u64)
                .saturating_mul(128)
                .saturating_mul(p.shooting_times.max(1) as u64)
                .saturating_mul(p.collision_rounds.max(1) as u64)
                .clamp(10_000, 5_000_000)
        });
        let params = MixParams {
            target_size: gates.len(),
            moves,
            w_fresh: if p.expansion_game { 0.05 } else { 0.0 },
            // Arbitrary transvection/swap twists can transiently XOR two
            // complementary carriers. Native SAMFs are added only after all
            // rewrites, using a dedicated independent mask wire below.
            w_twist_cnot: 0.0,
            w_twist_swap: 0.0,
            twist_min_len: total_wires.max(8).min(gates.len().max(1)),
            verify_every: (moves / 10).max(1),
            report_every: (moves / 5).max(1),
            seed: base_seed.wrapping_add(round as u64),
            ..MixParams::default()
        };
        println!(
            "[sss:cnot] round {}/{}: moves={} target={} (unsafe arbitrary CNOT/swap twists disabled)",
            round + 1,
            p.rounds,
            moves,
            params.target_size,
        );
        let mut mixer = Mixer::new(gates, total_wires, params);
        mixer.run();
        mixer.final_float();
        mixer.global_check();
        gates = mixer.arena.to_vec();
        assert_tdp_namespace(
            p.do_tdp4n,
            p.do_nonlinear_gadgetize,
            &gates,
            total_wires,
            p.n,
            "after mixing",
        );
        print_counts(&format!("round {} after mixing", round + 1), &gates);

        let compress_params = CompressParams {
            max_iters: p.stable_compressions.max(1).saturating_mul(32),
            seed: base_seed ^ (0xc0de_0000 + round as u64),
            ..CompressParams::default()
        };
        let before_compress_len = gates.len();
        let (compressed, report) = compress(gates, total_wires, &compress_params);
        gates = compressed;
        assert_tdp_namespace(
            p.do_tdp4n,
            p.do_nonlinear_gadgetize,
            &gates,
            total_wires,
            p.n,
            "after compression",
        );
        println!(
            "[sss:cnot] round {} compression: {} -> {} gates in {} sweeps (fixed-point={})",
            round + 1,
            before_compress_len,
            gates.len(),
            report.iters,
            report.reached_fixed_point
        );
        full_equivalence_check(
            &before,
            &gates,
            total_wires,
            if p.equality_check { 10_000 } else { 256 },
            base_seed ^ (round as u64).wrapping_mul(0x9e37_79b9),
        )
        .expect("CNOT mixing/compression changed the transformed circuit");
        functionality_check(
            original,
            &gates,
            view,
            p.n,
            total_wires,
            fixed_slice,
            if p.equality_check { 10_000 } else { 256 },
            base_seed ^ (0x57fe_0000 + round as u64),
        )
        .expect("CNOT round changed required functionality");
        print_counts(&format!("round {} final", round + 1), &gates);

        let save_base = p.save.strip_suffix(".txt").unwrap_or(p.save);
        let round_path = format!("{save_base}round{}.txt", round + 1);
        ensure_parent(&round_path);
        format::write_mpmct(&round_path, &gates, total_wires).expect("write CNOT round");
        if let Some((public_y, public_z)) = &public_slice_words {
            write_slice_metadata(
                &round_path,
                p.n,
                total_wires,
                gates.len(),
                p.slice_zero_random_gates,
                construction_seed,
                p.rg_freq,
                original.gates.len(),
                &source_repr_xxh3_128,
                public_y,
                public_z,
                public_slice_layout.expect("public slice layout"),
                p.do_nonlinear_gadgetize.then_some(ProdConfig::production()),
            );
        }
        if let Some(slice_end) = zero_nondata_slice_end(view) {
            write_zero_nondata_metadata(&round_path, p.n, total_wires, slice_end);
        }
        println!("[sss:cnot] round circuit written to {round_path}");
    }

    let mut samf_requested = if p.rounds == 0 {
        0
    } else {
        p.m.saturating_mul(gates.len().div_ceil(p.x))
    };
    if p.full_shuffle || p.full_shuffle_early {
        samf_requested = samf_requested.max(total_wires);
    }
    if p.do_tdp4n && samf_requested > 0 {
        println!(
            "[sss:cnot] strict 4n TDP: skipping final masked-SAMF pass because it requires an additional helper wire"
        );
    } else if samf_requested > 0 {
        assert!(
            total_wires < 1024,
            "masked native SAMFs require one additional random helper wire"
        );
        let before_samf = gates.clone();
        let mask_wire = total_wires as u16;
        let inserted =
            insert_masked_swap_samfs(&mut gates, total_wires, mask_wire, samf_requested, &mut rng);
        if inserted > 0 {
            total_wires += 1;
            println!(
                "[sss:cnot] native SAMF/unsamf: {} disjoint masked-swap brackets, {} CNOTs, independent helper wire {}",
                inserted,
                inserted * 10,
                mask_wire
            );
            full_equivalence_check(
                &before_samf,
                &gates,
                total_wires,
                if p.equality_check { 10_000 } else { 256 },
                base_seed ^ 0x5a4f_5a4f,
            )
            .expect("native SAMF/unsamf changed the circuit");
            functionality_check(
                original,
                &gates,
                view,
                p.n,
                total_wires,
                fixed_slice,
                if p.equality_check { 10_000 } else { 256 },
                base_seed ^ 0x5a4f_c0de,
            )
            .expect("native SAMF/unsamf changed required functionality");
            let metadata = format!(
                "mode=masked_swap_samf\nrandom_mask_wire={}\nrequirement=independent uniform random input bit for prefix masking\n",
                mask_wire
            );
            ensure_parent(p.save);
            std::fs::write(format!("{}.samf_mask", p.save), metadata)
                .expect("write native SAMF mask metadata");
        }
    }

    assert_tdp_namespace(
        p.do_tdp4n,
        p.do_nonlinear_gadgetize,
        &gates,
        total_wires,
        p.n,
        "before final write",
    );
    ensure_parent(p.save);
    format::write_mpmct(p.save, &gates, total_wires).expect("write final CNOT circuit");
    if let Some((public_y, public_z)) = &public_slice_words {
        write_slice_metadata(
            p.save,
            p.n,
            total_wires,
            gates.len(),
            p.slice_zero_random_gates,
            construction_seed,
            p.rg_freq,
            original.gates.len(),
            &source_repr_xxh3_128,
            public_y,
            public_z,
            public_slice_layout.expect("public slice layout"),
            p.do_nonlinear_gadgetize.then_some(ProdConfig::production()),
        );
    }
    if let Some(slice_end) = zero_nondata_slice_end(view) {
        write_zero_nondata_metadata(p.save, p.n, total_wires, slice_end);
    }
    print_counts("final", &gates);
    println!("[sss:cnot] final mpmct1 circuit written to {}", p.save);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postmix::xgate::eval_u1024;

    fn scalar_functionality_check(
        original: &CircuitSeq,
        transformed: &[XGate],
        view: FunctionView,
        n: usize,
        total_wires: usize,
        fixed_slice: Option<(U1024, U1024)>,
        samples: usize,
        seed: u64,
    ) -> Result<(), String> {
        let low_mask = mask(n);
        let mut rng = StdRng::seed_from_u64(seed);
        let exhaustive = total_wires <= EXHAUSTIVE_WIRE_LIMIT;
        let count = if exhaustive {
            1usize << total_wires
        } else {
            samples
        };
        for index in 0..count {
            let mut input = if exhaustive {
                U1024::from(index)
            } else {
                random_u1024(&mut rng) & mask(total_wires)
            };
            if let FunctionView::GadgetLowZeroNonData { slice_end } = view {
                let fixed_mask = mask(slice_end) ^ low_mask;
                input &= !fixed_mask;
            }
            if matches!(
                view,
                FunctionView::FeistalMiddle | FunctionView::Tdp4nMiddle
            ) {
                if let Some((public_y, public_z)) = fixed_slice {
                    let fixed_mask = (low_mask << n) | (low_mask << (2 * n));
                    input &= !fixed_mask;
                    input |= public_y << n;
                    input |= public_z << (2 * n);
                }
            }

            let logical_x = input & low_mask;
            let original_output = original.evaluate_1024(logical_x) & low_mask;
            let actual = eval_u1024(transformed, input);
            let matches = match view {
                FunctionView::Whole => actual == original.evaluate_1024(input),
                FunctionView::GadgetLow | FunctionView::GadgetLowZeroNonData { .. } => {
                    actual & low_mask == original_output
                }
                FunctionView::FeistalMiddle | FunctionView::Tdp4nMiddle => {
                    let y = (input >> n) & low_mask;
                    ((actual >> n) & low_mask) == (y ^ original_output)
                }
            };
            if !matches {
                return Err(format!(
                    "functionality mismatch in {:?} view at sampled input 0x{:x}",
                    view, input
                ));
            }
        }
        Ok(())
    }

    fn scalar_full_equivalence_check(
        before: &[XGate],
        after: &[XGate],
        total_wires: usize,
        samples: usize,
        seed: u64,
    ) -> Result<(), String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let exhaustive = total_wires <= EXHAUSTIVE_WIRE_LIMIT;
        let count = if exhaustive {
            1usize << total_wires
        } else {
            samples
        };
        for index in 0..count {
            let input = if exhaustive {
                U1024::from(index)
            } else {
                random_u1024(&mut rng) & mask(total_wires)
            };
            if eval_u1024(before, input) != eval_u1024(after, input) {
                return Err(format!(
                    "heterogeneous rewrite mismatch at input 0x{input:x}"
                ));
            }
        }
        Ok(())
    }

    fn source_xgates(source: &CircuitSeq) -> Vec<XGate> {
        source.gates.iter().copied().map(XGate::from_g57).collect()
    }

    fn xor_single_source_gate_into_middle(source: &CircuitSeq, n: usize) -> Vec<XGate> {
        assert_eq!(source.gates.len(), 1, "test helper assumes one source gate");
        let mut gates = (0..n)
            .map(|wire| XGate::cnot((n + wire) as u16, wire as u16))
            .collect::<Vec<_>>();
        let mut source_delta = XGate::from_g57(source.gates[0]);
        source_delta.target += n as u16;
        gates.push(source_delta);
        gates
    }

    #[test]
    fn lane_functionality_matches_scalar_for_every_view_and_first_error() {
        let source = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let reference = source_xgates(&source);
        let mut zero_nondata = reference.clone();
        // This extra CNOT is inert only because the production zero-slice view
        // fixes wire n through slice_end to zero.
        zero_nondata.push(XGate::cnot(0, 5));
        let cases = [
            (FunctionView::Whole, 3, 6, None, reference.clone(), 0),
            (FunctionView::GadgetLow, 7, 14, None, reference.clone(), 0),
            (
                FunctionView::GadgetLowZeroNonData { slice_end: 11 },
                5,
                14,
                None,
                zero_nondata,
                0,
            ),
            (
                FunctionView::FeistalMiddle,
                5,
                15,
                Some((U1024::from(0x15u64), U1024::from(0x0bu64))),
                xor_single_source_gate_into_middle(&source, 5),
                5,
            ),
            (
                FunctionView::Tdp4nMiddle,
                4,
                16,
                Some((U1024::from(0x09u64), U1024::from(0x06u64))),
                xor_single_source_gate_into_middle(&source, 4),
                4,
            ),
        ];

        for (view, n, total_wires, fixed_slice, transformed, corrupted_wire) in cases {
            let scalar = scalar_functionality_check(
                &source,
                &transformed,
                view,
                n,
                total_wires,
                fixed_slice,
                130,
                0x5eed_1234,
            );
            let lanes = functionality_check(
                &source,
                &transformed,
                view,
                n,
                total_wires,
                fixed_slice,
                130,
                0x5eed_1234,
            );
            assert!(scalar.is_ok(), "valid {view:?} fixture failed: {scalar:?}");
            assert_eq!(lanes, scalar, "valid {view:?} fixture differed");

            let mut corrupted = transformed;
            corrupted.push(XGate::x_gate(corrupted_wire));
            let scalar = scalar_functionality_check(
                &source,
                &corrupted,
                view,
                n,
                total_wires,
                fixed_slice,
                130,
                0x5eed_1234,
            );
            let lanes = functionality_check(
                &source,
                &corrupted,
                view,
                n,
                total_wires,
                fixed_slice,
                130,
                0x5eed_1234,
            );
            assert!(
                scalar.is_err(),
                "corrupt {view:?} fixture unexpectedly passed"
            );
            assert_eq!(lanes, scalar, "corrupt {view:?} first error differed");
        }
    }

    #[test]
    fn lane_full_equivalence_matches_scalar_and_first_error() {
        let before = vec![XGate::from_g57([0, 1, 2]), XGate::cnot(8, 3)];
        for (total_wires, samples) in [(6, 7), (17, 130)] {
            let scalar =
                scalar_full_equivalence_check(&before, &before, total_wires, samples, 0xe9a1_5eed);
            let lanes = full_equivalence_check(&before, &before, total_wires, samples, 0xe9a1_5eed);
            assert!(scalar.is_ok());
            assert_eq!(lanes, scalar);

            let mut corrupted = before.clone();
            // Full equivalence historically compares all 1024 state bits, not
            // just the declared input width. Keep that behavior covered.
            corrupted.push(XGate::x_gate(999));
            let scalar = scalar_full_equivalence_check(
                &before,
                &corrupted,
                total_wires,
                samples,
                0xe9a1_5eed,
            );
            let lanes =
                full_equivalence_check(&before, &corrupted, total_wires, samples, 0xe9a1_5eed);
            assert!(scalar.is_err());
            assert_eq!(lanes, scalar, "first mismatch input or error text differed");
        }
    }

    #[test]
    fn zero_slice_metadata_leaves_later_samf_helpers_independent() {
        let dir = std::env::temp_dir().join(format!(
            "local_mixing_zero_slice_metadata_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("out.mpmct1");
        write_zero_nondata_metadata(output.to_str().unwrap(), 3, 13, 12);
        let metadata =
            std::fs::read_to_string(format!("{}.slice_zero_ccnot", output.to_str().unwrap()))
                .unwrap();
        assert!(metadata.contains("fixed_input_wires=3..12\n"));
        assert!(metadata.contains("independent_helper_wires=12..13\n"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_round_driver_writes_mpmct_and_preserves_views() {
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        for (gadgetize, nonlinear_gadgetize, feistalize, tdp4n, expected_wires) in [
            (true, false, false, false, 6),
            (false, true, false, false, 9),
            (false, false, true, false, 9),
            (false, false, false, true, 12),
            (false, true, false, true, 18),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "local_mixing_cnot_driver_{}_{}_{}_{}_{}",
                std::process::id(),
                gadgetize as u8,
                nonlinear_gadgetize as u8,
                feistalize as u8,
                tdp4n as u8,
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let output = dir.join("out.txt");
            let gadget = dir.join("gadget.txt");
            let params = CnotSssParams {
                rounds: 0,
                n: 3,
                m: 1,
                x: 2,
                save: output.to_str().unwrap(),
                source: "source.txt",
                do_gadgetize: gadgetize,
                do_nonlinear_gadgetize: nonlinear_gadgetize,
                do_feistalize: feistalize,
                do_tdp4n: tdp4n,
                slice_zero: false,
                slice_zero_random: false,
                slice_zero_random_gates: 96,
                slice_zero_hardcoded: false,
                slice_zero_hardcoded_rounds: 1,
                gadget_path: Some(gadget.to_str().unwrap()),
                full_shuffle: false,
                full_shuffle_early: false,
                shooting_times: 1,
                collision_rounds: 1,
                stable_compressions: 1,
                expansion_game: false,
                equality_check: false,
                rg_freq: 2,
            };
            main_shuffle_shoot_shuffle_cnot(&source, &params);
            let (written, wires) = format::read_mpmct(output.to_str().unwrap()).unwrap();
            assert_eq!(wires, expected_wires);
            assert!(!written.is_empty());
            if nonlinear_gadgetize && !tdp4n {
                let metadata = std::fs::read_to_string(format!(
                    "{}.slice_zero_ccnot",
                    output.to_str().unwrap()
                ))
                .unwrap();
                assert!(metadata.contains("fixed_input_wires=3..9\n"));
                assert!(metadata.contains("independent_helper_wires=9..9\n"));
                assert!(metadata.contains("nonlinear_plan=2,2,2,3\n"));
                assert!(metadata.contains("nonlinear_fold=gray\n"));
                assert!(metadata.contains("nonlinear_single_carrier=true\n"));
                assert!(metadata.contains("band_is_in_fixed_slice=true\n"));
            }
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn tdp4n_random_slice_metadata_identifies_w_as_a_free_helper_block() {
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        let expected_source_fingerprint = format!(
            "{:032x}",
            xxhash_rust::xxh3::xxh3_128(source.repr().as_bytes())
        );
        let dir = std::env::temp_dir().join(format!(
            "local_mixing_cnot_tdp4n_metadata_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (nonlinear, expected_wires) in [(false, 12), (true, 18)] {
            let output = dir.join(format!("out_{}.mpmct1", nonlinear as u8));
            let gadget = dir.join(format!("gadget_{}.mpmct1", nonlinear as u8));
            let params = CnotSssParams {
                rounds: 0,
                n: 3,
                m: 1,
                x: 2,
                save: output.to_str().unwrap(),
                source: "source.txt",
                do_gadgetize: false,
                do_nonlinear_gadgetize: nonlinear,
                do_feistalize: false,
                do_tdp4n: true,
                slice_zero: false,
                slice_zero_random: true,
                slice_zero_random_gates: 96,
                slice_zero_hardcoded: false,
                slice_zero_hardcoded_rounds: 1,
                gadget_path: Some(gadget.to_str().unwrap()),
                full_shuffle: false,
                full_shuffle_early: false,
                shooting_times: 1,
                collision_rounds: 1,
                stable_compressions: 1,
                expansion_game: false,
                equality_check: false,
                rg_freq: 2,
            };
            main_shuffle_shoot_shuffle_cnot(&source, &params);

            let metadata =
                std::fs::read_to_string(format!("{}.slice_zero_random", output.to_str().unwrap()))
                    .unwrap();
            let (artifact_gates, _) = format::read_mpmct(output.to_str().unwrap()).unwrap();
            let expected_layout = if nonlinear {
                "layout=tdp4n_single_carrier_2223_gray_padded\n"
            } else {
                "layout=tdp4n_two_share\n"
            };
            assert!(metadata.contains(expected_layout));
            assert!(metadata.contains(&format!("total_wires={expected_wires}\n")));
            assert!(metadata.contains(&format!("gates={}\n", artifact_gates.len())));
            assert!(metadata.contains("slice_preblock_gates=96\n"));
            assert!(metadata.contains("construction_seed="));
            assert!(metadata.contains("rg_frequency=2\n"));
            assert!(metadata.contains("source_gates=2\n"));
            assert!(metadata.contains(&format!(
                "source_repr_xxh3_128={expected_source_fingerprint}\n"
            )));
            assert!(metadata.contains("w_wires=9..12\n"));
            assert!(metadata.contains(&format!("sat_helper_wires=9..{expected_wires}\n")));
            assert!(metadata.contains(&format!("extra_helper_wires=12..{expected_wires}\n")));
            assert!(metadata.contains("fixed_input_blocks=y,z\n"));
            if nonlinear {
                for expected in [
                    "nonlinear_enabled=true\n",
                    "nonlinear_config_version=2223-gray-v1\n",
                    "nonlinear_plan=2,2,2,3\n",
                    "nonlinear_fold=gray\n",
                    "nonlinear_single_carrier=true\n",
                    "nonlinear_k=3\n",
                    "nonlinear_deg=2\n",
                    "nonlinear_k_hi=1\n",
                    "nonlinear_deg_hi=3\n",
                    "nonlinear_band_config=0\n",
                    "nonlinear_band_size=6\n",
                    "nonlinear_band_wires=12..18\n",
                    "nonlinear_rsrc=1\n",
                    "nonlinear_max_width=0\n",
                    "nonlinear_fill_nl=2\n",
                    "nonlinear_roll=1\n",
                ] {
                    assert!(metadata.contains(expected), "missing metadata: {expected}");
                }
            } else {
                assert!(metadata.contains("nonlinear_enabled=false\n"));
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn strict_tdp4n_full_shuffle_never_adds_a_helper_wire_or_samf_mask() {
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        let dir = std::env::temp_dir().join(format!(
            "local_mixing_cnot_tdp4n_no_samf_helper_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("out.mpmct1");
        let params = CnotSssParams {
            rounds: 0,
            n: 3,
            m: 1,
            x: 2,
            save: output.to_str().unwrap(),
            source: "source.txt",
            do_gadgetize: false,
            do_nonlinear_gadgetize: false,
            do_feistalize: false,
            do_tdp4n: true,
            slice_zero: false,
            slice_zero_random: false,
            slice_zero_random_gates: 96,
            slice_zero_hardcoded: false,
            slice_zero_hardcoded_rounds: 1,
            gadget_path: None,
            full_shuffle: true,
            full_shuffle_early: true,
            shooting_times: 1,
            collision_rounds: 1,
            stable_compressions: 1,
            expansion_game: false,
            equality_check: true,
            rg_freq: 2,
        };
        main_shuffle_shoot_shuffle_cnot(&source, &params);

        let (written, wires) = format::read_mpmct(output.to_str().unwrap()).unwrap();
        assert_eq!(wires, 4 * params.n);
        assert!(written.iter().all(|gate| gate.max_wire() < wires as u16));
        assert!(
            !Path::new(&format!("{}.samf_mask", output.to_str().unwrap())).exists(),
            "strict 4n output must not emit masked-SAMF helper metadata"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[should_panic(expected = "strict 4n TDP control wire")]
    fn strict_tdp4n_guard_rejects_out_of_range_gate_wires() {
        let gate = XGate::cnot(0, 12);
        assert_tdp_namespace(true, false, &[gate], 12, 3, "in test");
    }

    /// The production TDP circuit ([2,2,2,3] product-share + Gray fold +
    /// single carrier) end to end: the real constructor, the real driver
    /// check, EXHAUSTIVELY over all 2^18 inputs.
    ///
    /// Pins three things that were previously only asserted in prose:
    ///  * the width is exactly 6n — 4n carriers plus a band_size(2n) = 2n band
    ///    — not "a bit over 4n" (that figure belongs to the retired [2,3,3]
    ///    band-56 preset);
    ///  * C's functionality lands on wires n..2n in ACCUMULATE form,
    ///    out[n+i] ^ in[n+i] == C(in)[i], with the X, W and band blocks left
    ///    free — so the check must pass with those wires driven randomly;
    ///  * a corruption inside the checked band is caught.
    #[test]
    fn production_tdp_is_6n_and_exhaustively_verified_on_the_middle_band() {
        let n = 3;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        };
        let mut rng = StdRng::seed_from_u64(0x7d9_0001);
        let built = crate::replace::gadgets::tdp4n_nonlinear_cnot(&source, n, 1, &mut rng);
        let total_wires = built.num_wires;

        assert_eq!(
            total_wires,
            6 * n,
            "production nonlinear TDP must be 6n wires (4n carriers + 2n band)"
        );
        // The driver's own namespace guard must agree.
        assert_tdp_namespace(true, true, &built.gates, total_wires, n, "in test");
        // 6n = 18 at n = 3, so this is at the exhaustive threshold: the check
        // below enumerates every input rather than sampling.
        assert!(total_wires <= 18, "n=3 TDP must be exhaustively verifiable");

        functionality_check(
            &source,
            &built.gates,
            FunctionView::Tdp4nMiddle,
            n,
            total_wires,
            None,
            0, // ignored: exhaustive mode enumerates 2^18 inputs
            0x7d9_0002,
        )
        .expect("production TDP failed its own middle-band contract");

        // Negative control: a corruption INSIDE the checked block must fail.
        // (Wire n = the first middle-band carrier.)
        let mut corrupted = built.gates.clone();
        corrupted.push(XGate::x_gate(n as u16));
        assert!(
            functionality_check(
                &source,
                &corrupted,
                FunctionView::Tdp4nMiddle,
                n,
                total_wires,
                None,
                0,
                0x7d9_0002,
            )
            .is_err(),
            "a corrupted middle band must be rejected"
        );

        // Documented blind spot, pinned so it cannot regress silently into a
        // false sense of coverage: the middle-band view examines n of 6n
        // wires, so a gate touching only the free blocks (X = D(C(x)), the
        // dead W block, or the product band) passes. That is BY DESIGN — those
        // wires carry no contract — but it means this check alone does not
        // establish that the artifact leaks nothing. See
        // verify_sliced_sandwich_zero_slice / slice_check_4n for the
        // whole-artifact checks.
        let mut off_band = built.gates.clone();
        off_band.push(XGate::x_gate((4 * n) as u16)); // first band wire
        assert!(
            functionality_check(
                &source,
                &off_band,
                FunctionView::Tdp4nMiddle,
                n,
                total_wires,
                None,
                0,
                0x7d9_0002,
            )
            .is_ok(),
            "band wires are outside the middle-band contract by construction"
        );
    }
}

#[cfg(test)]
mod tdp_semantics {
    use super::*;
    use crate::postmix::xgate::eval_u1024;

    fn m(bits: usize) -> U1024 {
        (U1024::one() << bits) - U1024::one()
    }
    fn blk(v: U1024, lo: usize, hi: usize) -> U1024 {
        (v >> lo) & m(hi - lo)
    }

    /// The measured input-to-output contract of the production TDP, pinned.
    ///
    /// On input (x, y, z, w, b0, b1) the 6n-wire artifact computes
    ///   X -> f(x)            junk-looking, but a function of x ALONE
    ///   Y -> y XOR C(x)      the payload
    ///   Z -> z               identity
    ///   W -> w               identity
    ///   band -> g(x, y, band)  dirty workspace; DOES carry x/y information
    ///
    /// Nothing here is random: a circuit is deterministic, so "random" in
    /// discussion means "junk-looking", not unpredictable. The dependency
    /// structure is the security-relevant part and is asserted below.
    #[test]
    fn tdp_semantics_dependency_map() {
        let n = 3;
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        };
        let mut rng = StdRng::seed_from_u64(0x7d9_0001);
        let built = crate::replace::gadgets::tdp4n_nonlinear_cnot(&source, n, 1, &mut rng);
        let tw = built.num_wires;
        // Blocks, in wire order.
        let names = [
            "X(0..n)",
            "Y(n..2n)",
            "Z(2n..3n)",
            "W(3n..4n)",
            "B0(4n..5n)",
            "B1(5n..6n)",
        ];
        let bounds: Vec<(usize, usize)> = (0..6).map(|i| (i * n, (i + 1) * n)).collect();

        // For each (out_block, in_block): does varying ONLY in_block ever move out_block?
        let mut probe = StdRng::seed_from_u64(1234);
        let mut depends = vec![vec![false; 6]; 6];
        for _ in 0..200 {
            let mut base = U1024::zero();
            for b in 0..tw {
                if probe.next_u32() & 1 == 1 {
                    base |= U1024::one() << b;
                }
            }
            let base_out = eval_u1024(&built.gates, base);
            for (bi, &(lo, hi)) in bounds.iter().enumerate() {
                // flip one random bit inside this input block
                let bit = lo + (probe.next_u32() as usize % (hi - lo));
                let alt = base ^ (U1024::one() << bit);
                let alt_out = eval_u1024(&built.gates, alt);
                for (oi, &(olo, ohi)) in bounds.iter().enumerate() {
                    if blk(base_out, olo, ohi) != blk(alt_out, olo, ohi) {
                        depends[oi][bi] = true;
                    }
                }
            }
        }
        println!("\n=== OUTPUT block depends on INPUT block? (rows=out, cols=in) ===");
        println!(
            "{:>10} {}",
            "",
            names.map(|s| format!("{:>10}", s)).join("")
        );
        for (oi, name) in names.iter().enumerate() {
            let row: String = (0..6)
                .map(|bi| format!("{:>10}", if depends[oi][bi] { "YES" } else { "." }))
                .collect();
            println!("{name:>10} {row}");
        }

        // The headline claim: out[Y] == in[Y] XOR C(in[X]) for all inputs.
        let mut y_ok = true;
        let mut w_id = true;
        for _ in 0..500 {
            let mut input = U1024::zero();
            for b in 0..tw {
                if probe.next_u32() & 1 == 1 {
                    input |= U1024::one() << b;
                }
            }
            let out = eval_u1024(&built.gates, input);
            // C evaluated on x alone, everything else zero
            let mut cs = vec![0u64; 64];
            for i in 0..n {
                cs[i] = if (blk(input, 0, n) >> i) & U1024::one() == U1024::one() {
                    u64::MAX
                } else {
                    0
                };
            }
            let mut c_out = U1024::zero();
            let cx = source.evaluate_1024(blk(input, 0, n));
            for i in 0..n {
                if (cx >> i) & U1024::one() == U1024::one() {
                    c_out |= U1024::one() << i;
                }
            }
            if blk(out, n, 2 * n) != (blk(input, n, 2 * n) ^ c_out) {
                y_ok = false;
            }
            if blk(out, 3 * n, 4 * n) != blk(input, 3 * n, 4 * n) {
                w_id = false;
            }
        }
        // Is Z identity as well, and does the band carry x/y information?
        let mut z_id = true;
        let mut band_moves_with_x = false;
        for _ in 0..300 {
            let mut input = U1024::zero();
            for b in 0..tw {
                if probe.next_u32() & 1 == 1 {
                    input |= U1024::one() << b;
                }
            }
            let out = eval_u1024(&built.gates, input);
            if blk(out, 2 * n, 3 * n) != blk(input, 2 * n, 3 * n) {
                z_id = false;
            }
            let alt_out = eval_u1024(&built.gates, input ^ U1024::one());
            if blk(out, 4 * n, tw) != blk(alt_out, 4 * n, tw) {
                band_moves_with_x = true;
            }
        }
        println!("\nout[Y] == in[Y] XOR C(in[X]) for all sampled inputs : {y_ok}");
        println!("out[W] == in[W] (identity)                          : {w_id}");
        println!("out[Z] == in[Z] (identity)                          : {z_id}");
        println!("band output moves when one bit of x flips           : {band_moves_with_x}");
        assert!(y_ok, "the TDP middle-band contract must hold exactly");
        assert!(w_id, "W must be pass-through");
        assert!(z_id, "Z must be pass-through");
        assert!(
            band_moves_with_x,
            "the band is expected to carry x information"
        );

        // The dependency structure itself, pinned. A change here means the
        // construction leaks somewhere new (or stopped leaking somewhere).
        let expect = [
            //  X      Y      Z      W     B0     B1
            [true, false, false, false, false, false], // out X
            [true, true, false, false, false, false],  // out Y
            [false, false, true, false, false, false], // out Z
            [false, false, false, true, false, false], // out W
            [true, true, false, false, true, true],    // out B0
            [true, true, false, false, true, true],    // out B1
        ];
        for oi in 0..6 {
            assert_eq!(
                depends[oi], expect[oi],
                "dependency row for output block {} changed",
                names[oi]
            );
        }
    }
}
