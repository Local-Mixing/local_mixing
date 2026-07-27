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
        xgate::{XGate, eval_u1024},
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
            ("tdp4n_two_share", format!("{}..{}", three_n, 4 * n), 4 * n)
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
             nonlinear_plan={plan}\n\
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
    let exhaustive = total_wires <= 12;
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

fn full_equivalence_check(
    before: &[XGate],
    after: &[XGate],
    total_wires: usize,
    samples: usize,
    seed: u64,
) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let exhaustive = total_wires <= 12;
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
        assert!(
            total_wires > base_wires,
            "nonlinear TDP must append a source band after 4n wires {stage}"
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
                    "slice-zero-random nonlinear 4n+band TDP"
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
                    "nonlinear 4n+band TDP"
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
            "slice-zero production nonlinear product-share gadgetized",
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
        0xc001_c0de,
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
            (false, true, false, false, 12),
            (false, false, true, false, 9),
            (false, false, false, true, 12),
            (false, true, false, true, 19),
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
                equality_check: true,
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
                assert!(metadata.contains("fixed_input_wires=3..12\n"));
                assert!(metadata.contains("independent_helper_wires=12..12\n"));
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
        for (nonlinear, expected_wires) in [(false, 12), (true, 19)] {
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
                equality_check: true,
                rg_freq: 2,
            };
            main_shuffle_shoot_shuffle_cnot(&source, &params);

            let metadata =
                std::fs::read_to_string(format!("{}.slice_zero_random", output.to_str().unwrap()))
                    .unwrap();
            let (artifact_gates, _) = format::read_mpmct(output.to_str().unwrap()).unwrap();
            assert!(metadata.contains("layout=tdp4n_two_share\n"));
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
                    "nonlinear_plan=3,3\n",
                    "nonlinear_k=0\n",
                    "nonlinear_deg=2\n",
                    "nonlinear_k_hi=2\n",
                    "nonlinear_deg_hi=3\n",
                    "nonlinear_band_config=0\n",
                    "nonlinear_band_size=7\n",
                    "nonlinear_band_wires=12..19\n",
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
}
