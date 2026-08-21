//! Heterogeneous `sss --cnot` driver.
//!
//! The source file is still the legacy G57/base-83 format. From the requested
//! transformation onward this driver keeps the richer [`XGate`] representation
//! so native CNOTs and conjunction fragments are never coerced back into G57
//! helper sequences.

use std::path::Path;

use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

use crate::{
    circuit::{
        CircuitSeq, U1024,
        xgate::{XGate, eval_u1024},
    },
    engine::{
        format,
        mix::{MixParams, Mixer},
    },
    postprocessing::compress::{CompressParams, compress},
    preprocessing::{
        gadgets::{
            CnotCircuit, MaskConfig, ProdConfig, feistalize_cnot, feistalize_with_slice_zero_cnot,
            feistalize_with_slice_zero_hardcoded_cnot, feistalize_with_slice_zero_random_cnot,
            gadgetize_cnot, gadgetize_cnot_five_carrier, gadgetize_cnot_seven_carrier,
            gadgetize_cnot_single, gadgetize_cnot_six_carrier, gadgetize_cnot_strong_five_carrier,
            gadgetize_cnot_strong_six_carrier, gadgetize_with_slice_zero_ccnot,
            gadgetize_with_slice_zero_ccnot_five_carrier,
            gadgetize_with_slice_zero_ccnot_seven_carrier,
            gadgetize_with_slice_zero_ccnot_six_carrier,
            gadgetize_with_slice_zero_ccnot_strong_five_carrier,
            gadgetize_with_slice_zero_ccnot_strong_six_carrier, packed_bit, sliced_sandwich_cnot,
        },
        samf::insert_masked_swap_samfs,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FunctionView {
    Whole,
    GadgetLow,
    FeistalMiddle,
    // Sliced sandwich: on the zero slice (second half = 0) the answer C(x)
    // exits on the HIGH n wires.
    SandwichSecond,
}

pub struct CnotSssParams<'a> {
    pub rounds: usize,
    pub n: usize,
    pub m: usize,
    pub x: usize,
    pub save: &'a str,
    pub source: &'a str,
    pub do_gadgetize: bool,
    pub five_carrier: bool,
    pub strong_five_carrier: bool,
    pub six_carrier: bool,
    pub strong_six_carrier: bool,
    pub seven_carrier: bool,
    pub do_feistalize: bool,
    pub slice_zero: bool,
    pub slice_zero_random: bool,
    pub slice_zero_random_gates: usize,
    pub slice_zero_hardcoded: bool,
    pub slice_zero_hardcoded_rounds: usize,
    pub slice_zero_ccnot: bool,
    pub slice_zero_ccnot_gates: usize,
    pub sliced_sandwich: bool,
    pub sandwich_m: usize,
    pub sandwich_s: usize,
    pub gadget_path: Option<&'a str>,
    pub full_shuffle: bool,
    pub full_shuffle_early: bool,
    pub shooting_times: usize,
    pub collision_rounds: usize,
    pub stable_compressions: usize,
    pub expansion_game: bool,
    pub equality_check: bool,
    pub rg_freq: usize,
    pub masks: MaskConfig,
    pub prod: ProdConfig,
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
    gate_count: usize,
    public_y: &[u64],
    public_z: &[u64],
) {
    let y = packed_words_to_hex(public_y, n);
    let z = packed_words_to_hex(public_z, n);
    let meta_path = format!("{path}.slice_zero_random");
    let contents = format!(
        "mode=slice_zero_random\n\
         representation=mpmct1\n\
         n={n}\n\
         gates={gate_count}\n\
         y_hex={y}\n\
         z_hex={z}\n\
         bit_order=bit i is wire n+i for y and wire 2n+i for z\n"
    );
    std::fs::write(&meta_path, contents).expect("write slice-zero-random metadata");
    println!("[sss:cnot] public slice Y={y} Z={z} ({meta_path})");
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
        if let Some((public_y, public_z)) = fixed_slice {
            match view {
                FunctionView::FeistalMiddle => {
                    input &= low_mask;
                    input |= public_y << n;
                    input |= public_z << (2 * n);
                }
                // Gadget slice width is representation-specific (band only,
                // paired aux+band, or four extra carrier lanes+band). Pinning
                // everything above the logical low n wires covers all modes.
                FunctionView::GadgetLow => {
                    input &= low_mask;
                    input |= public_y << n;
                }
                // The sliced sandwich's slice is the zero second half; pin
                // it (public_y is 0 here).
                FunctionView::SandwichSecond => {
                    input &= low_mask;
                    input |= public_y << n;
                }
                FunctionView::Whole => {}
            }
        }

        let logical_x = input & low_mask;
        let original_output = original.evaluate_1024(logical_x) & low_mask;
        let actual = eval_u1024(transformed, input);
        let matches = match view {
            FunctionView::Whole => actual == original.evaluate_1024(input),
            FunctionView::GadgetLow => actual & low_mask == original_output,
            FunctionView::FeistalMiddle => {
                let y = (input >> n) & low_mask;
                ((actual >> n) & low_mask) == (y ^ original_output)
            }
            // On the zero slice the answer C(x) lands on the high n wires.
            FunctionView::SandwichSecond => (actual >> n) & low_mask == original_output,
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

/// Run the XGate-native shuffle/shoot/shuffle path selected by `sss --cnot`.
pub fn main_shuffle_shoot_shuffle_cnot(original: &CircuitSeq, p: &CnotSssParams<'_>) {
    assert!(p.x > 0, "--x must be nonzero");
    assert!(p.n > 0, "--n must be nonzero");
    assert!(p.rg_freq > 0, "--rg-frequency must be nonzero");
    assert!(
        !p.five_carrier || p.do_gadgetize,
        "--five-carrier requires --gadgetize"
    );
    assert!(
        !p.strong_five_carrier || p.do_gadgetize,
        "--strong-five-carrier requires --gadgetize"
    );
    assert!(
        !p.six_carrier || p.do_gadgetize,
        "--six-carrier requires --gadgetize"
    );
    assert!(
        !p.strong_six_carrier || p.do_gadgetize,
        "--strong-six-carrier requires --gadgetize"
    );
    assert!(
        !p.seven_carrier || p.do_gadgetize,
        "--seven-carrier requires --gadgetize"
    );
    assert!(
        (p.five_carrier as u8)
            + (p.strong_five_carrier as u8)
            + (p.six_carrier as u8)
            + (p.strong_six_carrier as u8)
            + (p.seven_carrier as u8)
            <= 1,
        "nonlinear carrier representation flags are mutually exclusive"
    );
    assert!(
        (p.do_gadgetize as u8) + (p.do_feistalize as u8) + (p.sliced_sandwich as u8) <= 1,
        "--gadgetize, --feistalize and --sliced-sandwich are mutually exclusive"
    );
    if p.five_carrier || p.strong_five_carrier {
        assert!(
            p.prod.enabled(),
            "five-carrier modes require a nonempty product-mask plan"
        );
        assert!(
            !p.prod.dist(),
            "five-carrier modes do not support distributed product-mask sourcing"
        );
        let expected_wires =
            p.n.checked_mul(5)
                .and_then(|carriers| carriers.checked_add(p.prod.band_size(p.n)))
                .expect("five-carrier wire count overflow");
        assert!(
            expected_wires <= 1024,
            "the selected five-carrier mode produces {expected_wires} wires (5n carriers plus band), but the --cnot driver supports at most 1024"
        );
    }
    if p.six_carrier || p.strong_six_carrier {
        assert!(
            p.prod.enabled(),
            "six-carrier modes require a nonempty product-mask plan"
        );
        assert!(
            !p.prod.dist(),
            "six-carrier modes do not support distributed product-mask sourcing"
        );
        let expected_wires =
            p.n.checked_mul(6)
                .and_then(|carriers| carriers.checked_add(p.prod.band_size(p.n)))
                .expect("six-carrier wire count overflow");
        assert!(
            expected_wires <= 1024,
            "the selected six-carrier mode produces {expected_wires} wires (6n carriers plus band), but the --cnot driver supports at most 1024"
        );
    }
    if p.seven_carrier {
        assert!(
            p.prod.enabled(),
            "--seven-carrier requires a nonempty product-mask plan"
        );
        assert!(
            !p.prod.dist(),
            "--seven-carrier does not support distributed product-mask sourcing"
        );
        let expected_wires =
            p.n.checked_mul(7)
                .and_then(|carriers| carriers.checked_add(p.prod.band_size(p.n)))
                .expect("seven-carrier wire count overflow");
        assert!(
            expected_wires <= 1024,
            "--seven-carrier produces {expected_wires} wires (7n carriers plus band), but the --cnot driver supports at most 1024"
        );
    }
    println!(
        "[sss:cnot] XGate-native backend selected: G57 ingress, heterogeneous mpmct1 thereafter"
    );
    if p.full_shuffle || p.full_shuffle_early {
        println!("[sss:cnot] full-shuffle requests an enlarged final masked-SAMF pass");
    }
    if p.expansion_game {
        println!("[sss:cnot] expansion-game maps to native fresh-wire fragment splits");
    }

    let mut rng = rand::rng();
    let mut public_slice_words: Option<(Vec<u64>, Vec<u64>)> = None;
    let mut fixed_slice = None;
    let (transformed, view, label): (CnotCircuit, FunctionView, &str) = if p.do_feistalize {
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
        if p.seven_carrier && p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot_seven_carrier(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized (seven-carrier decode)",
            )
        } else if p.seven_carrier {
            (
                gadgetize_cnot_seven_carrier(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (seven-carrier decode)",
            )
        } else if p.strong_six_carrier && p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot_strong_six_carrier(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized (strong structural six-carrier decode)",
            )
        } else if p.strong_six_carrier {
            (
                gadgetize_cnot_strong_six_carrier(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (strong structural six-carrier decode)",
            )
        } else if p.six_carrier && p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot_six_carrier(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized (six-carrier decode)",
            )
        } else if p.six_carrier {
            (
                gadgetize_cnot_six_carrier(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (six-carrier decode)",
            )
        } else if p.strong_five_carrier && p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot_strong_five_carrier(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized (strong cubic five-carrier decode)",
            )
        } else if p.strong_five_carrier {
            (
                gadgetize_cnot_strong_five_carrier(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (strong cubic five-carrier decode)",
            )
        } else if p.five_carrier && p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot_five_carrier(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized (five-carrier decode)",
            )
        } else if p.five_carrier {
            (
                gadgetize_cnot_five_carrier(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (five-carrier decode)",
            )
        } else if p.slice_zero_ccnot {
            fixed_slice = Some((U1024::zero(), U1024::zero()));
            (
                gadgetize_with_slice_zero_ccnot(
                    original,
                    p.n,
                    p.rg_freq,
                    p.slice_zero_ccnot_gates,
                    &p.masks,
                    &p.prod,
                    &mut rng,
                ),
                FunctionView::GadgetLow,
                "slice-zero-ccnot gadgetized",
            )
        } else if p.prod.single_carrier() {
            (
                gadgetize_cnot_single(original, p.n, p.rg_freq, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized (single-carrier decode)",
            )
        } else {
            (
                gadgetize_cnot(original, p.n, p.rg_freq, &p.masks, &p.prod, &mut rng),
                FunctionView::GadgetLow,
                "gadgetized",
            )
        }
    } else if p.sliced_sandwich {
        fixed_slice = Some((U1024::zero(), U1024::zero()));
        (
            sliced_sandwich_cnot(original, p.n, p.sandwich_m, p.sandwich_s, &mut rng),
            FunctionView::SandwichSecond,
            "sliced sandwich (answer on wires n..2n on the zero slice)",
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

    if p.do_gadgetize || p.do_feistalize || p.sliced_sandwich || p.gadget_path.is_some() {
        let path = p
            .gadget_path
            .map(str::to_owned)
            .unwrap_or_else(|| default_gadget_path(p.source));
        ensure_parent(&path);
        format::write_mpmct(&path, &gates, total_wires).expect("write transformed mpmct1");
        println!("[sss:cnot] transformed circuit written to {path}");
        if let Some((public_y, public_z)) = &public_slice_words {
            write_slice_metadata(&path, p.n, p.slice_zero_random_gates, public_y, public_z);
        }
    }

    let base_seed = env_u64("SSS_CNOT_SEED").unwrap_or_else(|| rng.random());
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
        print_counts(&format!("round {} after mixing", round + 1), &gates);

        let compress_params = CompressParams {
            max_iters: p.stable_compressions.max(1).saturating_mul(32),
            seed: base_seed ^ (0xc0de_0000 + round as u64),
            ..CompressParams::default()
        };
        let before_compress_len = gates.len();
        let (compressed, report) = compress(gates, total_wires, &compress_params);
        gates = compressed;
        println!(
            "[sss:cnot] round {} compression: {} -> {} gates in {} sweeps",
            round + 1,
            before_compress_len,
            gates.len(),
            report.iters,
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
    if samf_requested > 0 {
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

    ensure_parent(p.save);
    format::write_mpmct(p.save, &gates, total_wires).expect("write final CNOT circuit");
    if let Some((public_y, public_z)) = &public_slice_words {
        write_slice_metadata(p.save, p.n, p.slice_zero_random_gates, public_y, public_z);
    }
    print_counts("final", &gates);
    println!("[sss:cnot] final mpmct1 circuit written to {}", p.save);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_round_driver_writes_mpmct_and_preserves_views() {
        let source = CircuitSeq {
            gates: vec![[0, 1, 2], [2, 0, 1]],
        };
        for (
            gadgetize,
            feistalize,
            slice_ccnot,
            five_carrier,
            strong_five_carrier,
            six_carrier,
            strong_six_carrier,
            seven_carrier,
        ) in [
            (true, false, false, false, false, false, false, false),
            (false, true, false, false, false, false, false, false),
            (true, false, true, false, false, false, false, false),
            (true, false, false, true, false, false, false, false),
            (true, false, true, true, false, false, false, false),
            (true, false, false, false, true, false, false, false),
            (true, false, true, false, true, false, false, false),
            (true, false, false, false, false, true, false, false),
            (true, false, true, false, false, true, false, false),
            (true, false, false, false, false, false, true, false),
            (true, false, true, false, false, false, true, false),
            (true, false, false, false, false, false, false, true),
            (true, false, true, false, false, false, false, true),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "local_mixing_cnot_driver_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                std::process::id(),
                gadgetize as u8,
                feistalize as u8,
                slice_ccnot as u8,
                five_carrier as u8,
                strong_five_carrier as u8,
                six_carrier as u8,
                strong_six_carrier as u8,
                seven_carrier as u8
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let output = dir.join("out.txt");
            let gadget = dir.join("gadget.txt");
            let prod = if seven_carrier {
                ProdConfig::production_seven_carrier()
            } else if six_carrier || strong_six_carrier {
                ProdConfig::production_six_carrier()
            } else if five_carrier || strong_five_carrier {
                ProdConfig::production_five_carrier()
            } else {
                ProdConfig::off()
            };
            let expected_wires = if feistalize {
                9
            } else if seven_carrier {
                7 * 3 + prod.band_size(3)
            } else if six_carrier || strong_six_carrier {
                6 * 3 + prod.band_size(3)
            } else if five_carrier || strong_five_carrier {
                5 * 3 + prod.band_size(3)
            } else {
                6
            };
            let params = CnotSssParams {
                rounds: 0,
                n: 3,
                m: 1,
                x: 2,
                save: output.to_str().unwrap(),
                source: "source.txt",
                do_gadgetize: gadgetize,
                five_carrier,
                strong_five_carrier,
                six_carrier,
                strong_six_carrier,
                seven_carrier,
                do_feistalize: feistalize,
                slice_zero: false,
                slice_zero_random: false,
                slice_zero_random_gates: 96,
                slice_zero_hardcoded: false,
                slice_zero_hardcoded_rounds: 1,
                slice_zero_ccnot: slice_ccnot,
                slice_zero_ccnot_gates: if seven_carrier {
                    72
                } else if six_carrier || strong_six_carrier {
                    63
                } else if five_carrier || strong_five_carrier {
                    54
                } else {
                    18
                },
                sliced_sandwich: false,
                sandwich_m: 12,
                sandwich_s: 6,
                gadget_path: Some(gadget.to_str().unwrap()),
                full_shuffle: false,
                full_shuffle_early: false,
                shooting_times: 1,
                collision_rounds: 1,
                stable_compressions: 1,
                expansion_game: false,
                equality_check: true,
                rg_freq: 2,
                masks: MaskConfig::off(),
                prod,
            };
            main_shuffle_shoot_shuffle_cnot(&source, &params);
            let (written, wires) = format::read_mpmct(output.to_str().unwrap()).unwrap();
            assert_eq!(wires, expected_wires);
            assert!(!written.is_empty());
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}
