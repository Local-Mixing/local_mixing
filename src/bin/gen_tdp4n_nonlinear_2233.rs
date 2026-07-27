//! Focused `[2,2,3,3]` nonlinear TDP4n constructor.
//!
//! This binary exists for controlled research comparisons. It deliberately
//! does not change the production `[3,3]` preset used by `sss`.

use std::path::Path;

use clap::Parser;
use local_mixing::{
    circuit::CircuitSeq,
    postmix::format,
    replace::gadgets::{
        ProdConfig, packed_bit, tdp4n_nonlinear_with_slice_zero_random_cnot_with_config,
    },
};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Debug, Parser)]
#[command(name = "gen_tdp4n_nonlinear_2233")]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    #[arg(long, default_value_t = 128)]
    n: usize,
    #[arg(long, default_value_t = 1)]
    rg_draws: usize,
    #[arg(long, default_value_t = 4096)]
    slice_gates: usize,
    #[arg(long, default_value_t = 2026072501)]
    seed: u64,
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

fn main() {
    let args = Args::parse();
    assert!(args.n >= 3, "--n must be at least 3");
    assert!(args.rg_draws > 0, "--rg-draws must be nonzero");

    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }

    let source = std::fs::read_to_string(&args.input).expect("read G57 input");
    let original = CircuitSeq::from_string(&source);
    let source_repr_xxh3_128 = format!(
        "{:032x}",
        xxhash_rust::xxh3::xxh3_128(original.repr().as_bytes())
    );
    assert!(
        original
            .gates
            .iter()
            .flatten()
            .all(|&wire| (wire as usize) < args.n),
        "input circuit uses a wire outside 0..n"
    );

    let config = ProdConfig::research_2233();
    assert_eq!(
        (config.k, config.deg, config.k_hi, config.deg_hi),
        (2, 2, 2, 3)
    );
    let mut rng = StdRng::seed_from_u64(args.seed);
    let constructed = tdp4n_nonlinear_with_slice_zero_random_cnot_with_config(
        &original,
        args.n,
        args.rg_draws,
        args.slice_gates,
        &config,
        &mut rng,
    );

    let expected_band = (((4 * (2 * args.n) * 4) as f64).sqrt().ceil() as usize)
        .max(6)
        .max(6);
    let expected_wires = 4 * args.n + expected_band;
    assert_eq!(
        constructed.circuit.num_wires, expected_wires,
        "unexpected [2,2,3,3] TDP wire count"
    );

    format::write_mpmct(
        &args.output,
        &constructed.circuit.gates,
        constructed.circuit.num_wires,
    )
    .expect("write constructed circuit");

    let y = packed_words_to_hex(&constructed.public_y, args.n);
    let z = packed_words_to_hex(&constructed.public_z, args.n);
    let three_n = 3 * args.n;
    let four_n = 4 * args.n;
    let metadata = format!(
        "mode=slice_zero_random\n\
         representation=mpmct1\n\
         layout=tdp4n_two_share\n\
         nonlinear_plan=2,2,3,3\n\
         nonlinear_k={}\n\
         nonlinear_deg={}\n\
         nonlinear_k_hi={}\n\
         nonlinear_deg_hi={}\n\
         nonlinear_band_config={}\n\
         nonlinear_band_size={}\n\
         nonlinear_rsrc={}\n\
         nonlinear_max_width={}\n\
         nonlinear_fill_nl={}\n\
         nonlinear_roll={}\n\
         n={}\n\
         total_wires={}\n\
         gates={}\n\
         slice_preblock_gates={}\n\
         construction_seed={}\n\
         rg_draws={}\n\
         source_gates={}\n\
         source_repr_xxh3_128={}\n\
         y_hex={}\n\
         z_hex={}\n\
         x_wires=0..{}\n\
         y_wires={}..{}\n\
         z_wires={}..{}\n\
         w_wires={}..{}\n\
         sat_helper_wires={}..{}\n\
         extra_helper_wires={}..{}\n\
         degree2_band_wires={}..{}\n\
         middle_output_wires={}..{}\n\
         fixed_input_blocks=y,z\n\
         bit_order=bit i is wire n+i for y and wire 2n+i for z\n",
        config.k,
        config.deg,
        config.k_hi,
        config.deg_hi,
        config.band,
        expected_band,
        config.rsrc,
        config.max_width,
        config.fill_nl,
        config.roll,
        args.n,
        constructed.circuit.num_wires,
        constructed.circuit.gates.len(),
        args.slice_gates,
        args.seed,
        args.rg_draws,
        original.gates.len(),
        source_repr_xxh3_128,
        y,
        z,
        args.n,
        args.n,
        2 * args.n,
        2 * args.n,
        three_n,
        three_n,
        four_n,
        three_n,
        constructed.circuit.num_wires,
        four_n,
        constructed.circuit.num_wires,
        four_n,
        constructed.circuit.num_wires,
        args.n,
        2 * args.n,
    );
    let metadata_path = format!("{}.slice_zero_random", args.output);
    std::fs::write(&metadata_path, metadata).expect("write slice metadata");

    println!(
        "[tdp4n-2233] input_gates={} output_gates={} wires={} carrier_wires={} band_wires={} seed={} Y={} Z={} output={} metadata={}",
        original.gates.len(),
        constructed.circuit.gates.len(),
        constructed.circuit.num_wires,
        four_n,
        expected_band,
        args.seed,
        y,
        z,
        args.output,
        metadata_path,
    );
}
