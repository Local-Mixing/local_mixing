//! Build the sole production nonlinear TDP construction:
//! [2,2,2,3], single carrier, Gray fold, and a 1:1 product band.

use std::path::Path;

use clap::Parser;
use local_mixing::{
    circuit::CircuitSeq,
    postmix::format,
    replace::gadgets::{packed_bit, tdp4n_nonlinear_with_slice_zero_random_cnot},
};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Debug, Parser)]
#[command(name = "gen_tdp4n_nonlinear_2223")]
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
    assert!(
        original
            .gates
            .iter()
            .flatten()
            .all(|&wire| (wire as usize) < args.n),
        "input circuit uses a wire outside 0..n"
    );
    let source_repr_xxh3_128 = format!(
        "{:032x}",
        xxhash_rust::xxh3::xxh3_128(original.repr().as_bytes())
    );

    let mut rng = StdRng::seed_from_u64(args.seed);
    let constructed = tdp4n_nonlinear_with_slice_zero_random_cnot(
        &original,
        args.n,
        args.rg_draws,
        args.slice_gates,
        &mut rng,
    );

    // Two n-wire carrier blocks, two reserved n-wire helper blocks, then a
    // 2n-wire product band: 6n total.
    let band_width = 2 * args.n;
    let four_n = 4 * args.n;
    let expected_wires = four_n + band_width;
    assert_eq!(
        constructed.circuit.num_wires, expected_wires,
        "unexpected [2,2,2,3] Gray TDP wire count"
    );

    format::write_mpmct(
        &args.output,
        &constructed.circuit.gates,
        constructed.circuit.num_wires,
    )
    .expect("write constructed circuit");

    let y = packed_words_to_hex(&constructed.public_y, args.n);
    let z = packed_words_to_hex(&constructed.public_z, args.n);
    let metadata = format!(
        "mode=slice_zero_random\n\
         representation=mpmct1\n\
         layout=tdp4n_single_carrier_2223_gray_padded\n\
         nonlinear_config_version=2223-gray-v1\n\
         nonlinear_plan=2,2,2,3\n\
         nonlinear_fold=gray\n\
         nonlinear_single_carrier=true\n\
         nonlinear_band_size={band_width}\n\
         n={}\n\
         total_wires={}\n\
         gates={}\n\
         slice_preblock_gates={}\n\
         construction_seed={}\n\
         rg_draws={}\n\
         source_gates={}\n\
         source_repr_xxh3_128={}\n\
         y_hex={y}\n\
         z_hex={z}\n\
         x_carrier_wires=0..{}\n\
         y_carrier_wires={}..{}\n\
         reserved_aux_wires={}..{four_n}\n\
         product_band_wires={four_n}..{}\n\
         middle_output_wires={}..{}\n\
         fixed_input_blocks=y,z\n\
         bit_order=bit i is wire n+i for y and wire 2n+i for z\n",
        args.n,
        constructed.circuit.num_wires,
        constructed.circuit.gates.len(),
        args.slice_gates,
        args.seed,
        args.rg_draws,
        original.gates.len(),
        source_repr_xxh3_128,
        args.n,
        args.n,
        2 * args.n,
        2 * args.n,
        constructed.circuit.num_wires,
        args.n,
        2 * args.n,
    );
    let metadata_path = format!("{}.slice_zero_random", args.output);
    std::fs::write(&metadata_path, metadata).expect("write slice metadata");

    println!(
        "[tdp4n-2223-gray] input_gates={} output_gates={} wires={} carriers={} reserved={} band={} seed={} Y={} Z={} output={} metadata={}",
        original.gates.len(),
        constructed.circuit.gates.len(),
        constructed.circuit.num_wires,
        2 * args.n,
        2 * args.n,
        band_width,
        args.seed,
        y,
        z,
        args.output,
        metadata_path,
    );
}
