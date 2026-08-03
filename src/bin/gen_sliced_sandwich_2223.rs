//! Build the sliced-sandwich experiment with the sole production nonlinear
//! construction: [2,2,2,3], one carrier, Gray folding, and a 1:1 band.

use std::path::Path;

use clap::Parser;
use local_mixing::{
    circuit::circuit::U1024,
    postmix::{format::write_mpmct, xgate::eval_u1024},
    random::random_data::random_circuit,
    replace::gadgets::{
        gadgetize_xgates_with_slice_zero_ccnot, sandwich_default_s, sliced_sandwich_cnot,
    },
};
use rand::{RngCore, SeedableRng, rngs::StdRng};

#[derive(Debug, Parser)]
#[command(name = "gen_sliced_sandwich_2223")]
struct Args {
    #[arg(long)]
    output: String,
    #[arg(long, default_value_t = 128)]
    n: usize,
    #[arg(long, default_value_t = 3000)]
    c_gates: usize,
    #[arg(long, default_value_t = 3000)]
    d_gates: usize,
    /// Zero selects round(n log2 n).
    #[arg(long, default_value_t = 0)]
    slice_block_gates: usize,
    #[arg(long, default_value_t = 1)]
    rg_draws: usize,
    /// Zero selects 10 times the sandwich width.
    #[arg(long, default_value_t = 0)]
    outer_slice_gates: usize,
    #[arg(long, default_value_t = 1)]
    source_seed: u64,
    #[arg(long, default_value_t = 1)]
    sandwich_seed: u64,
    #[arg(long, default_value_t = 1)]
    gadget_seed: u64,
    #[arg(long, default_value_t = 512)]
    verify_samples: usize,
}

fn bit_mask(bits: usize) -> U1024 {
    if bits >= 1024 {
        U1024::MAX
    } else {
        (U1024::one() << bits) - U1024::one()
    }
}

fn main() {
    let args = Args::parse();
    assert!(args.n >= 3, "--n must be at least 3");
    assert!(args.rg_draws > 0, "--rg-draws must be nonzero");
    let sandwich_n = 2 * args.n;
    let slice_block_gates = if args.slice_block_gates == 0 {
        sandwich_default_s(args.n)
    } else {
        args.slice_block_gates
    };
    let outer_slice_gates = if args.outer_slice_gates == 0 {
        10 * sandwich_n
    } else {
        args.outer_slice_gates
    };

    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output directory");
        }
    }

    fastrand::seed(args.source_seed);
    let source = random_circuit(args.n, args.c_gates);
    assert_eq!(source.gates.len(), args.c_gates);
    let source_path = format!("{}.source_c.g57", args.output);
    std::fs::write(&source_path, source.repr()).expect("write source C");

    let mut sandwich_rng = StdRng::seed_from_u64(args.sandwich_seed ^ 0x5150_1ced);
    let sandwich = sliced_sandwich_cnot(
        &source,
        args.n,
        args.d_gates,
        slice_block_gates,
        &mut sandwich_rng,
    );
    assert_eq!(sandwich.num_wires, sandwich_n);
    let sandwich_path = format!("{}.sandwich.mpmct1", args.output);
    write_mpmct(&sandwich_path, &sandwich.gates, sandwich.num_wires)
        .expect("write sliced sandwich");

    let mut gadget_rng = StdRng::seed_from_u64(args.gadget_seed ^ 0x6ad6_e75e);
    let gadget = gadgetize_xgates_with_slice_zero_ccnot(
        &sandwich.gates,
        sandwich_n,
        args.rg_draws,
        outer_slice_gates,
        &mut gadget_rng,
    );
    let expected_band = sandwich_n;
    assert_eq!(gadget.num_wires, sandwich_n + expected_band);
    write_mpmct(&args.output, &gadget.gates, gadget.num_wires).expect("write gadgetized sandwich");

    assert!(gadget.num_wires <= 1024);
    let low_sandwich = bit_mask(sandwich_n);
    let mut verify_rng = StdRng::seed_from_u64(args.source_seed ^ 0x0a11_ce55);
    let mut bytes = [0u8; 128];
    for sample in 0..args.verify_samples {
        verify_rng.fill_bytes(&mut bytes);
        // The production preblock's fixed slice is exactly its product band.
        let input = U1024::from_little_endian(&bytes) & low_sandwich;
        let got = eval_u1024(&gadget.gates, input) & low_sandwich;
        let expected = eval_u1024(&sandwich.gates, input) & low_sandwich;
        assert_eq!(
            got, expected,
            "outer gadget slice mismatch at sample {sample}"
        );
    }

    let metadata = format!(
        "mode=sliced_sandwich_product_share\n\
         nonlinear_config_version=2223-gray-v1\n\
         nonlinear_plan=2,2,2,3\n\
         nonlinear_fold=gray\n\
         nonlinear_single_carrier=true\n\
         n={}\n\
         source_gates={}\n\
         random_d_gates={}\n\
         sandwich_slice_block_gates={}\n\
         sandwich_gates={}\n\
         sandwich_wires={}\n\
         outer_slice_gates={}\n\
         gadget_gates={}\n\
         gadget_wires={}\n\
         carrier_wires=0..{sandwich_n}\n\
         product_band_wires={sandwich_n}..{}\n\
         source_input_wires=0..{}\n\
         sandwich_zero_slice_wires={}..{}\n\
         outer_zero_slice_wires={sandwich_n}..{}\n\
         answer_wires={}..{}\n\
         source_seed={}\n\
         sandwich_seed={}\n\
         gadget_seed={}\n\
         contract=G(x,zeros)[n..2n] equals C(x)\n",
        args.n,
        source.gates.len(),
        args.d_gates,
        slice_block_gates,
        sandwich.gates.len(),
        sandwich.num_wires,
        outer_slice_gates,
        gadget.gates.len(),
        gadget.num_wires,
        gadget.num_wires,
        args.n,
        args.n,
        2 * args.n,
        gadget.num_wires,
        args.n,
        2 * args.n,
        args.source_seed,
        args.sandwich_seed,
        args.gadget_seed,
    );
    let metadata_path = format!("{}.slice_sandwich", args.output);
    std::fs::write(&metadata_path, metadata).expect("write experiment metadata");

    println!(
        "[sandwich-2223-gray] n={} C={} D={} S={} sandwich={}g/{}w outer_slice={} band={} gadget={}g/{}w verify_samples={} output={} metadata={}",
        args.n,
        source.gates.len(),
        args.d_gates,
        slice_block_gates,
        sandwich.gates.len(),
        sandwich.num_wires,
        outer_slice_gates,
        expected_band,
        gadget.gates.len(),
        gadget.num_wires,
        args.verify_samples,
        args.output,
        metadata_path,
    );
}
