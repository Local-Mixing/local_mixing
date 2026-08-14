//! Fragment wide MPMCT gates after layout/shuffling, keeping each restored
//! dirty-helper macro contiguous for the frozen short-window rewriter.

use clap::Parser;
use local_mixing::postmix::format::{read_mpmct, write_mpmct};
use local_mixing::postmix::fragment::{FragmentStyle, fragment_wide_post_shuffle};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fragment_wide")]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: String,
    /// exact | native-deep
    #[arg(long, default_value = "native-deep")]
    style: String,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let style = FragmentStyle::parse(&args.style)
        .unwrap_or_else(|| panic!("--style must be exact or native-deep, got {:?}", args.style));
    let (mut gates, num_wires) = read_mpmct(&args.input).expect("read --input mpmct1");
    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xF12A_6E17_2026_0813);
    let stats = fragment_wide_post_shuffle(&mut gates, num_wires, style, &mut rng)
        .unwrap_or_else(|error| panic!("post-shuffle fragmentation failed: {error}"));
    write_mpmct(&args.output, &gates, num_wires).expect("write --output mpmct1");
    println!(
        "[fragment-wide] style={style:?} gates {} -> {} ({} wide macros), max controls {} -> {}, native emissions={}, exact-rung emissions={}",
        stats.input_gates,
        stats.output_gates,
        stats.fragmented_gates,
        stats.max_controls_before,
        stats.max_controls_after,
        stats.native_emissions,
        stats.exact_rung_emissions,
    );
}
