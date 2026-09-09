//! Sample a random r57 circuit, sandwich-obfuscate it, and print chunk permutations.

use clap::Parser;
use local_mixing::sandwich::{
    check_correctness_random, check_correctness_random_dirty_ancilla, sample_and_obfuscate,
    ChunkBody, ChunkedCircuit,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[command(version, about = "Sandwich-obfuscate a random r57 circuit into S_8 / correction chunks")]
struct Args {
    /// Number of data wires
    #[arg(short = 'n', long, default_value_t = 32)]
    wires: usize,

    /// Number of r57 gates
    #[arg(short = 'm', long, default_value_t = 256)]
    gates: usize,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Print chunks in cycle / reset notation (can be large)
    #[arg(long, default_value_t = false)]
    print_chunks: bool,

    /// Max chunks to print when --print-chunks is set (0 = all)
    #[arg(long, default_value_t = 0)]
    max_print: usize,

    /// Probabilistic correctness trials (0 = skip)
    #[arg(long, default_value_t = 100)]
    check: usize,
}

fn main() {
    let args = Args::parse();
    assert!(args.wires >= 3, "need at least 3 wires");
    assert!(args.gates >= 1, "need at least 1 gate");

    let mut rng = StdRng::seed_from_u64(args.seed);
    let (plain, obf) = sample_and_obfuscate(args.wires, args.gates, &mut rng);

    println!(
        "plaintext: {} gates on {} data wires",
        plain.gates.len(),
        args.wires
    );
    println!(
        "obfuscated: {} chunks, {} total wires ({} data + {} ancilla)",
        obf.chunks.len(),
        obf.total_wires,
        obf.data_wires,
        obf.total_wires.saturating_sub(obf.data_wires)
    );

    let bits = obf.representation_bits();
    println!(
        "representation: {} bits ({} kibytes). blowup: {}x",
        bits,
        obf.representation_bytes() / 1024,
        bits / (plain.gates.len() * args.wires.ilog2() as usize),
    );
    summarize_chunk_sizes(&obf);

    if args.check > 0 {
        match check_correctness_random(&plain, &obf, args.check, &mut rng) {
            Ok(()) => println!("correctness check (anc=0): {} random inputs OK", args.check),
            Err(e) => {
                eprintln!("correctness check FAILED: {e}");
                std::process::exit(1);
            }
        }
        match check_correctness_random_dirty_ancilla(&plain, &obf, args.check, &mut rng) {
            Ok(()) => {
                println!("correctness check (dirty ancilla): {} random inputs OK", args.check)
            }
            Err(e) => {
                eprintln!("dirty-ancilla correctness FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if args.print_chunks {
        let limit = if args.max_print == 0 {
            obf.chunks.len()
        } else {
            args.max_print.min(obf.chunks.len())
        };
        for (i, chunk) in obf.chunks.iter().take(limit).enumerate() {
            let kind = match chunk.body {
                ChunkBody::Perm(_) => format!("perm k={}", chunk.k()),
                ChunkBody::Zero => "reset".to_string(),
            };
            println!("\n# chunk {i}  {kind}");
            println!("{}", chunk.cycle_notation());
        }
        if limit < obf.chunks.len() {
            println!(
                "\n... ({} more chunks not printed)",
                obf.chunks.len() - limit
            );
        }
    }
}

fn summarize_chunk_sizes(obf: &ChunkedCircuit) {
    let mut by_k = std::collections::BTreeMap::<usize, usize>::new();
    let mut resets = 0usize;
    for c in &obf.chunks {
        match c.body {
            ChunkBody::Zero => resets += 1,
            ChunkBody::Perm(_) => *by_k.entry(c.k()).or_default() += 1,
        }
    }
    print!("chunk widths:");
    for (k, count) in by_k {
        print!("  S_{{2^{k}}}×{count}");
    }
    if resets > 0 {
        print!("  RESET×{resets}");
    }
    println!();
}
