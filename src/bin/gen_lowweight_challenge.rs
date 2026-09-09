//! Generate a sealed sandwich-obfuscation challenge (no plaintext in the output).

use clap::Parser;
use local_mixing::sandwich::sample_and_obfuscate;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Generate a sealed low-weight search challenge from sandwich obfuscation")]
struct Args {
    #[arg(short = 'n', long, default_value_t = 64)]
    wires: usize,

    #[arg(short = 'm', long, default_value_t = 512)]
    gates: usize,

    /// Leading data wires fixed to 0 in the attack
    #[arg(long, default_value_t = 32)]
    prefix_zeros: usize,

    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,

    #[arg(long, default_value = "challenges/lowweight_obf.txt")]
    out: PathBuf,

    /// Optional sealed plaintext (for verifier only — do not use while attacking)
    #[arg(long, default_value = "challenges/lowweight_plain.SEALED.txt")]
    plain_out: PathBuf,
}

fn main() {
    let args = Args::parse();
    assert!(args.prefix_zeros < args.wires);
    assert!(args.wires - args.prefix_zeros <= 64);

    let mut rng = StdRng::seed_from_u64(args.seed);
    let (plain, obf) = sample_and_obfuscate(args.wires, args.gates, &mut rng);

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    let mut header = String::new();
    header.push_str("# LOW-WEIGHT SEARCH CHALLENGE (sealed obfuscated circuit)\n");
    header.push_str("# Goal: minimize Hamming weight (# of 1s) of y = C(0^{prefix} || x)\n");
    header.push_str(&format!("# data_wires={} prefix_zeros={} free_bits={}\n", args.wires, args.prefix_zeros, args.wires - args.prefix_zeros));
    header.push_str(&format!("# seed={} gates={} (plaintext NOT included below)\n", args.seed, args.gates));
    header.push_str(&format!("challenge_prefix_zeros {}\n", args.prefix_zeros));
    header.push_str(&format!("challenge_seed {}\n\n", args.seed));

    let body = obf.to_readable_string();
    fs::write(&args.out, format!("{header}{body}")).unwrap();

    // Sealed plaintext — attacker must not read this for the search.
    let mut plain_s = String::new();
    plain_s.push_str("# SEALED plaintext r57 circuit — do not use during attack\n");
    plain_s.push_str(&format!("n {}\n", args.wires));
    plain_s.push_str(&format!("m {}\n", plain.gates.len()));
    for g in &plain.gates {
        plain_s.push_str(&format!("{},{},{}\n", g[0], g[1], g[2]));
    }
    fs::write(&args.plain_out, plain_s).unwrap();

    println!("wrote obfuscated challenge → {}", args.out.display());
    println!(
        "  {} chunks, {} total wires, {} bytes",
        obf.chunks.len(),
        obf.total_wires,
        fs::metadata(&args.out).unwrap().len()
    );
    println!(
        "wrote SEALED plaintext → {} (do not open for attack)",
        args.plain_out.display()
    );
}
