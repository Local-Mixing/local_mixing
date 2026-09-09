//! Export obfuscated chunk supports + plaintext/obfuscated circuit text.

use clap::Parser;
use local_mixing::sandwich::{
    format_plaintext_circuit, sample_and_obfuscate, write_obfuscated_circuit, SandwichParams,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(about = "Export sandwich chunk wire supports and circuit text")]
struct Args {
    #[arg(short = 'n', long, default_value_t = 64)]
    wires: usize,
    #[arg(short = 'm', long, default_value_t = 1024)]
    gates: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(short = 'o', long, default_value = "sandwich_viz.csv")]
    output: String,
    #[arg(long, default_value = "plain_circuit.txt")]
    circuit_out: String,
    #[arg(long, default_value = "obfuscated_circuit.txt")]
    obf_out: String,
    #[arg(long, default_value = "sandwich_timeline.png")]
    plot_out: String,
    #[arg(long, default_value = "scripts/plot_sandwich_timeline.py")]
    plot_script: String,
    #[arg(long, help = "Skip matplotlib timeline plot")]
    no_plot: bool,
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let (plain, obf) = sample_and_obfuscate(args.wires, args.gates, &mut rng);
    let params = SandwichParams::for_n(args.wires);

    std::fs::write(
        &args.circuit_out,
        format_plaintext_circuit(&plain, args.wires),
    )
    .expect("write plaintext circuit");

    let mut obf_file = std::fs::File::create(&args.obf_out).expect("create obfuscated circuit");
    write_obfuscated_circuit(
        &obf,
        params.init_gates,
        plain.gates.len(),
        &mut obf_file,
    )
    .expect("write obfuscated circuit");

    let mut f = std::fs::File::create(&args.output).expect("create csv");
    writeln!(f, "data_wires,{}", obf.data_wires).unwrap();
    writeln!(f, "total_wires,{}", obf.total_wires).unwrap();
    writeln!(f, "plain_gates,{}", plain.gates.len()).unwrap();
    writeln!(f, "init_gates,{}", params.init_gates).unwrap();
    writeln!(f, "n_chunks,{}", obf.chunks.len()).unwrap();
    writeln!(f).unwrap();

    for (i, chunk) in obf.chunks.iter().enumerate() {
        let ws: Vec<String> = chunk.wires.iter().map(|w| w.to_string()).collect();
        writeln!(f, "{i},{}", ws.join(" ")).unwrap();
    }

    let epilogue = obf
        .chunks
        .len()
        .saturating_sub(params.init_gates + plain.gates.len());
    eprintln!("wrote {}", args.circuit_out);
    eprintln!("wrote {}", args.obf_out);
    eprintln!(
        "wrote {} ({} chunks: {} init + {} gate + {} epilogue)",
        args.output,
        obf.chunks.len(),
        params.init_gates,
        plain.gates.len(),
        epilogue
    );

    if !args.no_plot {
        let plot_script = std::path::Path::new(&args.plot_script);
        if !plot_script.exists() {
            eprintln!(
                "warning: plot script {} not found; skipping timeline plot",
                plot_script.display()
            );
        } else {
            let python = if std::path::Path::new(".venv/bin/python").exists() {
                ".venv/bin/python"
            } else {
                "python3"
            };
            let status = std::process::Command::new(python)
                .arg(plot_script)
                .arg(&args.output)
                .arg("-o")
                .arg(&args.plot_out)
                .env("MPLBACKEND", "Agg")
                .env(
                    "MPLCONFIGDIR",
                    std::env::var("MPLCONFIGDIR").unwrap_or_else(|_| ".matplotlib".into()),
                )
                .status()
                .expect("run plot script");
            if status.success() {
                eprintln!("wrote {}", args.plot_out);
            } else {
                eprintln!(
                    "warning: plot script exited with status {}",
                    status.code().unwrap_or(-1)
                );
            }
        }
    }
}
