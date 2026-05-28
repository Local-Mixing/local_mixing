use clap::{Parser, ValueEnum};
use entropy::{diehard, dieharder};
use entropy::{nist, rng::Rng};
use fastrand::shuffle;
use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::random_circuit;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OperationMode {
    Ctr,
    Ofb,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CircuitGenerationMode {
    Balanced,
    Random,
}

struct CircuitPrg {
    circuit: CircuitSeq,
    counter: u128,
    operation_mode: OperationMode,
}

impl CircuitPrg {
    fn new(circuit: CircuitSeq, operation_mode: OperationMode) -> Self {
        Self {
            circuit,
            counter: 0,
            operation_mode,
        }
    }

    fn next_state(&mut self) -> u128 {
        let input = self.counter;
        let output = self.circuit.evaluate_128(input);
        self.counter = match self.operation_mode {
            OperationMode::Ctr => input + 1,
            OperationMode::Ofb => output,
        };
        output
    }
}

impl Rng for CircuitPrg {
    fn next_u32(&mut self) -> u32 {
        let s = self.next_state();
        ((s >> (3 * 32)) ^ (s >> (2 * 32)) ^ (s >> (1 * 32)) ^ s) as u32
    }

    fn next_u64(&mut self) -> u64 {
        let s = self.next_state();
        ((s >> 64) ^ s) as u64
    }
}

fn balanced_ckt_ord(n: usize, m: usize) -> CircuitSeq {
    let mut c = Vec::with_capacity(m);
    let mut active_counts = vec![0usize; n];

    let n8 = n as u8;

    for _ in 0..m {
        loop {
            let min_count = active_counts.iter().copied().min().unwrap_or(0);
            let least_popular: Vec<usize> = active_counts
                .iter()
                .enumerate()
                .filter_map(|(wire, &count)| (count == min_count).then_some(wire))
                .collect();

            let gate0 = least_popular[fastrand::usize(..least_popular.len())] as u8;
            let gate1 = fastrand::u8(..n8);
            let gate2 = fastrand::u8(..n8);
            let gate = [gate0, gate1, gate2];

            // No trivial identites, no duplicated pins
            if c.last() == Some(&gate)
                || (gate[0] == gate[1] || gate[0] == gate[2] || gate[1] == gate[2])
            {
                continue;
            } else {
                c.push(gate);
                active_counts[gate0 as usize] += 1;
                break;
            }
        }
    }
    CircuitSeq { gates: c }
}

#[allow(unused)]
fn balanced_ckt_uniform(n: usize, m: usize) -> CircuitSeq {
    let mut c = Vec::with_capacity(m);

    let n8 = n as u8;

    for i in 0..m {
        loop {
            let gate = [(i % n) as u8, fastrand::u8(..n8), fastrand::u8(..n8)];

            if c.last() == Some(&gate)
                || (gate[0] == gate[1] || gate[0] == gate[2] || gate[1] == gate[2])
            {
                continue;
            } else {
                c.push(gate);
                break;
            }
        }
    }

    shuffle(&mut c);

    CircuitSeq { gates: c }
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', long, default_value_t = 128)]
    wires: usize,

    #[arg(short = 'm', long, default_value_t = 1024)]
    gates: usize,

    #[arg(short, long, default_value_t = 1 << 20)]
    length: usize,

    #[arg(short, long, default_value_t = 1)]
    reps: usize,

    #[arg(short, long, value_enum, default_value_t = OperationMode::Ctr)]
    operation_mode: OperationMode,

    #[arg(short, long, value_enum, default_value_t = CircuitGenerationMode::Random)]
    circuit_gen: CircuitGenerationMode,

    #[arg(short, long, default_value_t = false)]
    file: bool,
}

fn main() {
    let args = Args::parse();

    let wires = args.wires;
    let gates = args.gates;
    let length = args.length;
    let reps = args.reps;
    let operation_mode = args.operation_mode;
    let gen_mode = args.circuit_gen;

    if !args.file {
        println!("Random reversible circuit: {wires} wires, {gates} gates");
        println!(
            "Feeding seq len {length} into NIST+DH+DHR battery ({operation_mode:?} mode, {gen_mode:?} circuit)"
        );
    }

    let failed_counts: Vec<usize> = (0..reps)
        .into_par_iter()
        .map(|_| {
            let circuit = match gen_mode {
                CircuitGenerationMode::Random => random_circuit(wires, gates),
                CircuitGenerationMode::Balanced => balanced_ckt_ord(wires, gates),
            };

            let mut rng = CircuitPrg::new(circuit, operation_mode);
            let results = [
                nist::run_all(&mut rng, length),
                diehard::run_all(&mut rng, length, true),
                dieharder::run_all(&mut rng, length, true),
            ]
            .concat();

            // for result in &results {
            //     println!("{result}");
            // }

            let passed = results.iter().filter(|result| result.passed()).count();
            let skipped = results.iter().filter(|result| result.skipped()).count();
            let total = results.len();
            let failed = total - passed - skipped;

            if !args.file {
                println!(
                    "{} / {} passed, {} skipped  => {:.1}%",
                    passed,
                    total,
                    skipped,
                    100.0 * (passed as f64) / ((total - skipped) as f64)
                );
            }

            failed
        })
        .collect();

    if args.file {
        println!(
            "{}: {}", gates,
            failed_counts
                .into_iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
