use clap::Parser;
use entropy::{diehard, dieharder};
use entropy::{nist, rng::Rng};
use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::random_circuit;
use rayon::prelude::*;

struct CircuitPrg {
    circuit: CircuitSeq,
    counter: u128,
}

impl CircuitPrg {
    fn new(circuit: CircuitSeq) -> Self {
        Self {
            circuit,
            counter: 0,
        }
    }

    fn next_state(&mut self) -> u128 {
        let output = self.circuit.evaluate_128(self.counter);
        self.counter += 1;
        output
    }
}

impl Rng for CircuitPrg {
    fn next_u32(&mut self) -> u32 {
        self.next_state() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_state() as u64
    }
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

    #[arg(short = 'r', long, default_value_t = 1)]
    reps: usize,
}

fn main() {
    let args = Args::parse();

    let wires = args.wires;
    let gates = args.gates;
    let length = args.length;
    let reps = args.reps;

    println!("Random reversible circuit: {wires} wires, {gates} gates");
    println!("Feeding seq len {length} into NIST+DH+DHR battery");

    (0..reps).into_par_iter().for_each(|_| {
        let circuit = random_circuit(wires, gates);
        let mut rng = CircuitPrg::new(circuit);
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

        println!(
            "{} / {} passed, {} skipped  => {:.1}%",
            passed,
            total,
            skipped,
            100.0 * (passed as f64) / ((total - skipped) as f64)
        );
    });
}
