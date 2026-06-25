use std::fs::File;
use std::io::Write;

use clap::{Parser, ValueEnum};
use cryptography::{Aes128, BlockCipher};
use entropy::{diehard, dieharder};
use entropy::{nist, rng::Rng};
use fastrand::shuffle;
use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::random_circuit;
use primitive_types::U256 as u256;
use rand::RngCore;
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
    Aes,
}

struct AesPrg {
    cipher: Aes128,
    counter: u128,
    operation_mode: OperationMode,
    word_index: u8,
    current_state: u128,
}

impl AesPrg {
    fn new(operation_mode: OperationMode) -> Self {
        let key = fastrand::u128(..).to_be_bytes();
        Self {
            cipher: Aes128::new(&key),
            counter: fastrand::u128(..),
            operation_mode,
            word_index: 4,
            current_state: 0,
        }
    }

    fn next_state(&mut self) -> u128 {
        let mut block = self.counter.to_be_bytes();
        self.cipher.encrypt(&mut block);
        let output = u128::from_be_bytes(block);
        self.counter = match self.operation_mode {
            OperationMode::Ctr => self.counter + 1,
            OperationMode::Ofb => output,
        };
        output
    }
}

impl Rng for AesPrg {
    fn next_u32(&mut self) -> u32 {
        if self.word_index >= 4 {
            self.current_state = self.next_state();
            self.word_index = 0;
        }

        let word = (self.current_state >> (32 * self.word_index)) as u32;
        self.word_index += 1;
        word
    }
}

struct CircuitPrg {
    circuit: CircuitSeq,
    counter: u256,
    operation_mode: OperationMode,
    max_word_idx: u8,
    word_index: u8,
    current_state: u256,
}

impl CircuitPrg {
    fn new(circuit: CircuitSeq, operation_mode: OperationMode) -> Self {
        let n = &circuit.max_wire() + 1;
        let mut p = Self {
            circuit,
            // Random IV
            counter: fastrand::u128(..).into(),
            operation_mode,
            max_word_idx: (n / 32) as u8,
            word_index: 0,
            current_state: 0.into(),
        };
        p.word_index = p.max_word_idx;
        p
    }

    fn next_state(&mut self) -> u256 {
        let output = self.circuit.evaluate_256(self.counter);
        self.counter = match self.operation_mode {
            OperationMode::Ctr => self.counter + 1,
            OperationMode::Ofb => output,
        };
        output
    }
}

impl Rng for CircuitPrg {
    fn next_u32(&mut self) -> u32 {
        if self.word_index >= self.max_word_idx {
            self.current_state = self.next_state();
            self.word_index = 0;
        }

        let word = (self.current_state >> (32 * self.word_index)).low_u32();
        self.word_index += 1;
        word
    }
}

enum BlockPrg {
    Circuit(CircuitPrg),
    Aes(AesPrg),
}

impl Rng for BlockPrg {
    fn next_u32(&mut self) -> u32 {
        match self {
            BlockPrg::Circuit(rng) => rng.next_u32(),
            BlockPrg::Aes(rng) => rng.next_u32(),
        }
    }
}

fn balanced_ckt_ord(n: usize, m: usize) -> CircuitSeq {
    let mut c = Vec::with_capacity(m);
    let mut active_counts = vec![0usize; n];

    let maxw = (n - 1) as u8;

    for _ in 0..m {
        loop {
            let min_count = active_counts.iter().copied().min().unwrap_or(0);
            let least_popular: Vec<usize> = active_counts
                .iter()
                .enumerate()
                .filter_map(|(wire, &count)| (count == min_count).then_some(wire))
                .collect();

            let gate0 = least_popular[fastrand::usize(..least_popular.len())] as u8;
            let gate1 = fastrand::u8(..=maxw);
            let gate2 = fastrand::u8(..=maxw);
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

    #[arg(short, long, default_value_t = false)]
    brute: bool,
}

const Y: usize = 0xfedcba9876543210;

fn next_u256() -> u256 {
    let mut b = [0u8; 32];
    rand::rng().fill_bytes(&mut b);
    u256::from_little_endian(&b)
}

fn random_ckt_brute(args: &Args) {
    use indicatif::ProgressBar;

    let n = args.wires;

    let c = match args.circuit_gen {
        CircuitGenerationMode::Random => random_circuit(n, args.gates),
        CircuitGenerationMode::Balanced => balanced_ckt_uniform(n, args.gates),
        CircuitGenerationMode::Aes => panic!("AES not supported."),
    };

    let num_trials: usize = 1 << args.reps;

    let progress = ProgressBar::new(num_trials as u64);
    progress.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let y256 = u256::from(Y);
    let yy: u256 = if n > 128 { y256 << 64 | y256 } else { y256 };

    let r = (0..num_trials)
        .par_bridge()
        .map(|_| {
            let v = next_u256();
            let vv = if n > 128 { v } else { u256::from(v.low_u128()) };
            progress.inc(1);
            let z = c.evaluate_256((vv << (n / 2)) | yy);
            if n > 128 { z } else { u256::from(z.low_u128()) }
        })
        .min()
        .unwrap();

    let mut cc = c.clone();
    cc.gates.reverse();
    let pre = cc.evaluate_256(r);

    let disp = if n > 128 {
        format!(
            "Min n={},m={} t={} {:?}: {:064x} => {:064x} ({} zeros)",
            args.wires,
            args.gates,
            num_trials,
            args.circuit_gen,
            pre,
            r,
            r.leading_zeros()
        )
    } else {
        format!(
            "Min n={},m={} t={} {:?}: {:032x} => {:032x} ({} zeros)",
            args.wires,
            args.gates,
            num_trials,
            args.circuit_gen,
            pre,
            r,
            r.leading_zeros() - 128
        )
    };

    let mut file = File::create(format!(
        "ckt-out/n{}m{}-{:?}.txt",
        args.wires, args.gates, args.circuit_gen
    ))
    .expect("could not open file!");

    (|| -> std::io::Result<()> {
        write!(file, "# {}\n\n\n", disp)?;
        write!(file, "{}", c.repr())?;
        write!(file, "\n")?;
        Ok(())
    })()
    .expect("failed to write!");

    println!("{}", disp);
}

fn main() {
    let args = Args::parse();

    let wires = args.wires;
    let gates = args.gates;
    let length = args.length;
    let reps = args.reps;
    let operation_mode = args.operation_mode;
    let gen_mode = args.circuit_gen;

    if args.brute {
        random_ckt_brute(&args);
        return;
    }

    if !args.file {
        println!("Random reversible circuit: {wires} wires, {gates} gates");
        println!(
            "Feeding seq len {length} into NIST+DH+DHR battery ({operation_mode:?} mode, {gen_mode:?} circuit)"
        );
    }

    let failed_counts: Vec<usize> = (0..reps)
        .into_par_iter()
        .map(|_| {
            let mut rng = match gen_mode {
                CircuitGenerationMode::Random => {
                    let circuit = random_circuit(wires, gates);
                    BlockPrg::Circuit(CircuitPrg::new(circuit, operation_mode))
                }
                CircuitGenerationMode::Balanced => {
                    let circuit = balanced_ckt_ord(wires, gates);
                    BlockPrg::Circuit(CircuitPrg::new(circuit, operation_mode))
                }
                CircuitGenerationMode::Aes => BlockPrg::Aes(AesPrg::new(operation_mode)),
            };
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
            "{}: {}",
            gates,
            failed_counts
                .into_iter()
                .map(|count| count.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
