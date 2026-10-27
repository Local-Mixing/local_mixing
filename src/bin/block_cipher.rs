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
use rand::seq::{IteratorRandom, SliceRandom};
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
    FanoutTwo,
    CMR22,
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
    wires: usize,
    /// Low `partial_bits` of `partial` are keystream bits not yet returned.
    partial: u256,
    partial_bits: usize,
}

fn mask_bits(bits: usize) -> u256 {
    assert!(bits <= 256, "circuit width exceeds 256 wires");
    if bits == 256 {
        u256::MAX
    } else {
        (u256::one() << bits) - 1
    }
}

impl CircuitPrg {
    fn new(circuit: CircuitSeq, wires: usize, operation_mode: OperationMode) -> Self {
        assert!(wires > 0, "circuit needs at least one wire");
        Self {
            circuit,
            // Random IV in the low `wires` bits. Wider circuits keep the extra wires at 0.
            counter: u256::from(fastrand::u128(..)) & mask_bits(wires),
            operation_mode,
            wires,
            partial: 0.into(),
            partial_bits: 0,
        }
    }

    fn next_block(&mut self) -> u256 {
        let output = self.circuit.evaluate_256(self.counter) & mask_bits(self.wires);
        self.counter = match self.operation_mode {
            OperationMode::Ctr => (self.counter + 1) & mask_bits(self.wires),
            OperationMode::Ofb => output,
        };
        output
    }
}

impl Rng for CircuitPrg {
    fn next_u32(&mut self) -> u32 {
        // Pack successive n-bit blocks back-to-back. A block narrower than 32 bits
        // spans words; a wider block fills several words, and a remainder stays buffered.
        loop {
            if self.partial_bits >= 32 {
                let word = self.partial.low_u32();
                self.partial >>= 32;
                self.partial_bits -= 32;
                return word;
            }

            let mut block = self.next_block();
            let mut block_bits = self.wires;
            let need = 32 - self.partial_bits;
            if block_bits < need {
                self.partial |= block << self.partial_bits;
                self.partial_bits += block_bits;
                continue;
            }

            let fill = block & ((u256::one() << need) - 1);
            let word = (self.partial | (fill << self.partial_bits)).low_u32();
            block >>= need;
            block_bits -= need;
            self.partial = block;
            self.partial_bits = block_bits;
            return word;
        }
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

    let maxw = (n - 1) as u16;

    for _ in 0..m {
        loop {
            let min_count = active_counts.iter().copied().min().unwrap_or(0);
            let least_popular: Vec<usize> = active_counts
                .iter()
                .enumerate()
                .filter_map(|(wire, &count)| (count == min_count).then_some(wire))
                .collect();

            let gate0 = least_popular[fastrand::usize(..least_popular.len())] as u16;
            let gate1 = fastrand::u16(..=maxw);
            let gate2 = fastrand::u16(..=maxw);
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

    let n16 = n as u16;

    for i in 0..m {
        loop {
            let gate = [(i % n) as u16, fastrand::u16(..n16), fastrand::u16(..n16)];

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

/// Generate a circuit where each active pin has fanout two.
fn fanout_two_ckt(n: usize, m: usize) -> CircuitSeq {
    assert_eq!(m % n, 0);

    let all_wires = 0..m;

    // build a simple automata
    #[derive(PartialEq, Copy, Clone)]
    enum WireStates {
        Startup,
        StartupOneCtrl,
        StartupTwoCtrl,
        Active,
        OneCtrl,
        TwoCtrl,
    }

    let mut ws = vec![WireStates::Startup; n];
    let mut gates = Vec::with_capacity(m);

    let mut rng = rand::rng();

    // First round uses startup
    for _ in all_wires {
        // Next active pin chosen randomly from startup OR two-control pins
        let active = ws
            .iter()
            .enumerate()
            .filter(|&(_, &x)| x != WireStates::Active && x != WireStates::OneCtrl)
            .map(|(i, _)| i)
            .choose(&mut rng)
            .expect("no eligible active wire");

        // Next controls chosen from any state besides the two-control states
        let [c1, c2] = <[usize; 2]>::try_from(
            ws.iter()
                .enumerate()
                .filter(|&(i, &x)| {
                    i != active && x != WireStates::StartupTwoCtrl && x != WireStates::TwoCtrl
                })
                .map(|(i, _)| i)
                .choose_multiple(&mut rng, 2),
        )
        .expect("fewer than two eligible control wires");

        ws[active] = WireStates::Active;
        ws[c1] = match ws[c1] {
            WireStates::Startup => WireStates::StartupOneCtrl,
            WireStates::StartupOneCtrl => WireStates::StartupTwoCtrl,
            WireStates::Active => WireStates::OneCtrl,
            WireStates::OneCtrl => WireStates::TwoCtrl,
            _ => panic!(),
        };
        ws[c2] = match ws[c2] {
            WireStates::Startup => WireStates::StartupOneCtrl,
            WireStates::StartupOneCtrl => WireStates::StartupTwoCtrl,
            WireStates::Active => WireStates::OneCtrl,
            WireStates::OneCtrl => WireStates::TwoCtrl,
            _ => panic!(),
        };

        gates.push([active as u16, c1 as u16, c2 as u16]);
    }

    CircuitSeq { gates }
}

fn cmr_ckt(n: usize, m: usize) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);

    // wires = 3^q for some integer q
    let levels = ((n as f64).log(3.0)).round() as u32;
    assert_eq!(3u32.pow(levels), n as u32);

    const DENSITY: u32 = 5;
    let reps: u32 = levels;

    let mut rng = rand::rng();

    for _ in 0..reps {
        for l in 0..levels {
            for i in 0..3u32.pow(levels - l - 1) {
                for j in 0..3u32.pow(l) {
                    for _ in 0..DENSITY {
                        let mut g = <[u16; 3]>::try_from(
                            (0..=2)
                                .map(|v| ((3 * i + v) * 3u32.pow(l) + j) as u16)
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                        g.shuffle(&mut rng);
                        gates.push(g);
                        println!(" gate {:?}", g);
                    }
                }
            }
        }
    }

    CircuitSeq { gates }
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

/// Repeat the 64-bit pattern `Y` across the low `bits`.
fn pattern_bits(bits: usize) -> u256 {
    let y = u256::from(Y);
    let mut out = u256::zero();
    let mut at = 0;
    while at < bits {
        let take = (bits - at).min(64);
        let chunk = if take == 64 {
            y
        } else {
            y & ((u256::one() << take) - 1)
        };
        out |= chunk << at;
        at += take;
    }
    out
}

fn random_ckt_brute(args: &Args) {
    use indicatif::ProgressBar;

    let n = args.wires;

    let c = match args.circuit_gen {
        CircuitGenerationMode::Random => random_circuit(n, args.gates),
        CircuitGenerationMode::Balanced => balanced_ckt_uniform(n, args.gates),
        CircuitGenerationMode::FanoutTwo => fanout_two_ckt(n, args.gates),
        CircuitGenerationMode::CMR22 => cmr_ckt(n, args.gates),
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

    assert!(n > 0 && n <= 256, "brute search supports 1..=256 wires");
    let low_bits = n / 2;
    let high_bits = n - low_bits;
    let yy = pattern_bits(low_bits);
    let out_mask = mask_bits(n);

    let r = (0..num_trials)
        .par_bridge()
        .map(|_| {
            let vv = next_u256() & mask_bits(high_bits);
            progress.inc(1);
            c.evaluate_256((vv << low_bits) | yy) & out_mask
        })
        .min()
        .unwrap();

    let mut cc = c.clone();
    cc.gates.reverse();
    let pre = cc.evaluate_256(r) & out_mask;
    let zeros = r.leading_zeros() - (256 - n) as u32;
    let hex_width = n.div_ceil(4);

    let disp = format!(
        "Min n={},m={} t={} {:?}: {:0hex_width$x} => {:0hex_width$x} ({} zeros)",
        args.wires,
        args.gates,
        num_trials,
        args.circuit_gen,
        pre,
        r,
        zeros,
        hex_width = hex_width
    );

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
                    BlockPrg::Circuit(CircuitPrg::new(circuit, wires, operation_mode))
                }
                CircuitGenerationMode::Balanced => {
                    let circuit = balanced_ckt_ord(wires, gates);
                    BlockPrg::Circuit(CircuitPrg::new(circuit, wires, operation_mode))
                }
                CircuitGenerationMode::FanoutTwo => {
                    let circuit = fanout_two_ckt(wires, gates);
                    println!("{}", circuit.repr());
                    BlockPrg::Circuit(CircuitPrg::new(circuit, wires, operation_mode))
                }
                CircuitGenerationMode::CMR22 => {
                    let circuit = cmr_ckt(wires, gates);
                    println!("CMR len: {}", circuit.gates.len());
                    BlockPrg::Circuit(CircuitPrg::new(circuit, wires, operation_mode))
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

            if failed < 16 {
                for s in results.iter().filter(|r| !r.passed() && !r.skipped()) {
                    println!(" f: {} [{}]", s.name, s.note.clone().unwrap());
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn keystream(
        circuit: CircuitSeq,
        wires: usize,
        counter: u256,
        mode: OperationMode,
        words: usize,
    ) -> Vec<u32> {
        let mut prg = CircuitPrg {
            circuit,
            counter,
            operation_mode: mode,
            wires,
            partial: 0.into(),
            partial_bits: 0,
        };
        (0..words).map(|_| prg.next_u32()).collect()
    }

    /// Independent bit packer: low `wires` bits of each block, least-significant bit first.
    fn oracle(
        circuit: &CircuitSeq,
        wires: usize,
        mut counter: u256,
        mode: OperationMode,
        words: usize,
    ) -> Vec<u32> {
        let mask = mask_bits(wires);
        let mut bits = Vec::with_capacity(words * 32);
        while bits.len() < words * 32 {
            let output = circuit.evaluate_256(counter) & mask;
            counter = match mode {
                OperationMode::Ctr => (counter + 1) & mask,
                OperationMode::Ofb => output,
            };
            for b in 0..wires {
                bits.push(((output >> b) & u256::one()).low_u32());
            }
        }
        bits.chunks(32)
            .take(words)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, bit)| acc | (bit << i))
            })
            .collect()
    }

    #[test]
    fn sequential_blocks_pack_into_words() {
        let samples = [
            (4, u256::from(0u8), OperationMode::Ctr),
            (12, u256::from(0xabcu16), OperationMode::Ctr),
            (32, u256::from(0x1234_5678u32), OperationMode::Ctr),
            (40, u256::from(7u8), OperationMode::Ofb),
            (64, u256::from(u64::MAX), OperationMode::Ctr),
            (128, u256::from(0x0123_4567_89ab_cdefu64), OperationMode::Ctr),
            (
                200,
                (u256::one() << 180) | u256::from(0x1234u16),
                OperationMode::Ofb,
            ),
        ];

        for (wires, counter, mode) in samples {
            let circuit = if wires >= 3 {
                CircuitSeq {
                    gates: vec![[0, 1, 2]],
                }
            } else {
                CircuitSeq { gates: vec![] }
            };
            let got = keystream(circuit.clone(), wires, counter, mode, 24);
            let expect = oracle(&circuit, wires, counter, mode, 24);
            assert_eq!(got, expect, "wires={wires} mode={mode:?}");
        }
    }

    #[test]
    fn pattern_repeats_across_the_low_half() {
        assert_eq!(pattern_bits(0), u256::zero());
        assert_eq!(pattern_bits(32), u256::from(0x7654_3210u32));
        assert_eq!(pattern_bits(64), u256::from(Y));
        assert_eq!(
            pattern_bits(128),
            u256::from(Y) | (u256::from(Y) << 64)
        );
    }
}
