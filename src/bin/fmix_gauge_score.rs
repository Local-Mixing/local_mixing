//! Rank assignments to a semantically free input block by exact three-valued
//! propagation through a released mpmct1 circuit.
//!
//! In the cdcnot challenge, fixing the initial accumulator z is sound because
//! the predicate is z_out XOR z_in = target and the promised construction makes
//! that difference independent of z.  Different z assignments nevertheless
//! expose different constants to the mixed gate representation.  This tool
//! scores deterministic random gauges without invoking a SAT solver.

use clap::Parser;
use local_mixing::postmix::format;
use local_mixing::postmix::xgate::XGate;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fmix_gauge_score")]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    fixed_start: usize,
    #[arg(long)]
    fixed_bits: usize,
    #[arg(long, default_value_t = 256)]
    samples: usize,
    #[arg(long, default_value_t = 20)]
    top: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Score {
    noops: u64,
    fixed_fires: u64,
    true_lits: u64,
    false_shortcuts: u64,
    final_constants: u64,
}

fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 != 0
}

fn score(gates: &[XGate], wires: usize, start: usize, bits: usize, gauge: &[u64]) -> Score {
    let mut state = vec![None; wires];
    for i in 0..bits {
        state[start + i] = Some(bit(gauge, i));
    }
    let mut s = Score::default();
    for g in gates {
        let mut any_false = false;
        let mut all_true = true;
        for &(w, polarity) in &g.ctrls {
            match state[w as usize] {
                Some(v) if v == polarity => s.true_lits += 1,
                Some(_) => {
                    any_false = true;
                    all_true = false;
                    s.false_shortcuts += 1;
                    break;
                }
                None => all_true = false,
            }
        }
        let and_value = if any_false {
            Some(false)
        } else if all_true {
            Some(true)
        } else {
            None
        };
        let fire = and_value.map(|v| v ^ g.comp);
        match fire {
            Some(false) => s.noops += 1,
            Some(true) => {
                s.fixed_fires += 1;
                if let Some(v) = state[g.target as usize] {
                    state[g.target as usize] = Some(!v);
                }
            }
            None => state[g.target as usize] = None,
        }
    }
    s.final_constants = state.iter().filter(|x| x.is_some()).count() as u64;
    s
}

fn hex_gauge(words: &[u64], bits: usize) -> String {
    let digits = bits.div_ceil(4);
    let mut full = String::new();
    for &w in words.iter().rev() {
        full.push_str(&format!("{w:016x}"));
    }
    let first = full.len().saturating_sub(digits);
    format!("0x{}", &full[first..])
}

fn main() {
    let args = Args::parse();
    let (gates, wires) = format::read_mpmct(&args.input).expect("read mpmct1 circuit");
    assert!(args.fixed_start + args.fixed_bits <= wires);
    let words = args.fixed_bits.div_ceil(64);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut ranked = Vec::with_capacity(args.samples + 1);
    let zero = vec![0u64; words];
    ranked.push((
        score(&gates, wires, args.fixed_start, args.fixed_bits, &zero),
        zero,
    ));
    for _ in 0..args.samples {
        let mut gauge: Vec<u64> = (0..words).map(|_| rng.random()).collect();
        if args.fixed_bits % 64 != 0 {
            *gauge.last_mut().unwrap() &= (1u64 << (args.fixed_bits % 64)) - 1;
        }
        ranked.push((
            score(&gates, wires, args.fixed_start, args.fixed_bits, &gauge),
            gauge,
        ));
    }
    ranked.sort_unstable_by_key(|(s, _)| std::cmp::Reverse(*s));
    for (rank, (s, gauge)) in ranked.iter().take(args.top).enumerate() {
        println!(
            "rank={} gauge={} noops={} fixed_fires={} true_lits={} false_shortcuts={} final_constants={}",
            rank + 1,
            hex_gauge(gauge, args.fixed_bits),
            s.noops,
            s.fixed_fires,
            s.true_lits,
            s.false_shortcuts,
            s.final_constants
        );
    }
}
