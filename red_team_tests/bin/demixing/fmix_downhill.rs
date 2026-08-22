//! Find locally profitable inverse-fmix conjugations using only the released
//! `mpmct1` circuit.  No mixer provenance, seed, or source circuit is used.
//!
//! For a same-target block A and an adjacent gate h that does not read A's
//! target, moving A across h conjugates A's ESOP control function by h.  If h
//! targets b, this is exactly the substitution b <- b XOR fire(h).  Forward
//! fmix R1 crossings expand one cube into a case-split ladder; the reverse
//! substitution can therefore collapse that ladder again.  This tool scans
//! both sides of every maximal same-target run and reports substitutions that
//! reduce the number of ESOP cubes after exact catalogue reduction.
//!
//! The scan/apply core lives in local_mixing::postprocessing::downhill, shared with
//! fcompress (which interleaves the same pass with its gather/reduce loop).

use clap::Parser;
use local_mixing::postprocessing::downhill::{apply_candidates, scan};
use local_mixing::engine::format;
use local_mixing::circuit::xgate::eval_lanes;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fmix_downhill")]
struct Args {
    /// Released mpmct1 challenge circuit.
    #[arg(long)]
    input: String,
    /// Print at most this many best candidates.
    #[arg(long, default_value_t = 30)]
    top: usize,
    /// Apply this many non-overlapping downhill passes (zero = scan only).
    #[arg(long, default_value_t = 0)]
    passes: usize,
    /// Write the demixed mpmct1 circuit after applying passes.
    #[arg(long)]
    output: Option<String>,
    /// Random 64-lane equivalence checks after rewriting.
    #[arg(long, default_value_t = 16)]
    verify_rounds: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let (mut gates, wires) = format::read_mpmct(&args.input).expect("read mpmct1 circuit");
    let original = gates.clone();
    let mut apply_rng = StdRng::seed_from_u64(args.seed ^ 0xD0DE);
    for pass in 0..=args.passes {
        let (runs, multi_runs, candidates) = scan(&gates);
        let gate_savings: usize = candidates
            .iter()
            .map(|c| c.before_gates - c.after_gates)
            .sum();
        println!(
            "[downhill] pass={} wires={} gates={} runs={} multi_runs={} profitable={} opportunity_gate_savings={}",
            pass,
            wires,
            gates.len(),
            runs,
            multi_runs,
            candidates.len(),
            gate_savings
        );
        for c in candidates.iter().take(args.top) {
            println!(
                "[candidate] side={} block={}..{} neighbor={} targets={}/{} gates={}->{} lits={}->{}",
                c.side,
                c.lo,
                c.hi,
                c.neighbor,
                c.block_target,
                c.neighbor_target,
                c.before_gates,
                c.after_gates,
                c.before_lits,
                c.after_lits
            );
        }
        if pass == args.passes || candidates.is_empty() {
            break;
        }
        let before = gates.len();
        let (next, _, swaps) = apply_candidates(gates, None, &candidates, &mut apply_rng, false);
        gates = next;
        println!(
            "[downhill] applied pass={} swaps={} gates {} -> {}",
            pass + 1,
            swaps,
            before,
            gates.len()
        );
    }

    if args.passes > 0 {
        let mut rng = StdRng::seed_from_u64(args.seed);
        for round in 0..args.verify_rounds {
            let state: Vec<u64> = (0..wires).map(|_| rng.random()).collect();
            let mut a = state.clone();
            let mut b = state;
            eval_lanes(original.iter(), &mut a);
            eval_lanes(gates.iter(), &mut b);
            assert_eq!(a, b, "global equivalence failed in round {round}");
        }
        println!(
            "[downhill] verified {} rounds x64 lanes; gates {} -> {}",
            args.verify_rounds,
            original.len(),
            gates.len()
        );
    }
    if let Some(path) = &args.output {
        format::write_mpmct(path, &gates, wires).expect("write output circuit");
        println!("[downhill] wrote {}", path);
    }
}
