// Expand a circuit against the regular frozen store (one sss-style inflation pass) so the
// instrumented compress command can be run on a realistic expanded circuit.
//
// Usage: expand_sample <in_circuit> <out_circuit> <n_wires> <trials>

use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::frozen::FrozenDb;
use local_mixing::replace::replace::{ExpandPairMode, expand_frozen};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = args.get(1).expect("in circuit");
    let dst = args.get(2).expect("out circuit");
    let n: usize = args.get(3).expect("n").parse().unwrap();
    let trials: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(3000);

    let c = CircuitSeq::from_string(&std::fs::read_to_string(src).unwrap());
    println!("in gates: {}", c.gates.len());

    let db = FrozenDb::from_env();

    let expanded = expand_frozen(&c, trials, n, &db, &ExpandPairMode::Regular);
    println!("out gates: {}", expanded.gates.len());
    std::fs::write(dst, expanded.repr()).unwrap();
}
