// Expand a circuit against the shard DB (one sss-style inflation pass) so the
// instrumented compress command can be run on a realistic expanded circuit.
//
// Usage: expand_sample <in_circuit> <out_circuit> <n_wires> <trials>

use lmdb::{Environment, EnvironmentFlags};
use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::main_mix::open_all_dbs;
use local_mixing::replace::replace::{ExpandPairMode, expand_lmdb};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let src = args.get(1).expect("in circuit");
    let dst = args.get(2).expect("out circuit");
    let n: usize = args.get(3).expect("n").parse().unwrap();
    let trials: usize = args.get(4).map(|v| v.parse().unwrap()).unwrap_or(3000);

    let c = CircuitSeq::from_string(&std::fs::read_to_string(src).unwrap());
    println!("in gates: {}", c.gates.len());

    let mut env_flags = EnvironmentFlags::READ_ONLY | EnvironmentFlags::NO_LOCK;
    if std::env::var("LMDB_READAHEAD").map(|v| v == "1") != Ok(true) {
        env_flags |= EnvironmentFlags::NO_READAHEAD;
    }
    let env = Environment::new()
        .set_flags(env_flags)
        .set_max_dbs(556)
        .set_max_readers(10000)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(Path::new("./db"))
        .expect("open lmdb");
    let (shard_dbs, _) = open_all_dbs(&env);

    let expanded = expand_lmdb(&c, trials, n, &env, &shard_dbs, &ExpandPairMode::Db);
    println!("out gates: {}", expanded.gates.len());
    std::fs::write(dst, expanded.repr()).unwrap();
}
