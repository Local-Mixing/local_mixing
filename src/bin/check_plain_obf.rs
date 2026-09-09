
use local_mixing::circuit::CircuitSeq;
use local_mixing::sandwich::{hamming_weight, ChunkedCircuit};
use primitive_types::U256 as u256;
use std::fs;
use std::env;

fn load_plain(path: &str) -> (usize, CircuitSeq) {
    let s = fs::read_to_string(path).unwrap();
    let mut n = 0usize;
    let mut gates = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        if let Some(r) = t.strip_prefix("n ") { n = r.parse().unwrap(); continue; }
        if t.starts_with("m ") { continue; }
        let p: Vec<u16> = t.split(',').map(|x| x.parse().unwrap()).collect();
        gates.push([p[0], p[1], p[2]]);
    }
    (n, CircuitSeq { gates })
}

fn load_obf(path: &str) -> (ChunkedCircuit, usize) {
    let s = fs::read_to_string(path).unwrap();
    let mut prefix = 64usize;
    let mut body = String::new();
    for line in s.lines() {
        if let Some(r) = line.trim().strip_prefix("challenge_prefix_zeros ") {
            prefix = r.parse().unwrap(); continue;
        }
        if line.trim().starts_with("challenge_seed ") { continue; }
        body.push_str(line); body.push('\n');
    }
    (ChunkedCircuit::from_readable_string(&body).unwrap(), prefix)
}

fn main() {
    let xhex = env::args().nth(1).unwrap();
    let x = u64::from_str_radix(xhex.trim_start_matches("0x"), 16).unwrap();
    let (n, plain) = load_plain("challenges/lowweight128_plain.SEALED.txt");
    let (obf, prefix) = load_obf("challenges/lowweight128_obf.txt");
    assert_eq!(n, obf.data_wires);
    let mut input = vec![false; n];
    for i in 0..64 {
        input[prefix + i] = ((x >> i) & 1) != 0;
    }
    let mut state = u256::zero();
    for i in 0..64 {
        if ((x >> i) & 1) != 0 { state |= u256::one() << (prefix + i); }
    }
    let pout = plain.evaluate_256(state);
    let mut pbits = Vec::new();
    for i in 0..n { pbits.push(((pout >> i) & u256::one()) == u256::one()); }
    let obits = obf.evaluate_data_bits(&input);
    println!("x=0x{:016x}", x);
    println!("plain_wt={} obf_wt={}", hamming_weight(&pbits), hamming_weight(&obits));
    println!("outputs_equal={}", pbits == obits);
}
