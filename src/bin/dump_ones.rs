use local_mixing::sandwich::ChunkedCircuit;
use std::fs;
fn main() {
    let s = fs::read_to_string("challenges/lowweight128_obf.txt").unwrap();
    let mut prefix = 64usize;
    let mut body = String::new();
    for line in s.lines() {
        if let Some(r) = line.trim().strip_prefix("challenge_prefix_zeros ") {
            prefix = r.parse().unwrap();
            continue;
        }
        if line.trim().starts_with("challenge_seed ") {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    let c = ChunkedCircuit::from_readable_string(&body).unwrap();
    let x: u64 = 0x7d3b3d864606c8e8;
    let mut input = vec![false; c.data_wires];
    for i in 0..64 {
        input[prefix + i] = ((x >> i) & 1) != 0;
    }
    let y = c.evaluate_data_bits(&input);
    let ones: Vec<_> = y.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect();
    println!("wt={} ones={:?}", ones.len(), ones);
}
