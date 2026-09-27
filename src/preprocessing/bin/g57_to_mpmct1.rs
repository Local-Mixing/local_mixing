//! Throwaway converter: g57 -> mpmct1. Usage: g57_to_mpmct1 <in.g57> <out.mpmct1> [num_wires]
use local_mixing::engine::format::{read_g57_file, write_mpmct};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let gates = read_g57_file(&a[1]).expect("read g57");
    let max_w = gates
        .iter()
        .flat_map(|g| {
            std::iter::once(g.target).chain(g.ctrls.iter().map(|&(w, _)| w))
        })
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0);
    let nw = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(max_w);
    write_mpmct(&a[2], &gates, nw).expect("write mpmct1");
    println!("wrote {} ({} gates, {} wires; max_wire+1={max_w})", a[2], gates.len(), nw);
}
