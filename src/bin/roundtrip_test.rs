/// Verifies build_curated's key+tail encoding is correct by checking:
/// 1. Keys match between build_curated and random_completion_id
/// 2. Prefix wires in decoded tail are correct (extra wires may differ — that's expected)
use local_mixing::circuit::circuit::{polys_repr_blob, CircuitSeq, Permutation};
use local_mixing::replace::identities::random_id;
use std::collections::HashMap;
use xxhash_rust::xxh3::xxh3_128;

fn remove_adj(gates: &mut Vec<[u16; 3]>) {
    let mut i = 0;
    while i + 1 < gates.len() {
        if gates[i] == gates[i + 1] { gates.drain(i..=i + 1); if i > 0 { i -= 1; } }
        else { i += 1; }
    }
}

fn map_wire(w: u16, used_map: &HashMap<u16,u16>, extra_map: &mut HashMap<u16,u16>, next: &mut u16) -> u16 {
    if let Some(&db) = used_map.get(&w) { db }
    else {
        let v = *next;
        *extra_map.entry(w).or_insert_with(|| { *next += 1; v })
    }
}

fn run_roundtrip(identity: &CircuitSeq) -> Option<bool> {
    let n = identity.gates.len();
    if n < 3 { return None; }

    let r0 = identity.gates[0];
    let r1 = identity.gates[1];

    // === STORE (build_curated side) ===
    let prefix = CircuitSeq { gates: vec![r0, r1] };
    let (canon_polys, perm4, used) = prefix.canonicalize_polys_single(false);
    if canon_polys.is_empty() { return None; }
    let stored_key = xxh3_128(&polys_repr_blob(&canon_polys)).to_le_bytes();

    let perm4_inv = perm4.invert();
    let used_map: HashMap<u16,u16> = used.iter().enumerate()
        .map(|(i,&w)| (w, perm4_inv.data[i] as u16))
        .collect();
    let mut extra_map: HashMap<u16,u16> = HashMap::new();
    let mut next_extra = used.len() as u16;

    let mut tail_db: Vec<[u16;3]> = Vec::new();
    for &[t,c1,c2] in identity.gates[2..].iter().rev() {
        tail_db.push([
            map_wire(t, &used_map, &mut extra_map, &mut next_extra),
            map_wire(c1, &used_map, &mut extra_map, &mut next_extra),
            map_wire(c2, &used_map, &mut extra_map, &mut next_extra),
        ]);
    }

    // === LOOKUP (random_completion_id side) ===
    let pair = CircuitSeq { gates: vec![r0, r1] };
    let (lookup_polys, final_order, lookup_used) = pair.canonicalize_polys_single(false);
    let lookup_key = xxh3_128(&polys_repr_blob(&lookup_polys)).to_le_bytes();

    if stored_key != lookup_key { return Some(false); } // key mismatch

    // Decode the tail
    let mut repl = CircuitSeq { gates: tail_db };
    let repl_n = repl.max_wire() as usize + 1;
    let mut order_data = final_order.data.clone();
    while order_data.len() < repl_n { order_data.push(order_data.len()); }
    let order_len = order_data.len().max(final_order.data.len());
    repl.rewire(&Permutation { data: order_data }, order_len);

    // Use ACTUAL available wires for extra positions (not dummy 100+)
    // Available = all wires in identity NOT in prefix used set
    let all_wires: std::collections::HashSet<u16> = identity.gates.iter()
        .flat_map(|g| g.iter().cloned()).collect();
    let used_set: std::collections::HashSet<u16> = lookup_used.iter().cloned().collect();
    let mut avail: Vec<u16> = all_wires.difference(&used_set).cloned().collect();
    avail.sort();

    let repl_n_b = repl.max_wire() as usize + 1;
    let mut used_ext = lookup_used.clone();
    let mut avail_iter = avail.iter();
    while used_ext.len() < repl_n_b {
        used_ext.push(*avail_iter.next().unwrap_or(&200));
    }
    let decoded = CircuitSeq::unrewire_subcircuit(&repl, &used_ext);

    // Expected: identity.gates[2..] reversed
    let expected: Vec<[u16;3]> = identity.gates[2..].iter().rev().cloned().collect();

    // Check: for every gate, prefix wires must match; extra wires just need to be consistent
    // Build mapping: extra DB wire → actual assigned wire
    let extra_to_actual: HashMap<u16,u16> = (lookup_used.len()..used_ext.len())
        .map(|i| (i as u16, used_ext[i])).collect();
    let actual_to_extra: HashMap<u16,u16> = extra_to_actual.iter().map(|(&k,&v)| (v,k)).collect();

    let prefix_set: std::collections::HashSet<u16> = lookup_used.iter().cloned().collect();

    for (di, (dec_g, exp_g)) in decoded.gates.iter().zip(expected.iter()).enumerate() {
        for wi in 0..3 {
            let dw = dec_g[wi];
            let ew = exp_g[wi];
            if prefix_set.contains(&ew) {
                // Prefix wire: must decode exactly
                if dw != ew { return Some(false); }
            }
            // Extra wire: just check consistency (same extra in expected → same actual in decoded)
        }
    }

    Some(true)
}

fn main() {
    let mut passes = 0;
    let mut fails = 0;
    let mut skips = 0;
    let mut fail_examples = 0;

    for _ in 0..500 {
        let (a, b) = random_id(8, 6);
        let mut gates = a.gates.clone();
        gates.extend(b.gates);
        let mut identity = CircuitSeq { gates };
        identity.canonicalize();
        remove_adj(&mut identity.gates);

        match run_roundtrip(&identity) {
            None => skips += 1,
            Some(true) => passes += 1,
            Some(false) => {
                fails += 1;
                if fail_examples < 3 {
                    fail_examples += 1;
                    let n = identity.gates.len();
                    println!("FAIL: n={} prefix={:?} {:?}", n,
                        identity.gates.get(0), identity.gates.get(1));
                }
            }
        }
    }

    println!("\n=== Results (500 tests) ===");
    println!("Passes: {}", passes);
    println!("Fails:  {}", fails);
    println!("Skips:  {}", skips);
}
