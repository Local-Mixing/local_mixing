use std::{
    cmp::max,
    collections::HashMap,
    io::{self, Read},
    os::unix::io::AsRawFd,
    sync::atomic::Ordering,
    time::Instant,
};

use libc::{fcntl, F_GETFL, F_SETFL, O_NONBLOCK};

use rand::{prelude::SliceRandom, Rng};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

extern crate lmdb_sys;

use crate::{
    circuit::circuit::CircuitSeq,
    random::random_data::{random_circuit, shoot_random_gate_gate_ver},
    replace::{
        identities::{get_random_identity, random_canonical_id},
        replace::IDENTITY_TIME,
    },
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gate taxonomy and Replacement Pair 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollisionType {
    OnActive,
    OnCtrl1,
    OnCtrl2,
    OnNew,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GatePair {
    pub a: CollisionType,
    pub c1: CollisionType,
    pub c2: CollisionType
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GateTri {
    first: GatePair,
    second: GatePair,
    gap: GatePair,
}

impl GatePair {
    pub fn new() -> Self {
        GatePair { a: CollisionType::OnNew, c1: CollisionType::OnNew, c2: CollisionType::OnNew }
    }

    pub fn is_none(gate_pair: &Self) -> bool {
        gate_pair.a == CollisionType::OnNew && gate_pair.c1 == CollisionType::OnNew && gate_pair.c2 == CollisionType::OnNew
    }

    pub fn to_int(gp: &Self) -> usize {
        let a = gp.a;
        let b = gp.c1;
        let c = gp.c2;

        if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnNew {
            0
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnNew {
            1
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnNew {
            2
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnNew {
            3
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnNew {
            4
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            5
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            6
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnActive {
            7
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            8
        } else if a == CollisionType::OnNew && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            9
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            10
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            11
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            12
        } else if a == CollisionType::OnActive && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            13
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl1 && c == CollisionType::OnCtrl2 {
            14
        } else if a == CollisionType::OnActive && b == CollisionType::OnCtrl2 && c == CollisionType::OnCtrl1 {
            15
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnActive && c == CollisionType::OnNew {
            16
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnCtrl2 && c == CollisionType::OnNew {
            17
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnActive {
            18
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnNew && c == CollisionType::OnCtrl2 {
            19
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnActive && c == CollisionType::OnCtrl2 {
            20
        } else if a == CollisionType::OnCtrl1 && b == CollisionType::OnCtrl2 && c == CollisionType::OnActive {
            21
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnActive && c == CollisionType::OnNew {
            22
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnCtrl1 && c == CollisionType::OnNew {
            23
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnActive {
            24
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnNew && c == CollisionType::OnCtrl1 {
            25
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnActive && c == CollisionType::OnCtrl1 {
            26
        } else if a == CollisionType::OnCtrl2 && b == CollisionType::OnCtrl1 && c == CollisionType::OnActive {
            27
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnCtrl1 {
            28
        } else if a == CollisionType::OnNew && b == CollisionType::OnActive && c == CollisionType::OnCtrl2 {
            29
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnActive {
            30
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl1 && c == CollisionType::OnCtrl2 {
            31
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnActive {
            32
        } else if a == CollisionType::OnNew && b == CollisionType::OnCtrl2 && c == CollisionType::OnCtrl1 {
            33
        } else {
            panic!("Not a valid GatePair");
        }
    }

    pub fn from_int(i: usize) -> Self {
        use CollisionType::*;

        match i {
            0 => GatePair { a: OnNew, c1: OnNew, c2: OnNew },
            1 => GatePair { a: OnActive, c1: OnNew, c2: OnNew },
            2 => GatePair { a: OnCtrl1,  c1: OnNew, c2: OnNew },
            3 => GatePair { a: OnCtrl2,  c1: OnNew, c2: OnNew },
            4 => GatePair { a: OnNew, c1: OnActive, c2: OnNew },
            5 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnNew },
            6 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnNew },
            7 => GatePair { a: OnNew, c1: OnNew, c2: OnActive },
            8 => GatePair { a: OnNew, c1: OnNew, c2: OnCtrl1 },
            9 => GatePair { a: OnNew, c1: OnNew, c2: OnCtrl2 },
            10 => GatePair { a: OnActive, c1: OnCtrl1, c2: OnNew },
            11 => GatePair { a: OnActive, c1: OnCtrl2, c2: OnNew },
            12 => GatePair { a: OnActive, c1: OnNew, c2: OnCtrl1 },
            13 => GatePair { a: OnActive, c1: OnNew, c2: OnCtrl2 },
            14 => GatePair { a: OnActive, c1: OnCtrl1, c2: OnCtrl2 },
            15 => GatePair { a: OnActive, c1: OnCtrl2, c2: OnCtrl1 },
            16 => GatePair { a: OnCtrl1, c1: OnActive, c2: OnNew },
            17 => GatePair { a: OnCtrl1, c1: OnCtrl2, c2: OnNew },
            18 => GatePair { a: OnCtrl1, c1: OnNew, c2: OnActive },
            19 => GatePair { a: OnCtrl1, c1: OnNew, c2: OnCtrl2 },
            20 => GatePair { a: OnCtrl1, c1: OnActive, c2: OnCtrl2 },
            21 => GatePair { a: OnCtrl1, c1: OnCtrl2, c2: OnActive },
            22 => GatePair { a: OnCtrl2, c1: OnActive, c2: OnNew },
            23 => GatePair { a: OnCtrl2, c1: OnCtrl1, c2: OnNew },
            24 => GatePair { a: OnCtrl2, c1: OnNew, c2: OnActive },
            25 => GatePair { a: OnCtrl2, c1: OnNew, c2: OnCtrl1 },
            26 => GatePair { a: OnCtrl2, c1: OnActive, c2: OnCtrl1 },
            27 => GatePair { a: OnCtrl2, c1: OnCtrl1, c2: OnActive },
            28 => GatePair { a: OnNew, c1: OnActive, c2: OnCtrl1 },
            29 => GatePair { a: OnNew, c1: OnActive, c2: OnCtrl2 },
            30 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnActive },
            31 => GatePair { a: OnNew, c1: OnCtrl1, c2: OnCtrl2 },
            32 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnActive },
            33 => GatePair { a: OnNew, c1: OnCtrl2, c2: OnCtrl1 },

            _ => panic!("Invalid GatePair index"),
        }
    }
}

pub fn get_collision_type(g1: &[u8; 3], pin: u8) -> CollisionType {
    match pin {
        x if x == g1[0] => CollisionType::OnActive,
        x if x == g1[1] => CollisionType::OnCtrl1,
        x if x == g1[2] => CollisionType::OnCtrl2,
        _ => CollisionType::OnNew,
    }
}

pub fn gate_pair_taxonomy(g1: &[u8;3], g2: &[u8;3]) -> GatePair {
    GatePair {
        a: get_collision_type(&g1, g2[0]),
        c1: get_collision_type(&g1, g2[1]),
        c2: get_collision_type(&g1, g2[2]),
    }
}

fn gate_tri_taxonomy(g0: &[u8;3], g1: &[u8;3], g2: &[u8;3]) -> GateTri {
    GateTri {
        first: gate_pair_taxonomy(g0, g1),
        second: gate_pair_taxonomy(g1, g2),
        gap: gate_pair_taxonomy(g0, g2)
    }
}

// Partitions circuit into pairs and then replaces each pair
pub fn replace_pairs(circuit: &mut CircuitSeq, num_wires: usize, conn: &mut Connection, env: &lmdb::Environment) {
    println!("Starting replace_pairs, circuit length: {}", circuit.gates.len());
    // let start = circuit.clone();
    let mut pairs: HashMap<GatePair, Vec<usize>> = HashMap::new();
    let gates = circuit.gates.clone();
    let m = circuit.gates.len();
    let mut replaced = 0;
    let mut to_replace: Vec<(Vec<[u8;3]>, Vec<[u8;3]>)> = vec![(Vec::new(), Vec::new()); m / 2];
    if m < 2 {
        println!("Circuit too small, returning");
        return;
    }

    println!("Building taxonomy pairs...");
    let mut i = 0;
    while i + 1 < m {
        let g1 = gates[i];
        let g2 = gates[i + 1];
        let taxonomy = gate_pair_taxonomy(&g1, &g2);

        if !GatePair::is_none(&taxonomy) {
            pairs.entry(taxonomy)
                .or_default()
                .push(i);
        }
        i += 2;
    }
    let num_pairs: usize = pairs.values().map(|v| v.len()).sum();
    println!("Pairs collected: {}", num_pairs);
    
    let mut rng = rand::rng();
    let mut fail = 0;
    while !pairs.is_empty() && fail < 100 {
        let n = rng.random_range(5..=7);
        let mut id = match random_canonical_id(&env, conn, n) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut replaced = false;

        // Forward scan: every adjacent pair
        for i in 0..id.gates.len() - 1 {
            let tax = gate_pair_taxonomy(&id.gates[i], &id.gates[i + 1]);
            if let Some(v) = pairs.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove the matching pair and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());
                    // Append gates after the pair
                    new_circuit.extend_from_slice(&id.gates[i + 2..]);
                    // Append gates before the pair
                    new_circuit.extend(id.gates[0..i].iter());
                    // let nc = CircuitSeq { gates: new_circuit.clone() };
                    // if nc.probably_equal(&CircuitSeq { gates: vec![id.gates[i+1], id.gates[i]]}, num_wires, 10000).is_err() { panic!("pairs dont match new"); }
                    to_replace[chosen / 2] = (new_circuit, vec![id.gates[i], id.gates[i+1]]);

                    if v.is_empty() {
                        pairs.remove(&tax);
                    }

                    replaced = true;
                    break; // stop scanning once a match is found
                }
            }
        }

        if replaced {
            continue;
        }

        // Reverse scan: every adjacent pair in reverse
        id.gates.reverse();
        for i in 0..id.gates.len() - 1 {
            let tax = gate_pair_taxonomy(&id.gates[i], &id.gates[i + 1]);
            if let Some(v) = pairs.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove the matching pair and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());
                    // Append gates after the pair
                    new_circuit.extend_from_slice(&id.gates[i + 2..]);
                    // Append gates before the pair, in reverse
                    new_circuit.extend(id.gates[0..i].iter());
                    // let nc = CircuitSeq { gates: new_circuit.clone() };
                    // if nc.probably_equal(&CircuitSeq { gates: vec![id.gates[i+1], id.gates[i]]}, num_wires, 10000).is_err() { panic!("reverse pairs dont match new"); }
                    to_replace[chosen / 2] = (new_circuit, vec![id.gates[i], id.gates[i+1]]);
                    
                    if v.is_empty() {
                        pairs.remove(&tax);
                    }

                    replaced = true;
                    break; // stop scanning once a match is found
                }
            }
        }

        if !replaced {
            fail += 1;
        }
    }

    println!("Applying replacements...");
    for (i, replacement) in to_replace.into_iter().enumerate().rev() {
        if replacement.0.is_empty() {
            continue;
        }

        // println!("Replacing at pair index {}", i);
        replaced += 1;
        let index = 2 * i;
        let (g1, g2) = (circuit.gates[index], circuit.gates[index + 1]);
        let replacement_circ = CircuitSeq { gates: replacement.0 };
        let mut used_wires: Vec<u8> = vec![(num_wires + 1) as u8; max(replacement_circ.max_wire(), CircuitSeq { gates: replacement.1.clone() }.max_wire()) + 1];
        let mut used = vec![0u8; max(replacement_circ.max_wire(), CircuitSeq { gates: replacement.1.clone() }.max_wire()) + 1];
        used_wires[replacement.1[0][0] as usize] = g1[0];
        used_wires[replacement.1[0][1] as usize] = g1[1];
        used_wires[replacement.1[0][2] as usize] = g1[2];
        used[replacement.1[0][0] as usize] = 1;
        used[replacement.1[0][1] as usize] = 1;
        used[replacement.1[0][2] as usize] = 1;
        // println!("Original wires: {:?}, used_wires initialized", used_wires);

        // println!("Gates g1: {:?} g2: {:?}", g1, g2);

        let tax = gate_pair_taxonomy(&g1, &g2);
        if tax.a == CollisionType::OnNew || tax.c1 == CollisionType::OnNew || tax.c2 == CollisionType::OnNew {
            // println!("Found OnNew collision, assigning new wires...");
        }

        // Assign new wires if OnNew
        let mut i = 0;
        for collision in &[tax.a, tax.c1, tax.c2] {
            if *collision == CollisionType::OnNew {
                used_wires[replacement.1[1][i] as usize] = g2[i]
            }
            i += 1;
        }

        // Fill any remaining placeholders
        for i in 0..used_wires.len() {
            if used[i] == 0 {
                loop {
                    let wire = rng.random_range(0..=(num_wires-1) as u8);
                    if used_wires.contains(&wire) {
                        continue
                    }
                    used_wires[i] = wire;
                    used[i]=1;
                    break
                }
            }
        }

        // println!("Final used_wires for this replacement: {:?}", used_wires);

        // if replacement.probably_equal(&CircuitSeq { gates: vec![[1,2,3], [1,2,3]]}, 64, 100000).is_err() {
        //     panic!("Replacement is not an id");
        // }
        circuit.gates.splice(
            index..=index + 1,
            CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                .gates
                .into_iter()
                .rev(),
        );

        // println!("Replacement: {:?}", CircuitSeq::unrewire_subcircuit(&replacement, &used_wires));
        // println!("Replacement applied at indices {}..{}", index, index + 1);
        // println!("Replacements so far: {}/{}", replaced, num_pairs);
    }
    println!("Replaced {}/{} pairs", replaced, num_pairs);
    // println!("Starting single gate replacements");
    // random_gate_replacements(circuit, min((num_pairs - replaced)/20 + (m/2 - num_pairs)/20, 1000), num_wires, conn, env);
    // if start.probably_equal(&circuit, num_wires, 10000).is_err() {
    //     panic!("replace pairs changed something");
    // }
    println!("Finished replace_pairs");
}

fn make_stdin_nonblocking() {
    let fd = io::stdin().as_raw_fd();
    unsafe {
        let flags = fcntl(fd, F_GETFL);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
}

// Do sequential method of replacing pairs
pub fn replace_sequential_pairs(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    _conn: &mut Connection,
    env: &lmdb::Environment,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    tower: bool
) -> (usize, usize, usize, usize) {
    make_stdin_nonblocking();
    let gates = circuit.gates.clone();
    let n = gates.len();
    if n < 2 {
        println!("Circuit too small, returning");
        return (0, 0, 0, 0);
    }

    let mut already_collided = 0;
    let shoot_count = 0;
    let curr_zero = 0;
    let traverse_left = 0;
    // let mut shoot_count = 0;
    // let mut curr_zero = 0;
    // let mut traverse_left = 0;

    let mut rng = rand::rng();
    let mut out: Vec<[u8; 3]> = Vec::new();

    // rolling state
    let mut left = gates[0];
    let mut i = 1;
    let mut fail = 0;

    while i < n {
        let mut buf = [0u8; 1];
        if let Ok(n) = io::stdin().read(&mut buf) {
            if n > 0 && buf[0] == b'\n' {
                println!("  i = {}", i);
            }
        }
        let right = gates[i];
        let tax = gate_pair_taxonomy(&left, &right);

        // if !GatePair::is_none(&tax) {
            already_collided += 1;
            let mut produced: Option<Vec<[u8; 3]>> = None;

            while produced.is_none() && fail < 100 {
                fail += 1;
                let id_len = if GatePair::is_none(&tax) {
                    let r = rng.random_range(0..100);
                    match r { 
                        0..45 => 6,   
                        45..90 => 7,   
                        _       => 16, 
                    }
                } else {
                    let r = rng.random_range(0..100);
                    match r {
                        0..30  => 5,   
                        30..60 => 6,   
                        60..90 => 7,   
                        _       => 16, 
                    }
                };
                // let id_len = 128;
                let t_id = Instant::now();
                let id = match get_random_identity(id_len, tax, env, dbs, tower) {
                    Ok(id) => {
                        IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        id
                    }
                    Err(_) => {
                        IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        fail += 1;
                        continue;
                    }
                };

                let new_circuit = id.gates[2..].to_vec();
                let replacement_circ = CircuitSeq { gates: new_circuit };

                let mut used_wires: Vec<u8> = vec![
                    (num_wires + 1) as u8;
                    std::cmp::max(
                        replacement_circ.max_wire(),
                        CircuitSeq {
                            gates: vec![id.gates[0], id.gates[1]],
                        }
                        .max_wire(),
                    ) + 1
                ];
                let mut used = vec![
                    0u8;
                    std::cmp::max(
                        replacement_circ.max_wire(),
                        CircuitSeq {
                            gates: vec![id.gates[0], id.gates[1]],
                        }
                        .max_wire(),
                    ) + 1];
                used_wires[id.gates[0][0] as usize] = left[0];
                used_wires[id.gates[0][1] as usize] = left[1];
                used_wires[id.gates[0][2] as usize] = left[2];
                used[id.gates[0][0] as usize] = 1;
                used[id.gates[0][1] as usize] = 1;
                used[id.gates[0][2] as usize] = 1;
                let mut k = 0;
                for collision in &[tax.a, tax.c1, tax.c2] {
                    if *collision == CollisionType::OnNew {
                        used_wires[id.gates[1][k] as usize] = right[k];
                    }
                    k += 1;
                }

                let mut available_wires: Vec<u8> = (0..=(num_wires-1) as u8)
                    .filter(|w| !used_wires.contains(w))
                    .collect();

                available_wires.shuffle(&mut rng);
                for w in 0..used_wires.len() {
                    if used[w] == 0 {
                        if let Some(&wire) = available_wires.get(0) {
                            used_wires[w] = wire;
                            used[w] = 1;
                            available_wires.remove(0);
                        } else {
                            panic!("No available wires left to assign!");
                        }
                    }
                }

                produced = Some(
                    CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                        .gates
                        .into_iter()
                        .rev()
                        .collect()
                );

                fail += 1;
            }

            if let Some(mut gates_out) = produced {
                out.append(&mut gates_out);
                left = out.pop().unwrap();
            } else {
                out.push(left);
                left = right;
            }

            fail = 0;
            i += 1;
        // Old code to use if the two gates do not touch in any way
        // } else {
        //     shoot_count += 1;
        //     out.push(gates[i]);
        //     let out_len = out.len();

        //     let new_index = shoot_left_vec(&mut out, out_len - 1);
        //     traverse_left += out_len - 1 - new_index;

        //     if new_index == 0 {
        //         curr_zero += 1;
        //         let g = &out[0];
        //         let temp_out_circ = CircuitSeq { gates: out.clone() };
        //         let num = rng.random_range(3..=7);

        //         if let Ok(mut id) = random_canonical_id(env, &conn, num) {
        //             let mut used_wires = vec![g[0], g[1], g[2]];
        //             let mut count = 3;

        //             while count < num {
        //                 let random = rng.random_range(0..num_wires);
        //                 if used_wires.contains(&(random as u8)) {
        //                     continue;
        //                 }
        //                 used_wires.push(random as u8);
        //                 count += 1;
        //             }
        //             used_wires.sort();

        //             let rewired_g =
        //                 CircuitSeq::rewire_subcircuit(&temp_out_circ, &vec![0], &used_wires);
        //             id.rewire_first_gate(rewired_g.gates[0], num);
        //             id = CircuitSeq::unrewire_subcircuit(&id, &used_wires);
        //             id.gates.remove(0);

        //             out.splice(0..1, id.gates);
        //         }

        //         fail = 0;
        //         i += 1;
        //         continue;
        //     }

        //     let left_gate = out[new_index - 1];
        //     let right_gate = out[new_index];
        //     let tax = gate_pair_taxonomy(&left_gate, &right_gate);

        //     if !GatePair::is_none(&tax) {
        //         let mut produced: Option<Vec<[u8; 3]>> = None;

        //         while produced.is_none() && fail < 100 {
        //             fail += 1;
        //             let id_len = rng.random_range(5..=7);

        //             let t_id = Instant::now();
        //             let id = match get_random_identity(id_len, tax, env, dbs) {
        //                 Ok(id) => {
        //                     IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
        //                     id
        //                 }
        //                 Err(_) => {
        //                     IDENTITY_TIME.fetch_add(t_id.elapsed().as_nanos() as u64, Ordering::Relaxed);
        //                     fail += 1;
        //                     continue;
        //                 }
        //             };

        //             let new_circuit = id.gates[2..].to_vec();
        //             let replacement_circ = CircuitSeq { gates: new_circuit };

        //             let mut used_wires: Vec<u8> = vec![
        //                 (num_wires + 1) as u8;
        //                 std::cmp::max(
        //                     replacement_circ.max_wire(),
        //                     CircuitSeq {
        //                         gates: vec![id.gates[0], id.gates[1]],
        //                     }
        //                     .max_wire(),
        //                 ) + 1
        //             ];

        //             used_wires[id.gates[0][0] as usize] = left_gate[0];
        //             used_wires[id.gates[0][1] as usize] = left_gate[1];
        //             used_wires[id.gates[0][2] as usize] = left_gate[2];

        //             let mut k = 0;
        //             for collision in &[tax.a, tax.c1, tax.c2] {
        //                 if *collision == CollisionType::OnNew {
        //                     used_wires[id.gates[1][k] as usize] = right_gate[k];
        //                 }
        //                 k += 1;
        //             }

        //             let mut available_wires: Vec<u8> = (0..num_wires as u8)
        //                 .filter(|w| !used_wires.contains(w))
        //                 .collect();

        //             available_wires.shuffle(&mut rng);
        //             for w in 0..used_wires.len() {
        //                 if used_wires[w] == (num_wires + 1) as u8 {
        //                     if let Some(&wire) = available_wires.get(0) {
        //                         used_wires[w] = wire;
        //                         available_wires.remove(0);
        //                     } else {
        //                         panic!("No available wires left to assign!");
        //                     }
        //                 }
        //             }

        //             produced = Some(
        //                 CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
        //                     .gates
        //                     .into_iter()
        //                     .rev()
        //                     .collect()
        //             );

        //             fail += 1;
        //         }

        //         if let Some(mut gates_out) = produced {
        //             out.splice((new_index - 1)..=new_index, gates_out.drain(..));
        //         }

        //         fail = 0;
        //         i += 1;
        //     }
        // }
    }

    out.push(left);
    circuit.gates = out;

    (already_collided, shoot_count, curr_zero, traverse_left)
}

// returns the id-2 and the length
pub fn replace_single_pair(
    left: &[u8;3],
    right: &[u8;3],
    num_wires: usize,
    _conn: &mut Connection,
    env: &lmdb::Environment,
    _bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    tower: bool
) -> (Vec<[u8;3]>, usize) {
    make_stdin_nonblocking();
    let mut rng = rand::rng();
    let tax = gate_pair_taxonomy(&left, &right);
    let mut id_gen = false;
    let mut id = CircuitSeq { gates: Vec::new() };
    while !id_gen {
        let id_len = if GatePair::is_none(&tax) {
            let r = rng.random_range(0..100);
            match r { 
                0..45 => 6,   
                45..90 => 7,   
                _       => 16, 
            }
        } else {
            let r = rng.random_range(0..100);
            match r {
                0..30  => 5,   
                30..60 => 6,   
                60..90 => 7,   
                _       => 16, 
            }
        };
        // let id_len = 128;
        id = match get_random_identity(id_len, tax, env, dbs, tower) {
            Ok(id) => {
                id_gen = true;
                id
            },
            Err(_) => {
                continue;
            }
        };
    }

    let new_circuit = id.gates[2..].to_vec();

    let replacement_circ = CircuitSeq { gates: new_circuit };

    let mut used_wires: Vec<u8> = vec![
        (num_wires + 1) as u8;
        std::cmp::max(
            replacement_circ.max_wire(),
            CircuitSeq {
                gates: vec![id.gates[0], id.gates[1]],
            }
            .max_wire(),
        ) + 1
    ];
    let mut used = vec![
        0u8; 
        std::cmp::max(
            replacement_circ.max_wire(),
            CircuitSeq {
                gates: vec![id.gates[0], id.gates[1]],
            }
            .max_wire(),
        ) + 1
    ];
    used_wires[id.gates[0][0] as usize] = left[0];
    used_wires[id.gates[0][1] as usize] = left[1];
    used_wires[id.gates[0][2] as usize] = left[2];
    used[id.gates[0][0] as usize] = 1;
    used[id.gates[0][1] as usize] = 1;
    used[id.gates[0][2] as usize] = 1;
    let mut k = 0;
    for collision in &[tax.a, tax.c1, tax.c2] {
        if *collision == CollisionType::OnNew {
            used_wires[id.gates[1][k] as usize] = right[k];
        }
        k += 1;
    }

    let mut available_wires: Vec<u8> = (0..=(num_wires-1) as u8)
        .filter(|w| !used_wires.contains(w))
        .collect();
    
    available_wires.shuffle(&mut rng);
    for w in 0..used_wires.len() {
        if used[w] == 0 {
            if let Some(&wire) = available_wires.get(0) {
                used_wires[w] = wire;
                used[w] = 1;
                available_wires.remove(0);
            } else {
                panic!("No available wires left to assign!");
            }
        }
    }


    (CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
        .gates
        .into_iter()
        .rev()
        .collect(),
    id.gates.len() - 2)
}

// replace pairs for RCD method
pub fn replace_pair_distances(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    tower: bool
) {
    let min = 30;

    let mut distances = vec![0usize; circuit.gates.len() + 1];

    let (mut left, mut right) = update_bounds(&distances);

    let mut curr = 0;
    loop {
        // Termination condition
        if curr >= min {
            break;
        }
        let mut pending: Vec<(usize, usize, Vec<[u8; 3]>)> = Vec::new();

        // scan
        let mut i = left + 1;
        while i < right {
            let mut buf = [0u8; 1];
            if let Ok(n) = io::stdin().read(&mut buf) {
                if n > 0 && buf[0] == b'\n' {
                    println!("  curr = {}\n
                                gates = {}", curr, circuit.gates.len());
                }
            }
            if distances[i] == curr {
                let (id, id_len) = replace_single_pair(
                    &circuit.gates[i - 1],
                    &circuit.gates[i],
                    num_wires,
                    conn,
                    env,
                    bit_shuf_list,
                    dbs,
                    tower,
                );

                // Save what to do later
                if curr == 0 {
                    circuit.gates.splice(i - 1..=i, id);
                    update_distance(&mut distances, i, id_len);

                    let (l, r) = update_bounds(&distances);
                    left = l;
                    right = r;

                    continue;
                } else {
                    pending.push((i, id_len, id));
                }
            }
            i += 1;
        }

        // Nothing at this level, move up
        if pending.is_empty() {
            curr += 1;
            continue;
        }

        // replace
        pending.reverse();

        for (i, id_len, id) in pending {
            circuit.gates.splice(i - 1..=i, id);
            update_distance(&mut distances, i, id_len);
        }

        // Recompute bounds once after batch
        let (l, r) = update_bounds(&distances);
        curr += 1;
        left = l;
        right = r;
    }
}

// Move the bounds inwards
// Left bound is when the ascending stops
// Right bound is when the descending begins
fn update_bounds(distances: &[usize]) -> (usize, usize) {
    let mut left = 0;
    while left + 1 < distances.len()
        && distances[left + 1] == distances[left] + 1
    {
        left += 1;
    }

    let mut right = distances.len() - 1;
    while right > 0
        && distances[right - 1] == distances[right] + 1
    {
        right -= 1;
    }

    (left, right)
}

// Update distances after a pair is replaced
pub fn update_distance(
    distances: &mut Vec<usize>,
    didx: usize,
    id_len: usize,
) {
    let k = id_len - 1;

    let left0 = distances[didx - 1] + 1;
    let right0 = distances[didx + 1] + 1;

    let mut replacement = Vec::with_capacity(k);

    for i in 0..k {
        let from_left = left0 + i;
        let from_right = right0 + (k - 1 - i);
        replacement.push(from_left.min(from_right));
    }

    distances.splice(didx..=didx, replacement);
}

// Faster method for RCD
pub fn replace_pair_distances_linear(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
    bit_shuf_list: &Vec<Vec<Vec<usize>>>,
    dbs: &HashMap<String, lmdb::Database>,
    min: usize,
    tower: bool,
) {
    // initialize pair distances
    
    let mut gates = circuit.gates.drain(..).collect::<Vec<_>>();
    let mut dists = vec![0usize; gates.len() + 1];
    let mut lb = 1;
    let mut rb = dists.len() - 1;

    for curr in 0..min {
        println!("Working on curr = {}", curr);
        let mut out_gates = Vec::with_capacity(gates.len());
        let mut out_dists = Vec::with_capacity(gates.len() + 1);
        let mut temp_lb = lb;
        for i in 0..lb {
            out_gates.push(gates[i].clone());
            out_dists.push(dists[i]);
        }

        let mut i = lb;
        while i < gates.len() {
            let left = out_gates.last().unwrap();
            let right = &gates[i];
            let dist = dists[i];

            if dist == curr && i <= rb{
                let (id, id_len) = replace_single_pair(
                    left,
                    right,
                    num_wires,
                    conn,
                    env,
                    bit_shuf_list,
                    dbs,
                    tower,
                );

                if id_len > 0 {
                    // remove left gate
                    out_gates.pop();
                    let left_dist = out_dists.last().unwrap() + 1;
                    let right_dist = dists[i+1] + 1;
                    // emit replacement
                    for j in 0..id_len {
                        out_gates.push(id[j].clone());
                        if j != id_len - 1 {
                            let d = (left_dist + j).min(right_dist + id_len - 2 - j);
                            out_dists.push(d);
                        }
                    }
                    if i == lb {
                        while temp_lb + 1 < out_dists.len()
                            && out_dists[temp_lb + 1] == out_dists[temp_lb] + 1
                        {
                            temp_lb += 1;
                        }
                        temp_lb += 1;
                    } 
                    i += 1;
                    continue;
                }
            }

            // no replacement
            out_gates.push(right.clone());
            out_dists.push(dist);
            i += 1;
        }
        lb = temp_lb;
        rb = out_dists.len() - 1;
        while rb > 0
            && out_dists[rb - 1] == out_dists[rb] + 1
        {
            rb -= 1;
        }
        rb -= 1;
        // close tail distance
        out_dists.push(0);

        gates = out_gates;
        shoot_random_gate_gate_ver(&mut gates, 100_000);
        dists = out_dists;
    }
    // println!("{:?}", dists);
    // println!("left = {} right = {}", lb, rb);
    circuit.gates = gates;
}

// Replace triple of gates
// Largely unused as if replacing pairs is effective, replacing triples would largely be the same
pub fn replace_tri(
    circuit: &mut CircuitSeq,
    num_wires: usize,
    conn: &mut Connection,
    env: &lmdb::Environment,
) {
    println!("Starting replace_tri, circuit length: {}", circuit.gates.len());
    // let start = circuit.clone();
    let mut tris: HashMap<GateTri, Vec<usize>> = HashMap::new();
    let gates = circuit.gates.clone();
    let m = gates.len();
    let mut replaced = 0;

    let mut to_replace: Vec<(Vec<[u8;3]>, Vec<[u8;3]>)> = vec![(Vec::new(), Vec::new()); m / 3];
    if m < 3 {
        println!("Circuit too small, returning");
        return;
    }

    // Build taxonomy 
    println!("Building taxonomy triples...");
    let mut i = 0;
    while i + 2 < m {
        let g0 = gates[i];
        let g1 = gates[i + 1];
        let g2 = gates[i + 2];

        let taxonomy = gate_tri_taxonomy(&g0, &g1, &g2);
        tris.entry(taxonomy).or_default().push(i);

        i += 3;
    }

    let num_tris: usize = tris.values().map(|v| v.len()).sum();
    println!("Triples collected: {}", num_tris);

    let mut rng = rand::rng();
    let mut fail = 0;

    while !tris.is_empty() && fail < 100 {
        let n = rng.random_range(5..=7);
        let mut id = match random_canonical_id(&env, conn, n) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let mut replaced_here = false;

        // Forward 
        for i in 0..id.gates.len().saturating_sub(2) {
            let tax = gate_tri_taxonomy(
                &id.gates[i],
                &id.gates[i + 1],
                &id.gates[i + 2],
            );

            if let Some(v) = tris.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove triple and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());

                    // after the triple
                    new_circuit.extend_from_slice(&id.gates[i + 3..]);

                    // before the triple, reversed
                    new_circuit.extend(id.gates[0..i].iter());

                    to_replace[chosen / 3] = (new_circuit, vec![id.gates[i], id.gates[i+1], id.gates[i+2]]);

                    if v.is_empty() {
                        tris.remove(&tax);
                    }

                    replaced_here = true;
                    break;
                }
            }
        }

        if replaced_here {
            continue;
        }

        // Reverse
        id.gates.reverse();
        for i in 0..id.gates.len().saturating_sub(2) {
            let tax = gate_tri_taxonomy(
                &id.gates[i],
                &id.gates[i + 1],
                &id.gates[i + 2],
            );

            if let Some(v) = tris.get_mut(&tax) {
                if !v.is_empty() {
                    let idx = fastrand::usize(..v.len());
                    let chosen = v.swap_remove(idx);

                    // Remove triple and reconstruct
                    let mut new_circuit = Vec::with_capacity(id.gates.len());

                    // after the triple
                    new_circuit.extend_from_slice(&id.gates[i + 3..]);

                    // before the triple, reversed
                    new_circuit.extend(id.gates[0..i].iter());

                    to_replace[chosen / 3] = (new_circuit, vec![id.gates[i], id.gates[i+1], id.gates[i+2]]);

                    if v.is_empty() {
                        tris.remove(&tax);
                    }

                    replaced_here = true;
                    break;
                }
            }
        }

        if !replaced_here {
            fail += 1;
        }
    }

    // Apply replacements
    println!("Applying triple replacements...");
    for (i, replacement) in to_replace.into_iter().enumerate().rev() {
        if replacement.0.is_empty() {
            continue;
        }

        replaced += 1;
        let index = 3 * i;

        let g0 = circuit.gates[index];
        let g1 = circuit.gates[index + 1];
        let g2 = circuit.gates[index + 2];

        let replacement_circ = CircuitSeq { gates: replacement.0 };

        let mut used_wires =
            vec![(num_wires + 1) as u8; max(replacement_circ.max_wire(), CircuitSeq { gates: replacement.1.clone() }.max_wire()) + 1];


        used_wires[replacement.1[0][0] as usize] = g0[0];
        used_wires[replacement.1[0][1] as usize] = g0[1];
        used_wires[replacement.1[0][2] as usize] = g0[2];

        let tax = gate_tri_taxonomy(&g0, &g1, &g2);
        // Assign new wires if OnNew
        let mut i = 0;
        for collision in &[tax.first.a, tax.first.c1, tax.first.c2] {
            if *collision == CollisionType::OnNew {
                used_wires[replacement.1[1][i] as usize] = g1[i]
            }
            i += 1;
        }

        let mut i = 0;
        for collision in &[(tax.second.a == CollisionType::OnNew) && (tax.gap.a == CollisionType::OnNew), (tax.second.c1 == CollisionType::OnNew) && (tax.gap.c1 == CollisionType::OnNew), (tax.second.c2 == CollisionType::OnNew) && (tax.gap.c2 == CollisionType::OnNew)] {
            if *collision == true {
                used_wires[replacement.1[2][i] as usize] = g2[i]
            }
            i += 1;
        }

        // Fill any remaining placeholders
        for i in 0..used_wires.len() {
            if used_wires[i] == (num_wires + 1) as u8 {
                loop {
                    let wire = rng.random_range(0..=(num_wires-1) as u8);
                    if used_wires.contains(&wire) {
                        continue
                    }
                    used_wires[i] = wire;
                    break
                }
            }
        }

        circuit.gates.splice(
            index..=index + 2,
            CircuitSeq::unrewire_subcircuit(&replacement_circ, &used_wires)
                .gates
                .into_iter()
                .rev(),
        );
    }
    // if start.probably_equal(&circuit, num_wires, 10000).is_err() {
    //     panic!("replace tris changed something");
    // }
    println!("Replaced {}/{} triples", replaced, num_tris);
    println!("Finished replace_tri");
}

// Used in the interleave method
// Create a circuit on n..2n wires and then interleave them
pub fn interleave(circuit: &CircuitSeq, n: usize) -> CircuitSeq {
    let m = circuit.gates.len();
    let mut random = random_circuit(n as u8, m);
    let mut gates = Vec::new();
    for gate in random.gates.iter_mut() {
        for pin in gate.iter_mut() {
            *pin += n as u8;
        }
    }
    for i in 0..m {
        gates.push(circuit.gates[i]);
        gates.push(random.gates[i]);
    }

    CircuitSeq{ gates }
}
