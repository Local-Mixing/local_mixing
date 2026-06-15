use local_mixing::circuit::{CircuitSeq, Gate};
use primitive_types::U512;
use rand::RngCore;
use std::{env, fs::File, io::Write};

#[derive(Clone, Copy)]
struct Info {
    gate: [u16; 3],
    b_const: Option<bool>,
    c_const: Option<bool>,
    cond_const: Option<bool>,
}

fn usage(program: &str) -> ! {
    eprintln!("usage: {program} <circuit> <out-slice.txt> [verify-samples=64]");
    std::process::exit(2);
}

fn cond_const(b: Option<bool>, c: Option<bool>) -> Option<bool> {
    match (b, c) {
        (Some(true), _) => Some(true),
        (_, Some(false)) => Some(true),
        (Some(false), Some(true)) => Some(false),
        (Some(false), None) => None,
        (None, Some(true)) => None,
        (None, None) => None,
    }
}

fn condition_needs_b(b: Option<bool>, c: Option<bool>) -> bool {
    if b == Some(true) || c == Some(false) {
        return false;
    }
    b.is_none()
}

fn condition_needs_c(b: Option<bool>, c: Option<bool>) -> bool {
    if b == Some(true) || c == Some(false) {
        return false;
    }
    c.is_none()
}

fn random_u128() -> U512 {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    U512::from_little_endian(&bytes)
}

fn middle(output: U512) -> U512 {
    (output >> 128) & ((U512::one() << 128) - U512::one())
}

fn eval_with_fixed_yz(circuit: &CircuitSeq, x: U512) -> U512 {
    Gate::evaluate_index_list_512(x, &circuit.gates)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
    }
    let samples = args
        .get(3)
        .map_or(Ok(64usize), |s| s.parse())
        .unwrap_or_else(|_| usage(&args[0]));
    let text = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| panic!("failed to read {}: {e}", args[1]));
    let circuit = CircuitSeq::from_string(&text);
    let wires = circuit
        .gates
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0usize, |w| w as usize + 1);
    assert!(wires <= 512, "only supports up to 512 wires");
    assert!(wires >= 256, "expected at least 256 wires");

    let mut constants = vec![None; wires];
    for c in constants.iter_mut().take(wires).skip(128) {
        *c = Some(false);
    }

    let mut infos = Vec::with_capacity(circuit.gates.len());
    for &gate @ [a, b, c] in &circuit.gates {
        let ac = constants[a as usize];
        let bc = constants[b as usize];
        let cc = constants[c as usize];
        let fc = cond_const(bc, cc);
        infos.push(Info {
            gate,
            b_const: bc,
            c_const: cc,
            cond_const: fc,
        });
        constants[a as usize] = match (ac, fc) {
            (Some(av), Some(fv)) => Some(av ^ fv),
            (old, Some(false)) => old,
            _ => None,
        };
    }

    let mut needed = vec![false; wires];
    for w in 128..256.min(wires) {
        needed[w] = true;
    }
    let mut keep = vec![false; infos.len()];
    for (idx, info) in infos.iter().enumerate().rev() {
        let [a, b, c] = info.gate;
        if !needed[a as usize] {
            continue;
        }
        if info.cond_const != Some(false) {
            keep[idx] = true;
            if condition_needs_b(info.b_const, info.c_const) {
                needed[b as usize] = true;
            }
            if condition_needs_c(info.b_const, info.c_const) {
                needed[c as usize] = true;
            }
        }
        needed[a as usize] = true;
    }

    let sliced = CircuitSeq {
        gates: circuit
            .gates
            .iter()
            .zip(&keep)
            .filter_map(|(&gate, &keep)| keep.then_some(gate))
            .collect(),
    };

    for i in 0..samples {
        let x = random_u128();
        let full = middle(eval_with_fixed_yz(&circuit, x));
        let part = middle(eval_with_fixed_yz(&sliced, x));
        if full != part {
            panic!(
                "slice mismatch at sample {i}: x=0x{:032x} full=0x{:032x} sliced=0x{:032x}",
                x.low_u128(),
                full.low_u128(),
                part.low_u128()
            );
        }
    }

    let mut out = File::create(&args[2]).unwrap_or_else(|e| panic!("failed to create {}: {e}", args[2]));
    write!(out, "{}", sliced.repr()).expect("failed to write slice");
    println!("input_gates {}", circuit.gates.len());
    println!("slice_gates {}", sliced.gates.len());
    println!("needed_initial_wires {}", needed.iter().filter(|&&x| x).count());
    println!("verified_samples {}", samples);
    println!("path {}", args[2]);
}
