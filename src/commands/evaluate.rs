use std::fs;

use primitive_types::U256 as u256;
use rand::RngCore;

use local_mixing::circuit::xgate::{XGate, eval_limbs, max_wire};
use local_mixing::circuit::{CircuitSeq, Gate, U1024};
use local_mixing::engine::format;

pub fn run(sub: &clap::ArgMatches) {
    let s: &str = sub.get_one::<String>("source").unwrap().as_str();
    let n: usize = *sub.get_one("n").unwrap();
    assert!(n <= 1024, "evaluate supports up to 1024 wires");

    // The source may be either representation. mpmct1 announces itself in the
    // header; anything else is the base-83 g57 encoding. Reading bytes rather
    // than a String skips a UTF-8 validation pass over the whole file, which
    // is real money on a multi-megabyte circuit.
    let raw = fs::read(s).expect("Failed to read circuit file");
    let source = if raw.starts_with(b"mpmct1") {
        let (g, _wires) = format::read_mpmct(s).expect("Failed to read mpmct1 circuit");
        Source::Mpmct(g)
    } else {
        Source::G57(CircuitSeq::from_bytes(&raw))
    };

    // Evaluation must cover every wire the circuit TOUCHES, not just `n`: a
    // gate above the reported width would otherwise be read out of range.
    let touched = match &source {
        Source::G57(c) if c.gates.is_empty() => 0,
        Source::G57(c) => c.max_wire() + 1,
        Source::Mpmct(g) if g.is_empty() => 0,
        Source::Mpmct(g) => max_wire(g.iter()) as usize + 1,
    };
    let eval_wires = n.max(touched);
    assert!(
        eval_wires <= 1024,
        "circuit touches wire {} but evaluate supports up to 1024 wires",
        eval_wires - 1
    );

    // Input is drawn/parsed on `n` bits; wires above `n` start at zero.
    let input = read_input(sub, n);

    println!("n: {}", n);
    println!("Input:  {}", format_bits(&input, n));

    // Dispatch to the narrowest kernel that covers every touched wire. The
    // limb array is the state in every case; the fixed-width bignum types are
    // just views onto it.
    let mut state = input;
    match &source {
        Source::G57(c) => eval_g57(c, &mut state, eval_wires),
        Source::Mpmct(g) => eval_limbs(g.iter(), &mut state),
    }

    println!("Output: {}", format_bits(&state, n));
}

/// The circuit as read: base-83 g57, or the general mpmct1 gate set (X gates,
/// CNOTs, complemented g57s and wider mixed-polarity conjunctions).
enum Source {
    G57(CircuitSeq),
    Mpmct(Vec<XGate>),
}

/// State as 16 little-endian limbs (1024 bits), the widest evaluate supports.
type State = [u64; 16];

fn eval_g57(c: &CircuitSeq, state: &mut State, eval_wires: usize) {
    if eval_wires <= 64 {
        state[0] = Gate::evaluate_index_list_64(state[0], &c.gates);
    } else if eval_wires <= 128 {
        let out = Gate::evaluate_index_list_128(
            (state[0] as u128) | ((state[1] as u128) << 64),
            &c.gates,
        );
        state[0] = out as u64;
        state[1] = (out >> 64) as u64;
    } else if eval_wires <= 256 {
        let mut v = u256::zero();
        v.0.copy_from_slice(&state[..4]);
        let out = Gate::evaluate_index_list_256(v, &c.gates);
        state[..4].copy_from_slice(&out.0);
    } else {
        let mut v = U1024::zero();
        v.0.copy_from_slice(&state[..]);
        let out = Gate::evaluate_index_list_1024(v, &c.gates);
        state.copy_from_slice(&out.0);
    }
}

fn read_input(sub: &clap::ArgMatches, n: usize) -> State {
    let mut state = [0u64; 16];
    if sub.get_flag("random") {
        let mut rng = rand::rng();
        for limb in state.iter_mut() {
            *limb = rng.next_u64();
        }
    } else {
        let raw = sub
            .get_one::<String>("input")
            .expect("-x required when not using -r");
        let v = parse_u1024(raw);
        state.copy_from_slice(&v.0);
    }
    mask_to(&mut state, n);
    state
}

/// Zero every bit at or above wire `n`.
fn mask_to(state: &mut State, n: usize) {
    for (i, limb) in state.iter_mut().enumerate() {
        let low = i * 64;
        *limb &= if n >= low + 64 {
            u64::MAX
        } else if n <= low {
            0
        } else {
            (1u64 << (n - low)) - 1
        };
    }
}

fn parse_u1024(s: &str) -> U1024 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        U1024::from_str_radix(hex, 16).expect("Invalid hex value")
    } else {
        U1024::from_dec_str(s).expect("Invalid decimal value")
    }
}

fn format_bits(state: &State, n: usize) -> String {
    let bits: String = (0..n)
        .map(|i| {
            if (state[i >> 6] >> (i & 63)) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    let needed = n.div_ceil(8);
    let hex: String = (0..needed)
        .rev()
        .map(|byte| format!("{:02x}", (state[byte >> 3] >> ((byte & 7) * 8)) as u8))
        .collect();
    format!("{} (0x{})", bits, hex)
}
