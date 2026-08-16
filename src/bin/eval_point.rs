//! Evaluate a circuit on ONE explicit input state and print the output state.
//! Complements check_output/verify_zero_slice (random self-checks only):
//! this is the tool for producing a concrete test vector y = C(x).
//!
//! Usage: eval_point <circuit> <g57|mpmct1> <hex_state> [reps=1]
//!
//! <hex_state> is the input state as a big-endian hex number with wire i =
//! bit i (so "x on wires 0..64, all else zero" is just the 16-digit hex of
//! x). Prints the output state on stdout as zero-padded hex covering every
//! circuit wire; parse/eval timing goes to stderr. reps>1 repeats the
//! evaluation for timing.

use local_mixing::circuit::circuit::U1024;
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{eval_u1024, max_wire};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: eval_point <circuit> <g57|mpmct1> <hex_state> [reps=1]");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let (g, wires) = match a[2].as_str() {
        "g57" => {
            let g = read_g57_file(&a[1]).expect("read g57");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        "mpmct1" => {
            // The crossing stage writes a bogus "mpmct1 1 N" header, so never
            // trust the header's wire count below what the gates themselves use.
            let (g, w) = read_mpmct(&a[1]).expect("read mpmct1");
            let w = w.max(max_wire(&g) as usize + 1);
            (g, w)
        }
        m => {
            eprintln!("unknown format {m} (want g57|mpmct1)");
            std::process::exit(2);
        }
    };
    let parse = t0.elapsed();

    let hexs = a[3].trim_start_matches("0x");
    let hs = if hexs.len() % 2 == 1 {
        format!("0{hexs}")
    } else {
        hexs.to_string()
    };
    let bytes: Vec<u8> = (0..hs.len() / 2)
        .map(|i| u8::from_str_radix(&hs[2 * i..2 * i + 2], 16).expect("hex input"))
        .collect();
    assert!(bytes.len() <= 128, "input wider than 1024 bits");
    let mut be = [0u8; 128];
    be[128 - bytes.len()..].copy_from_slice(&bytes);
    let input = U1024::from_big_endian(&be);
    assert!(
        wires >= 1024 || (input >> wires).is_zero(),
        "input has bits above wire {wires}"
    );

    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut out = input;
    let te = Instant::now();
    for _ in 0..reps.max(1) {
        out = eval_u1024(&g, input);
    }
    let eval = te.elapsed();

    let ob = out.to_big_endian();
    let digits = wires.div_ceil(4);
    let hex: String = ob.iter().map(|b| format!("{b:02x}")).collect();
    println!("{}", &hex[hex.len() - digits..]);
    eprintln!(
        "[eval_point] {} gates, {} wires; parse {:.2?}; {} eval(s) in {:.2?} ({:.1} ms/eval)",
        g.len(),
        wires,
        parse,
        reps.max(1),
        eval,
        eval.as_secs_f64() * 1e3 / reps.max(1) as f64
    );
}
