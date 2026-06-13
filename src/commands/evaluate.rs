use std::fs;

use primitive_types::{U256 as u256, U512 as u512};
use rand::RngCore;

use local_mixing::circuit::CircuitSeq;

pub fn run(sub: &clap::ArgMatches) {
    let s: &str = sub.get_one::<String>("source").unwrap().as_str();
    let n: usize = *sub.get_one("n").unwrap();
    let data = fs::read_to_string(s).expect("Failed to read circuit file");
    let circuit = CircuitSeq::from_string(&data);

    let use_512 = n > 256;

    if use_512 {
        let mask = if n < 512 {
            (u512::one() << n) - u512::one()
        } else {
            u512::MAX
        };

        let input: u512 = if sub.get_flag("random") {
            let mut bytes = [0u8; 64];
            rand::rng().fill_bytes(&mut bytes);
            u512::from_little_endian(&bytes) & mask
        } else {
            let raw = sub
                .get_one::<String>("input")
                .expect("-x required when not using -r");
            parse_u512(raw)
        };

        println!("n: {}", n);
        println!("Input:  {}", format_bits_512(input, n));
        let output = circuit.gates.iter().fold(input, |state, &gate| {
            let one = u512::one();
            let c1 = (state >> gate[1]) & one;
            let c2 = (state >> gate[2]) & one;
            state ^ ((c1 | (one ^ c2)) << gate[0])
        });
        println!("Output: {}", format_bits_512(output & mask, n));
    } else {
        let mask = if n < 256 {
            (u256::one() << n) - u256::one()
        } else {
            u256::MAX
        };

        let input: u256 = if sub.get_flag("random") {
            let mut bytes = [0u8; 32];
            rand::rng().fill_bytes(&mut bytes);
            u256::from_little_endian(&bytes) & mask
        } else {
            let raw = sub
                .get_one::<String>("input")
                .expect("-x required when not using -r");
            parse_u256(raw)
        };

        println!("n: {}", n);
        println!("Input:  {}", format_bits_256(input, n));
        let output = circuit.evaluate_256(input) & mask;
        println!("Output: {}", format_bits_256(output, n));
    }
}

fn parse_u256(s: &str) -> u256 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u256::from_str_radix(hex, 16).expect("Invalid hex value")
    } else {
        u256::from_dec_str(s).expect("Invalid decimal value")
    }
}

fn parse_u512(s: &str) -> u512 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u512::from_str_radix(hex, 16).expect("Invalid hex value")
    } else {
        // primitive_types::U512 has no from_dec_str; parse via big-endian hex workaround
        panic!("For n>256 please supply a hex value (0x...)");
    }
}

fn format_bits_256(val: u256, n: usize) -> String {
    let bits: String = (0..n)
        .map(|i| {
            if (val >> i) & u256::one() == u256::one() {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    let hex_bytes = val.to_little_endian();
    let needed = (n + 7) / 8;
    let hex: String = hex_bytes[..needed]
        .iter()
        .rev()
        .map(|b| format!("{:02x}", b))
        .collect();
    format!("{} (0x{})", bits, hex)
}

fn format_bits_512(val: u512, n: usize) -> String {
    let bits: String = (0..n)
        .map(|i| {
            if (val >> i) & u512::one() == u512::one() {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    let hex_bytes = val.to_little_endian();
    let needed = (n + 7) / 8;
    let hex: String = hex_bytes[..needed]
        .iter()
        .rev()
        .map(|b| format!("{:02x}", b))
        .collect();
    format!("{} (0x{})", bits, hex)
}
