use std::fs;

use primitive_types::U256 as u256;
use rand::RngCore;

use local_mixing::circuit::{CircuitSeq, U1024};

pub fn run(sub: &clap::ArgMatches) {
    let s: &str = sub.get_one::<String>("source").unwrap().as_str();
    let n: usize = *sub.get_one("n").unwrap();
    let data = fs::read_to_string(s).expect("Failed to read circuit file");
    let circuit = CircuitSeq::from_string(&data);

    let use_1024 = n > 256;

    if use_1024 {
        assert!(n <= 1024, "evaluate supports up to 1024 wires");
        let mask = if n < 1024 {
            (U1024::one() << n) - U1024::one()
        } else {
            U1024::MAX
        };

        let input: U1024 = if sub.get_flag("random") {
            let mut bytes = [0u8; 128];
            rand::rng().fill_bytes(&mut bytes);
            U1024::from_little_endian(&bytes) & mask
        } else {
            let raw = sub
                .get_one::<String>("input")
                .expect("-x required when not using -r");
            parse_u1024(raw)
        };

        println!("n: {}", n);
        println!("Input:  {}", format_bits_1024(input, n));
        let output = circuit.evaluate_1024(input) & mask;
        println!("Output: {}", format_bits_1024(output, n));
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

fn parse_u1024(s: &str) -> U1024 {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        U1024::from_str_radix(hex, 16).expect("Invalid hex value")
    } else {
        U1024::from_dec_str(s).expect("Invalid decimal value")
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

fn format_bits_1024(val: U1024, n: usize) -> String {
    let bits: String = (0..n)
        .map(|i| {
            if (val >> i) & U1024::one() == U1024::one() {
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
