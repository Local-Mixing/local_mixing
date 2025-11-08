use crate::{
    circuit::circuit::CircuitSeq,
    replace::replace::{compress, compress_big, obfuscate, outward_compress, random_id, expand_big},
};
use crate::random::random_data::shoot_random_gate;
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub fn instruction_butterfly(
    c: &CircuitSeq,
    conn: &mut Connection,
    n: usize,
) -> CircuitSeq {
    // Pick one random R
    let mut rng = rand::rng();
    let (r, r_inv) = random_id(n as u8, rng.random_range(100..=200)); 

    println!("Butterfly start: {} gates", c.gates.len());

    let r = &r;           // reference is enough; read-only
    let r_inv = &r_inv;   // same

    // Parallel processing of gates
    let blocks: Vec<CircuitSeq> = c.gates
        .par_iter()
        .enumerate()
        .map(|(i, &g)| {
            // wrap single gate as CircuitSeq
            let mut gi = r_inv.concat(&CircuitSeq { gates: vec![g] }).concat(&r);
            // create a read-only connection per thread
            let mut conn = Connection::open_with_flags(
            "circuits.db",
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).expect("Failed to open read-only connection");
        //shoot_random_gate(&mut gi, 100_000);
        // compress the block
        let compressed_block = compress_big(&expand_big(&gi, 10, n, &mut conn), 100, n, &mut conn);
        let before_len = r_inv.gates.len() * 2 + 1;
        let after_len = compressed_block.gates.len();
            
        let color_line = if after_len < before_len {
            "\x1b[32m──────────────\x1b[0m" // green
            } else if after_len > before_len {
                "\x1b[31m──────────────\x1b[0m" // red
            } else {
                "\x1b[90m──────────────\x1b[0m" // gray
        };

        println!(
            "  Block {}: before {} gates → after {} gates  {}",
            i, before_len, after_len, color_line
        );

        println!("  {}", compressed_block.repr());

        compressed_block
    })
    .collect();

    let mut acc = r;
    for block in blocks {
        acc = acc.concat(&block);
    }
    acc = acc.concat(&r_inv);
    acc
}