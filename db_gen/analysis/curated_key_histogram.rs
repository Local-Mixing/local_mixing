//! 2-D (gate count x wire count) histogram over one key's candidate circuits.
//!
//! Built to size a selective cap on the single-gate class
//! `66ca88066ac7c7a878442082635feaeb`, whose 512 M candidates are 32% of the
//! whole curated corpus and make its frozen bucket unreadable. The useful
//! replacements for a one-gate function are the short, narrow ones; this shows
//! exactly how many candidates sit at each (gates, wires) point so a cap can be
//! chosen against real counts instead of a guess.
//!
//! ```text
//! curated_key_histogram COMPOSITE_ROCKS KEY_HEX
//! ```
//!
//! Read-only. `wires` is the number of DISTINCT wire indices the circuit
//! touches; `max-wire` is the largest index, which differs when a circuit is
//! not densely labelled.

use local_mixing::db_generation::curated_full::{FUNCTION_KEY_BYTES, split_composite_key};
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::error::Error;

type AnyResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const MAX_GATES: usize = 48;
const MAX_WIRES: usize = 48;

fn parse_key(text: &str) -> AnyResult<[u8; FUNCTION_KEY_BYTES]> {
    if text.len() != FUNCTION_KEY_BYTES * 2 {
        return Err(format!("key must be {} hex chars", FUNCTION_KEY_BYTES * 2).into());
    }
    let mut key = [0u8; FUNCTION_KEY_BYTES];
    for (i, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)?;
    }
    Ok(key)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(key_hex)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: curated_key_histogram COMPOSITE_ROCKS KEY_HEX");
        std::process::exit(2);
    };
    let key = parse_key(key_hex)?;

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_compression_type(rocksdb::DBCompressionType::Zstd);
    options.set_bottommost_compression_type(rocksdb::DBCompressionType::Zstd);
    let database = DB::open_for_read_only(&options, path, false)?;

    // hist[gates][distinct wires]
    let mut hist = vec![vec![0u64; MAX_WIRES + 1]; MAX_GATES + 1];
    let mut max_wire_hist = vec![0u64; 256];
    let mut total = 0u64;
    let mut max_gates_seen = 0usize;
    let mut max_wires_seen = 0usize;

    for item in database.iterator(IteratorMode::From(&key, Direction::Forward)) {
        let (record, _) = item?;
        if record.len() < FUNCTION_KEY_BYTES || record[..FUNCTION_KEY_BYTES] != key[..] {
            break;
        }
        let (_, blob) = split_composite_key(&record)?;
        let gates = blob.len() / 3;
        let mut bitset = [0u64; 4];
        let mut max_wire = 0u8;
        for &wire in blob {
            bitset[(wire >> 6) as usize] |= 1u64 << (wire & 63);
            max_wire = max_wire.max(wire);
        }
        let wires: usize = bitset.iter().map(|w| w.count_ones() as usize).sum();
        if gates <= MAX_GATES && wires <= MAX_WIRES {
            hist[gates][wires] += 1;
        }
        max_wire_hist[max_wire as usize] += 1;
        max_gates_seen = max_gates_seen.max(gates);
        max_wires_seen = max_wires_seen.max(wires);
        total += 1;
        if total % 50_000_000 == 0 {
            eprintln!("[hist] scanned {total}");
        }
    }

    println!("key={key_hex}");
    println!("candidates={total}");
    println!("max-gates={max_gates_seen} max-distinct-wires={max_wires_seen}");

    let g_hi = max_gates_seen.min(MAX_GATES);
    let w_hi = max_wires_seen.min(MAX_WIRES);

    println!("\n=== 2-D histogram: rows = gates, cols = distinct wires ===");
    print!("{:>6}", "g\\w");
    for w in 0..=w_hi {
        print!("{w:>14}");
    }
    println!("{:>16}", "row total");
    for g in 0..=g_hi {
        let row_total: u64 = hist[g][..=w_hi].iter().sum();
        if row_total == 0 {
            continue;
        }
        print!("{g:>6}");
        for w in 0..=w_hi {
            if hist[g][w] == 0 {
                print!("{:>14}", ".");
            } else {
                print!("{:>14}", hist[g][w]);
            }
        }
        println!("{row_total:>16}");
    }
    print!("{:>6}", "tot");
    for w in 0..=w_hi {
        let col: u64 = (0..=g_hi).map(|g| hist[g][w]).sum();
        print!("{col:>14}");
    }
    println!("{total:>16}");

    println!("\n=== marginal by gates (and cumulative) ===");
    let mut cumulative = 0u64;
    for g in 0..=g_hi {
        let row: u64 = hist[g][..=w_hi].iter().sum();
        if row == 0 {
            continue;
        }
        cumulative += row;
        println!(
            "  gates<={g:<3} count={row:>13}  cumulative={cumulative:>13} ({:.4}%)",
            cumulative as f64 * 100.0 / total.max(1) as f64
        );
    }

    println!("\n=== marginal by distinct wires (and cumulative) ===");
    let mut cumulative = 0u64;
    for w in 0..=w_hi {
        let col: u64 = (0..=g_hi).map(|g| hist[g][w]).sum();
        if col == 0 {
            continue;
        }
        cumulative += col;
        println!(
            "  wires<={w:<3} count={col:>13}  cumulative={cumulative:>13} ({:.4}%)",
            cumulative as f64 * 100.0 / total.max(1) as f64
        );
    }

    // The actionable table: how many candidates survive a joint cap.
    println!("\n=== candidates kept by joint cap (gates <= G AND wires <= W) ===");
    print!("{:>6}", "G\\W");
    for w in 0..=w_hi {
        print!("{w:>14}");
    }
    println!();
    let mut prefix = vec![vec![0u64; w_hi + 2]; g_hi + 2];
    for g in 0..=g_hi {
        for w in 0..=w_hi {
            prefix[g + 1][w + 1] = hist[g][w] + prefix[g][w + 1] + prefix[g + 1][w] - prefix[g][w];
        }
    }
    for g in 0..=g_hi {
        if hist[g][..=w_hi].iter().sum::<u64>() == 0 && prefix[g + 1][w_hi + 1] == 0 {
            continue;
        }
        print!("{g:>6}");
        for w in 0..=w_hi {
            print!("{:>14}", prefix[g + 1][w + 1]);
        }
        println!();
    }

    println!("\n=== max wire index distribution (non-zero) ===");
    for (index, count) in max_wire_hist.iter().enumerate() {
        if *count > 0 {
            println!("  max-wire={index:<4} {count:>13}");
        }
    }
    Ok(())
}
