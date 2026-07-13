//! Sample real (key, value) pairs from the rocks source into a portable
//! probe pack for benchmarking on machines that don't hold the 700G rocks.
//! Format: [n: u32 LE] then n records of [key: 16B][vlen: u16 LE][value].
//!
//! usage: make_probes <rocks> <out_file> [n]

use rand::RngCore;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rocks = DB::open_for_read_only(&Options::default(), args[1].as_str(), false).unwrap();
    let out_path = &args[2];
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000);

    let mut rng = rand::rng();
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    out.write_all(&(n as u32).to_le_bytes()).unwrap();
    let mut written = 0usize;
    while written < n {
        let mut probe = [0u8; 16];
        rng.fill_bytes(&mut probe);
        let mut it = rocks.iterator(IteratorMode::From(&probe, Direction::Forward));
        if let Some(Ok((k, v))) = it.next() {
            if k.len() != 16 || v.len() > u16::MAX as usize {
                continue;
            }
            out.write_all(&k).unwrap();
            out.write_all(&(v.len() as u16).to_le_bytes()).unwrap();
            out.write_all(&v).unwrap();
            written += 1;
        }
    }
    out.flush().unwrap();
    println!("wrote {} probes to {}", written, out_path);
}
