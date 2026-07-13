//! Build the 256 per-shard BinaryFuse8 filters from a frozen table and
//! persist them to <dir>/filters.bin, OR load them back and verify.
//!
//! usage: frozen_filter_persist build <frozen_dir>
//!        frozen_filter_persist check <frozen_dir> <probes.bin>
//!
//! filters.bin: bincode of FiltersFile { table_entry_count, filters: Vec<BinaryFuse8> }

use bincode2::{Decode, Encode};
use rayon::prelude::*;
use std::os::unix::fs::FileExt;
use std::time::Instant;
use xorf::{BinaryFuse8, Filter};

const BUCKETS: usize = 1 << 20;

#[derive(Encode, Decode)]
#[bincode(crate = "bincode2")]
struct FiltersFile {
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
#[inline]
fn mix76(shard: usize, bucket: u32, tail: u64) -> u64 {
    splitmix64(tail ^ ((bucket as u64) << 44) ^ ((shard as u64) << 30))
}
#[inline]
fn split_key(k: &[u8]) -> (usize, u32, u64) {
    let hi = u64::from_be_bytes(k[0..8].try_into().unwrap());
    let lo = u64::from_be_bytes(k[8..15+1].try_into().unwrap());
    ((hi >> 56) as usize, ((hi >> 36) & 0xF_FFFF) as u32, ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52))
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}
impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        BitReader { buf, pos: 0, acc: 0, nbits: 0 }
    }
    #[inline]
    fn get(&mut self, bits: u32) -> u64 {
        while self.nbits < bits {
            let b = if self.pos < self.buf.len() { self.buf[self.pos] } else { 0 };
            self.acc |= (b as u64) << self.nbits;
            self.pos += 1;
            self.nbits += 8;
        }
        let v = self.acc & ((1u64 << bits) - 1);
        self.acc >>= bits;
        self.nbits -= bits;
        v
    }
    #[inline]
    fn get1(&mut self) -> u64 {
        self.get(1)
    }
}

fn scan_shard_keys(dir: &str, shard: usize) -> (Vec<u64>, u64) {
    let head_len = (24usize + (BUCKETS + 1) * 5) as u64;
    let file = std::fs::File::open(format!("{dir}/shard_{shard:02x}.frz")).unwrap();
    let mut head = vec![0u8; head_len as usize];
    file.read_exact_at(&mut head, 0).unwrap();
    assert_eq!(&head[0..8], b"FRZTBL01");
    let count = u64::from_le_bytes(head[8..16].try_into().unwrap());
    let mut offs = Vec::with_capacity(BUCKETS + 1);
    for i in 0..=BUCKETS {
        let mut b = [0u8; 8];
        b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
        offs.push(u64::from_le_bytes(b));
    }
    let data_len = offs[BUCKETS];
    let mut keys: Vec<u64> = Vec::with_capacity(count as usize);
    const CHUNK: u64 = 32 << 20;
    let (mut chunk_start, mut chunk_end) = (0u64, 0u64);
    let mut buf: Vec<u8> = Vec::new();
    for b in 0..BUCKETS {
        let (o0, o1) = (offs[b], offs[b + 1]);
        if o0 == o1 {
            continue;
        }
        if o1 > chunk_end {
            chunk_start = o0;
            chunk_end = (o0 + CHUNK.max(o1 - o0)).min(data_len);
            buf.resize((chunk_end - chunk_start) as usize, 0);
            file.read_exact_at(&mut buf, head_len + chunk_start).unwrap();
        }
        let slice = &buf[(o0 - chunk_start) as usize..(o1 - chunk_start) as usize];
        let mut r = BitReader::new(slice);
        let n = r.get(16) as usize;
        let l = r.get(6) as u32;
        let mut uppers = Vec::with_capacity(n);
        let mut up = 0u64;
        for _ in 0..n {
            while r.get1() == 0 {
                up += 1;
            }
            uppers.push(up);
        }
        for i in 0..n {
            let low = if l > 0 { r.get(l) } else { 0 };
            keys.push(mix76(shard, b as u32, (uppers[i] << l) | low));
        }
    }
    (keys, count)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("build") => {
            let dir = args[2].clone();
            let t0 = Instant::now();
            let mut filters: Vec<Option<BinaryFuse8>> = (0..256).map(|_| None).collect();
            let mut total = 0u64;
            for batch in 0..8 {
                let built: Vec<(usize, BinaryFuse8, u64)> = (batch * 32..(batch + 1) * 32)
                    .into_par_iter()
                    .map(|s| {
                        let (mut keys, cnt) = scan_shard_keys(&dir, s);
                        keys.sort_unstable();
                        keys.dedup();
                        (s, BinaryFuse8::try_from(&keys).expect("fuse build"), cnt)
                    })
                    .collect();
                for (s, f, c) in built {
                    filters[s] = Some(f);
                    total += c;
                }
                println!("batch {}/8 at {:.0}s", batch + 1, t0.elapsed().as_secs_f64());
            }
            let ff = FiltersFile {
                table_entry_count: total,
                filters: filters.into_iter().map(|f| f.unwrap()).collect(),
            };
            let f = std::fs::File::create(format!("{dir}/filters.bin")).unwrap();
            let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
            let nbytes =
                bincode2::encode_into_std_write(&ff, &mut w, bincode2::config::standard())
                    .expect("serialize");
            use std::io::Write;
            w.flush().unwrap();
            println!(
                "filters.bin written: {} bytes ({:.1} GB) for {} entries in {:.0}s",
                nbytes,
                nbytes as f64 / 1e9,
                total,
                t0.elapsed().as_secs_f64()
            );
        }
        Some("check") => {
            let dir = args[2].clone();
            let probes_path = args[3].clone();
            let t0 = Instant::now();
            let path = format!("{dir}/filters.bin");
            let fsize = std::fs::metadata(&path).unwrap().len();
            let f = std::fs::File::open(&path).unwrap();
            let mut r = std::io::BufReader::with_capacity(8 << 20, f);
            let ff: FiltersFile =
                bincode2::decode_from_std_read(&mut r, bincode2::config::standard())
                    .expect("deserialize");
            println!(
                "[load] filters.bin {:.1} GB loaded+parsed in {:.1}s (covers {} entries)",
                fsize as f64 / 1e9,
                t0.elapsed().as_secs_f64(),
                ff.table_entry_count
            );
            // verify: every probe key must be present; random keys FP rate
            let pdata = std::fs::read(&probes_path).expect("probes");
            let n = u32::from_le_bytes(pdata[0..4].try_into().unwrap()) as usize;
            let mut pos = 4usize;
            let mut fneg = 0u64;
            let mut t_ns: Vec<u64> = Vec::with_capacity(n);
            for _ in 0..n {
                let k = &pdata[pos..pos + 16];
                let vl = u16::from_le_bytes(pdata[pos + 16..pos + 18].try_into().unwrap()) as usize;
                pos += 18 + vl;
                let (s, b, tl) = split_key(k);
                let m = mix76(s, b, tl);
                let t = Instant::now();
                if !ff.filters[s].contains(&m) {
                    fneg += 1;
                }
                t_ns.push(t.elapsed().as_nanos() as u64);
            }
            use rand::RngCore;
            let mut rng = rand::rng();
            let mut fpos = 0u64;
            for _ in 0..n {
                let mut k = [0u8; 16];
                rng.fill_bytes(&mut k);
                let (s, b, tl) = split_key(&k);
                if ff.filters[s].contains(&mix76(s, b, tl)) {
                    fpos += 1;
                }
            }
            t_ns.sort();
            println!(
                "[verify] false_neg={} (MUST be 0), false_pos={}/{} ({:.3}%), check p50={}ns p99={}ns",
                fneg,
                fpos,
                n,
                100.0 * fpos as f64 / n as f64,
                t_ns[t_ns.len() / 2],
                t_ns[t_ns.len() * 99 / 100]
            );
            println!("CHECK {}", if fneg == 0 { "PASS" } else { "FAIL" });
        }
        _ => {
            eprintln!("usage: frozen_filter_persist build|check <frozen_dir> [probes.bin]");
            std::process::exit(2);
        }
    }
}
