//! Frozen-table reader + replay test: correctness vs the staged LMDB and the
//! rocks ground truth, plus latency/throughput benchmarks.
//!
//! usage: frozen_replay <frozen_dir> <lmdb_dir> <rocks> [n_per_kind]

use lmdb::Transaction;
use rand::RngCore;
use rayon::prelude::*;
use rocksdb::{DB, Direction, IteratorMode, Options};
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const BUCKETS: usize = 1 << 20;
const TAIL_BITS: u32 = 48;
const GI_CLAMP: u32 = 11;
const ESC: u32 = u32::MAX;
const MAXLEN: usize = 40;

// ---- bit reader / huffman / tables / value decode: mirrors frozen_table ----
struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}
impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        BitReader {
            buf,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }
    #[inline]
    fn get(&mut self, bits: u32) -> u64 {
        while self.nbits < bits {
            let b = if self.pos < self.buf.len() {
                self.buf[self.pos]
            } else {
                0
            };
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

#[derive(Clone, Default)]
struct HuffTable {
    enc: HashMap<u32, (u64, u32)>,
    syms: Vec<u32>,
    first_code: Vec<u64>,
    first_idx: Vec<usize>,
    count: Vec<usize>,
    single: bool,
}
fn rebuild_canonical(mut sym_lens: Vec<(u32, u32)>) -> HuffTable {
    let mut t = HuffTable::default();
    if sym_lens.is_empty() {
        return t;
    }
    if sym_lens.len() == 1 {
        t.enc.insert(sym_lens[0].0, (0, 0));
        t.syms = vec![sym_lens[0].0];
        t.single = true;
        return t;
    }
    t.first_code = vec![0; MAXLEN + 1];
    t.first_idx = vec![0; MAXLEN + 1];
    t.count = vec![0; MAXLEN + 1];
    sym_lens.sort_by_key(|&(s, l)| (l, s));
    let mut code = 0u64;
    let mut prev_len = sym_lens[0].1;
    let mut started = vec![false; MAXLEN + 1];
    for (idx, &(s, l)) in sym_lens.iter().enumerate() {
        if l > prev_len {
            code <<= l - prev_len;
            prev_len = l;
        }
        let li = l as usize;
        if !started[li] {
            started[li] = true;
            t.first_code[li] = code;
            t.first_idx[li] = idx;
        }
        t.count[li] += 1;
        t.enc.insert(s, (code, l));
        t.syms.push(s);
        code += 1;
    }
    t
}
impl HuffTable {
    #[inline]
    fn decode(&self, r: &mut BitReader) -> u32 {
        if self.single {
            return self.syms[0];
        }
        let mut code = 0u64;
        for len in 1..=MAXLEN {
            code = (code << 1) | r.get1();
            if self.count[len] > 0 {
                let fc = self.first_code[len];
                if code >= fc && code < fc + self.count[len] as u64 {
                    return self.syms[self.first_idx[len] + (code - fc) as usize];
                }
            }
        }
        panic!("bad huffman stream");
    }
}
struct Tables {
    header: HuffTable,
    gates: Vec<HuffTable>,
}
#[inline]
fn ctx_of(w: u32, gi: u32) -> usize {
    (w.min(32) as usize) * 12 + gi.min(GI_CLAMP) as usize
}
fn load_tables(path: &str) -> Tables {
    let data = std::fs::read(path).expect("read tables");
    let mut pos = 0usize;
    fn rd_u32(d: &[u8], p: &mut usize) -> u32 {
        let v = u32::from_le_bytes(d[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    }
    fn load_one(d: &[u8], p: &mut usize) -> HuffTable {
        let n = rd_u32(d, p) as usize;
        let mut sl = Vec::with_capacity(n);
        for _ in 0..n {
            let s = rd_u32(d, p);
            let l = d[*p] as u32;
            *p += 1;
            sl.push((s, l));
        }
        rebuild_canonical(sl)
    }
    let header = load_one(&data, &mut pos);
    let nctx = rd_u32(&data, &mut pos) as usize;
    let mut gates = Vec::with_capacity(nctx);
    for _ in 0..nctx {
        gates.push(load_one(&data, &mut pos));
    }
    Tables { header, gates }
}
fn decode_value(t: &Tables, r: &mut BitReader, out: &mut Vec<u8>) {
    loop {
        let hs = t.header.decode(r);
        let (g, w, chain);
        if hs == ESC {
            let ge = r.get(8) as u32;
            if ge == 121 {
                let len = r.get(16) as usize;
                for _ in 0..len {
                    out.push(r.get(8) as u8);
                }
                return;
            }
            g = ge;
            w = r.get(8) as u32;
            chain = r.get(1) as u32;
        } else {
            g = hs >> 7;
            w = (hs >> 1) & 0x3f;
            chain = hs & 1;
        }
        out.push((g * 3) as u8);
        for gi in 0..g {
            let tab = &t.gates[ctx_of(w, gi)];
            let sym = tab.decode(r);
            if sym == ESC {
                if r.get1() == 1 {
                    for _ in 0..3 {
                        out.push(r.get(8) as u8);
                    }
                } else {
                    let triple = r.get(15) as u32;
                    out.push(((triple >> 10) & 0x1f) as u8);
                    out.push(((triple >> 5) & 0x1f) as u8);
                    out.push((triple & 0x1f) as u8);
                }
            } else {
                out.push(((sym >> 10) & 0x1f) as u8);
                out.push(((sym >> 5) & 0x1f) as u8);
                out.push((sym & 0x1f) as u8);
            }
        }
        if chain == 0 {
            return;
        }
    }
}
#[inline]
fn split_key(k: &[u8]) -> (usize, u32, u64) {
    let hi = u64::from_be_bytes(k[0..8].try_into().unwrap());
    let lo = u64::from_be_bytes(k[8..16].try_into().unwrap());
    (
        (hi >> 56) as usize,
        ((hi >> 36) & 0xF_FFFF) as u32,
        ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52),
    )
}

// ------------------------------------------------------------ reader
struct FrozenShard {
    file: std::fs::File,
    offs: Vec<u64>, // BUCKETS+1 entries
    data_base: u64,
}
struct Frozen {
    shards: Vec<FrozenShard>,
    tables: Tables,
}
impl Frozen {
    fn open(dir: &str) -> Frozen {
        let tables = load_tables(&format!("{dir}/tables.bin"));
        let head_len = 24usize + (BUCKETS + 1) * 5;
        let shards: Vec<FrozenShard> = (0..256usize)
            .into_par_iter()
            .map(|s| {
                let file = std::fs::File::open(format!("{dir}/shard_{s:02x}.frz")).unwrap();
                let mut head = vec![0u8; head_len];
                file.read_exact_at(&mut head, 0).unwrap();
                assert_eq!(&head[0..8], b"FRZTBL01");
                let mut offs = Vec::with_capacity(BUCKETS + 1);
                for i in 0..=BUCKETS {
                    let mut b = [0u8; 8];
                    b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
                    offs.push(u64::from_le_bytes(b));
                }
                FrozenShard {
                    file,
                    offs,
                    data_base: head_len as u64,
                }
            })
            .collect();
        Frozen { shards, tables }
    }

    fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let (shard, bucket, tail) = split_key(key);
        let sh = &self.shards[shard];
        let o0 = sh.offs[bucket as usize];
        let o1 = sh.offs[bucket as usize + 1];
        if o0 == o1 {
            return None;
        }
        let mut buf = vec![0u8; (o1 - o0) as usize];
        sh.file.read_exact_at(&mut buf, sh.data_base + o0).ok()?;
        let mut r = BitReader::new(&buf);
        let n = r.get(16) as usize;
        let l = r.get(6) as u32;
        // EF decode uppers, then lowers; find first index with matching tail
        let mut uppers = Vec::with_capacity(n);
        let mut up = 0u64;
        for _ in 0..n {
            while r.get1() == 0 {
                up += 1;
            }
            uppers.push(up);
        }
        let mut idx: Option<usize> = None;
        if l > 0 {
            for i in 0..n {
                let low = r.get(l);
                let t = (uppers[i] << l) | low;
                if t == tail && idx.is_none() {
                    idx = Some(i);
                }
            }
        } else {
            for (i, &u) in uppers.iter().enumerate() {
                if u == tail && idx.is_none() {
                    idx = Some(i);
                }
            }
        }
        let idx = idx?;
        let mut out = Vec::new();
        for i in 0..=idx {
            out.clear();
            decode_value(&self.tables, &mut r, &mut out);
            if i == idx {
                return Some(out);
            }
        }
        unreachable!()
    }
}

fn pct(mut v: Vec<u64>) -> (u64, u64, u64, u64) {
    v.sort();
    let n = v.len();
    (
        v[n / 2],
        v[n * 9 / 10],
        v[n * 99 / 100],
        v.iter().sum::<u64>() / n as u64,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fdir = args
        .get(1)
        .expect("usage: frozen_replay <frozen> <lmdb> <rocks> [n]");
    let ldir = &args[2];
    let rdir = &args[3];
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    println!("opening frozen...");
    let t0 = Instant::now();
    let frozen = Frozen::open(fdir);
    println!(
        "frozen open in {:.1}s (offset tables in RAM)",
        t0.elapsed().as_secs_f64()
    );

    let env = lmdb::Environment::new()
        .set_flags(lmdb::EnvironmentFlags::READ_ONLY)
        .set_max_dbs(600)
        .set_max_readers(1024)
        .set_map_size(6 * 1024 * 1024 * 1024 * 1024)
        .open(std::path::Path::new(ldir))
        .expect("open lmdb");
    let dbs: Vec<lmdb::Database> = (0u16..=255)
        .map(|s| env.open_db(Some(format!("{:02x}", s).as_str())).unwrap())
        .collect();

    let rocks = DB::open_for_read_only(&Options::default(), rdir.as_str(), false).expect("rocks");

    // ---- sample keys ----
    println!("sampling {} hit keys + {} miss keys...", n, n);
    let mut rng = rand::rng();
    let mut hit_keys: Vec<([u8; 16], Vec<u8>)> = Vec::with_capacity(n);
    while hit_keys.len() < n {
        let mut probe = [0u8; 16];
        rng.fill_bytes(&mut probe);
        let mut it = rocks.iterator(IteratorMode::From(&probe, Direction::Forward));
        if let Some(Ok((k, v))) = it.next() {
            let mut kk = [0u8; 16];
            kk.copy_from_slice(&k);
            hit_keys.push((kk, v.to_vec()));
        }
    }
    let mut miss_keys: Vec<[u8; 16]> = Vec::with_capacity(n);
    while miss_keys.len() < n {
        let mut probe = [0u8; 16];
        rng.fill_bytes(&mut probe);
        miss_keys.push(probe); // P(exists) ~ 2^-93: certain miss
    }

    // ---- correctness + latency: frozen ----
    let mut f_hit_ns = Vec::with_capacity(n);
    let mut wrong = 0u64;
    for (k, truth) in &hit_keys {
        let t = Instant::now();
        let got = frozen.get(k);
        f_hit_ns.push(t.elapsed().as_nanos() as u64);
        match got {
            Some(v) if &v == truth => {}
            _ => wrong += 1,
        }
    }
    let mut f_miss_ns = Vec::with_capacity(n);
    let mut fp = 0u64;
    for k in &miss_keys {
        let t = Instant::now();
        let got = frozen.get(k);
        f_miss_ns.push(t.elapsed().as_nanos() as u64);
        if got.is_some() {
            fp += 1;
        }
    }
    println!(
        "frozen correctness: {} wrong hits, {} false positives (of {} each)",
        wrong, fp, n
    );

    // ---- latency: lmdb same keys ----
    let txn = env.begin_ro_txn().unwrap();
    let mut l_hit_ns = Vec::with_capacity(n);
    let mut l_wrong = 0u64;
    for (k, truth) in &hit_keys {
        let t = Instant::now();
        let got: Result<&[u8], _> = txn.get(dbs[k[0] as usize], k);
        l_hit_ns.push(t.elapsed().as_nanos() as u64);
        match got {
            Ok(v) if v == truth.as_slice() => {}
            _ => l_wrong += 1,
        }
    }
    let mut l_miss_ns = Vec::with_capacity(n);
    for k in &miss_keys {
        let t = Instant::now();
        let _: Result<&[u8], _> = txn.get(dbs[k[0] as usize], k);
        l_miss_ns.push(t.elapsed().as_nanos() as u64);
    }
    println!("lmdb correctness: {} wrong hits", l_wrong);

    for (name, v) in [
        ("frozen hit ", f_hit_ns),
        ("frozen miss", f_miss_ns),
        ("lmdb   hit ", l_hit_ns),
        ("lmdb   miss", l_miss_ns),
    ] {
        let (p50, p90, p99, mean) = pct(v);
        println!(
            "[latency {name}] p50={:.1}us p90={:.1}us p99={:.1}us mean={:.1}us",
            p50 as f64 / 1e3,
            p90 as f64 / 1e3,
            p99 as f64 / 1e3,
            mean as f64 / 1e3
        );
    }

    // ---- throughput: frozen, 64 threads, mixed 40% hit ----
    let probes: Vec<[u8; 16]> = hit_keys
        .iter()
        .map(|(k, _)| *k)
        .chain(miss_keys.iter().copied())
        .collect();
    let counter = AtomicU64::new(0);
    let t = Instant::now();
    let rounds = 8usize;
    (0..64usize).into_par_iter().for_each(|ti| {
        for r in 0..rounds {
            for (i, k) in probes.iter().enumerate() {
                if (i + ti + r) % 64 == ti % 64 {
                    let _ = frozen.get(k);
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });
    let el = t.elapsed().as_secs_f64();
    println!(
        "[throughput frozen] {} probes in {:.1}s = {:.0} probes/s (64 threads)",
        counter.load(Ordering::Relaxed),
        el,
        counter.load(Ordering::Relaxed) as f64 / el
    );
    println!(
        "REPLAY DONE {}",
        if wrong == 0 && fp == 0 && l_wrong == 0 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}
