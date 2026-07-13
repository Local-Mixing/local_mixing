//! Pilot benchmark for the frozen table on fleet-like hardware.
//! Uses a probe pack (real keys + ground-truth values) instead of the rocks
//! source. Run with cold page cache for honest disk numbers.
//!
//! usage: frozen_pilot <frozen_dir> <probes.bin> [threads]
//!
//! Stages: open -> correctness+cold latency (hits, misses) -> filter build
//! (timed) -> filtered miss latency + fp rate -> warm throughput.

use rand::RngCore;
use rayon::prelude::*;
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use xorf::{BinaryFuse8, Filter};

const BUCKETS: usize = 1 << 20;
const GI_CLAMP: u32 = 11;
const ESC: u32 = u32::MAX;
const MAXLEN: usize = 40;

// ---------------- bit reader / huffman / decode (mirrors frozen_table) ----
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
    ((hi >> 56) as usize, ((hi >> 36) & 0xF_FFFF) as u32, ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52))
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

// ------------------------------------------------------------ reader
struct FrozenShard {
    file: std::fs::File,
    offs: Vec<u64>,
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
                FrozenShard { file, offs, data_base: head_len as u64 }
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
        let mut uppers = Vec::with_capacity(n);
        let mut up = 0u64;
        for _ in 0..n {
            while r.get1() == 0 {
                up += 1;
            }
            uppers.push(up);
        }
        let mut idx: Option<usize> = None;
        for i in 0..n {
            let low = if l > 0 { r.get(l) } else { 0 };
            let t = (uppers[i] << l) | low;
            if t == tail && idx.is_none() {
                idx = Some(i);
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

/// tails-only scan for filter construction
fn scan_shard_keys(dir: &str, shard: usize) -> Vec<u64> {
    let head_len = (24usize + (BUCKETS + 1) * 5) as u64;
    let file = std::fs::File::open(format!("{dir}/shard_{shard:02x}.frz")).unwrap();
    let mut head = vec![0u8; head_len as usize];
    file.read_exact_at(&mut head, 0).unwrap();
    let mut offs = Vec::with_capacity(BUCKETS + 1);
    for i in 0..=BUCKETS {
        let mut b = [0u8; 8];
        b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
        offs.push(u64::from_le_bytes(b));
    }
    let data_len = offs[BUCKETS];
    let mut keys: Vec<u64> = Vec::new();
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
    keys
}

fn pct(mut v: Vec<u64>) -> (f64, f64, f64, f64) {
    v.sort();
    let n = v.len();
    (
        v[n / 2] as f64 / 1e3,
        v[n * 9 / 10] as f64 / 1e3,
        v[n * 99 / 100] as f64 / 1e3,
        (v.iter().sum::<u64>() / n as u64) as f64 / 1e3,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fdir = args.get(1).expect("usage: frozen_pilot <frozen> <probes.bin> [threads]").clone();
    let probes_path = args[2].clone();
    let nthreads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    // load probes
    let pdata = std::fs::read(&probes_path).expect("probes");
    let n = u32::from_le_bytes(pdata[0..4].try_into().unwrap()) as usize;
    let mut pos = 4usize;
    let mut hits: Vec<([u8; 16], Vec<u8>)> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut k = [0u8; 16];
        k.copy_from_slice(&pdata[pos..pos + 16]);
        pos += 16;
        let vl = u16::from_le_bytes(pdata[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        hits.push((k, pdata[pos..pos + vl].to_vec()));
        pos += vl;
    }
    let mut rng = rand::rng();
    let mut misses: Vec<[u8; 16]> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut k = [0u8; 16];
        rng.fill_bytes(&mut k);
        misses.push(k);
    }
    println!("probes loaded: {} hits, {} misses", hits.len(), misses.len());

    let t0 = Instant::now();
    let frozen = Frozen::open(&fdir);
    println!("[open] {:.1}s (offset tables -> RAM)", t0.elapsed().as_secs_f64());

    // cold latency (run this binary right after dropping caches)
    let mut wrong = 0u64;
    let mut h_ns = Vec::with_capacity(hits.len());
    for (k, truth) in &hits {
        let t = Instant::now();
        let got = frozen.get(k);
        h_ns.push(t.elapsed().as_nanos() as u64);
        match got {
            Some(v) if &v == truth => {}
            _ => wrong += 1,
        }
    }
    let mut fp0 = 0u64;
    let mut m_ns = Vec::with_capacity(misses.len());
    for k in &misses {
        let t = Instant::now();
        if frozen.get(k).is_some() {
            fp0 += 1;
        }
        m_ns.push(t.elapsed().as_nanos() as u64);
    }
    let (p50, p90, p99, mean) = pct(h_ns);
    println!("[hit  latency] p50={p50:.1}us p90={p90:.1}us p99={p99:.1}us mean={mean:.1}us wrong={wrong}");
    let (p50, p90, p99, mean) = pct(m_ns);
    println!("[miss latency] p50={p50:.1}us p90={p90:.1}us p99={p99:.1}us mean={mean:.1}us false_pos={fp0}");

    // filter build (fleet-like timing), batches bound RAM
    println!("[filter] building 256 BinaryFuse8 (batches of 32)...");
    let tf = Instant::now();
    let mut filters: Vec<Option<BinaryFuse8>> = (0..256).map(|_| None).collect();
    let mut fp_bytes = 0u64;
    for batch in 0..8 {
        let built: Vec<(usize, BinaryFuse8)> = (batch * 32..(batch + 1) * 32)
            .into_par_iter()
            .map(|s| {
                let mut keys = scan_shard_keys(&fdir, s);
                keys.sort_unstable();
                keys.dedup();
                let f = BinaryFuse8::try_from(&keys).expect("fuse build");
                (s, f)
            })
            .collect();
        for (s, f) in built {
            fp_bytes += f.len() as u64;
            filters[s] = Some(f);
        }
    }
    let filters: Vec<BinaryFuse8> = filters.into_iter().map(|f| f.unwrap()).collect();
    println!(
        "[filter] built in {:.0}s, {:.1} GB fingerprints in RAM",
        tf.elapsed().as_secs_f64(),
        fp_bytes as f64 / 1e9
    );

    // filtered miss path
    let mut ffp = 0u64;
    let mut fm_ns = Vec::with_capacity(misses.len());
    for k in &misses {
        let (s, b, tl) = split_key(k);
        let m = mix76(s, b, tl);
        let t = Instant::now();
        let present = filters[s].contains(&m);
        let r = if present { frozen.get(k) } else { None };
        fm_ns.push(t.elapsed().as_nanos() as u64);
        if present {
            ffp += 1;
        }
        let _ = r;
    }
    let mut ffn = 0u64;
    for (k, _) in &hits {
        let (s, b, tl) = split_key(k);
        if !filters[s].contains(&mix76(s, b, tl)) {
            ffn += 1;
        }
    }
    let (p50, p90, p99, mean) = pct(fm_ns);
    println!(
        "[filtered miss] p50={p50:.3}us p90={p90:.3}us p99={p99:.3}us mean={mean:.3}us fp={ffp}/{} ({:.3}%) false_neg={ffn} (MUST be 0)",
        misses.len(),
        100.0 * ffp as f64 / misses.len() as f64
    );

    // mixed throughput with filter (production shape: ~41% hit)
    let probes: Vec<[u8; 16]> = hits.iter().map(|(k, _)| *k).chain(misses.iter().copied()).collect();
    let counter = AtomicU64::new(0);
    let tt = Instant::now();
    (0..nthreads).into_par_iter().for_each(|ti| {
        for r in 0..8usize {
            for (i, k) in probes.iter().enumerate() {
                if (i + ti + r) % nthreads == ti % nthreads {
                    let (s, b, tl) = split_key(k);
                    if filters[s].contains(&mix76(s, b, tl)) {
                        let _ = frozen.get(k);
                    }
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });
    let el = tt.elapsed().as_secs_f64();
    println!(
        "[throughput] {} filtered probes in {:.1}s = {:.0} probes/s ({} threads)",
        counter.load(Ordering::Relaxed),
        el,
        counter.load(Ordering::Relaxed) as f64 / el,
        nthreads
    );
    println!("PILOT DONE {}", if wrong == 0 && ffn == 0 { "PASS" } else { "FAIL" });
}
