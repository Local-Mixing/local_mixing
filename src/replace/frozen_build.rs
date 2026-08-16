//! Build a frozen-table store (src/replace/frozen.rs read path) from the
//! legacy sharded LMDB. Library form of the `frozen_from_lmdb` bin so the
//! round-trip is testable. Port of the shuffletests `frozen_table` builder with
//! the RocksDB input layer swapped for LMDB cursors: shard XX = LMDB db "XX"
//! (or "curated_XX" with --curated), whose sorted full-key iteration matches
//! the per-shard range scans the rocks builder used. Same output format:
//! 256 shard files (shard = key byte 0); 2^20 buckets per shard (key bits
//! 8..27); per bucket: Elias-Fano-coded sorted 48-bit key tails (bits 28..75)
//! followed by canonical-Huffman-coded values (context = (circuit width,
//! gate index), symbol = whole gate triple). Escape codes make the format
//! total: every parseable or unparseable value round-trips byte-exactly.
//!
//! Subcommands (run tables, then write, then validate):
//!   frozen_from_lmdb tables   <lmdb_dir> <out_dir> [--curated] [per_range]
//!   frozen_from_lmdb write    <lmdb_dir> <out_dir> [--curated]
//!   frozen_from_lmdb validate <lmdb_dir> <out_dir> [--curated]

use lmdb::{Cursor, Transaction};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const BUCKET_BITS: u32 = 20;
const BUCKETS: usize = 1 << BUCKET_BITS;
const TAIL_BITS: u32 = 48;
const GI_CLAMP: u32 = 11;
const ESC: u32 = u32::MAX;
const MAXLEN: usize = 40; // max Huffman code length (tables flattened to fit)

// ---------------------------------------------------------------- bit I/O
struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}
impl BitWriter {
    fn new() -> Self {
        BitWriter {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    #[inline]
    fn put(&mut self, val: u64, bits: u32) {
        debug_assert!(bits <= 56);
        self.acc |= val << self.nbits;
        self.nbits += bits;
        while self.nbits >= 8 {
            self.buf.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push((self.acc & 0xff) as u8);
        }
        self.buf
    }
}
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

// ------------------------------------------------------- canonical Huffman
#[derive(Clone, Default)]
struct HuffTable {
    enc: HashMap<u32, (u64, u32)>,
    syms: Vec<u32>,
    first_code: Vec<u64>, // per length 0..=MAXLEN
    first_idx: Vec<usize>,
    count: Vec<usize>,
    single: bool,
}

fn huffman_depths(items: &[(u32, u64)]) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = items.len();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(n * 2);
    let mut depth = vec![0u32; n];
    for (i, &(_, f)) in items.iter().enumerate() {
        groups.push(vec![i]);
        heap.push(Reverse((f.max(1), i)));
    }
    while heap.len() > 1 {
        let Reverse((w1, i1)) = heap.pop().unwrap();
        let Reverse((w2, i2)) = heap.pop().unwrap();
        let mut g = std::mem::take(&mut groups[i1]);
        g.extend(std::mem::take(&mut groups[i2]));
        for &l in &g {
            depth[l] += 1;
        }
        let id = groups.len();
        groups.push(g);
        heap.push(Reverse((w1 + w2, id)));
    }
    depth
}

fn build_huffman(freqs: &HashMap<u32, u64>) -> HuffTable {
    let mut items: Vec<(u32, u64)> = freqs.iter().map(|(&s, &f)| (s, f)).collect();
    items.sort();
    if items.is_empty() {
        return HuffTable::default();
    }
    if items.len() == 1 {
        let mut t = HuffTable::default();
        t.enc.insert(items[0].0, (0, 0));
        t.syms = vec![items[0].0];
        t.single = true;
        return t;
    }
    // flatten frequencies until max depth fits MAXLEN
    let mut work = items.clone();
    let mut depth = huffman_depths(&work);
    while *depth.iter().max().unwrap() as usize > MAXLEN {
        for it in work.iter_mut() {
            it.1 = (it.1 >> 1) + 1;
        }
        depth = huffman_depths(&work);
    }
    let sym_lens: Vec<(u32, u32)> = items
        .iter()
        .zip(&depth)
        .map(|(&(s, _), &d)| (s, d))
        .collect();
    rebuild_canonical(sym_lens)
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
    fn encode(&self, w: &mut BitWriter, sym: u32) -> bool {
        match self.enc.get(&sym) {
            Some(&(code, len)) => {
                for b in (0..len).rev() {
                    w.put((code >> b) & 1, 1);
                }
                true
            }
            None => false,
        }
    }
    #[inline]
    fn emit_escape(&self, w: &mut BitWriter) {
        let &(code, len) = self.enc.get(&ESC).expect("escape code");
        for b in (0..len).rev() {
            w.put((code >> b) & 1, 1);
        }
    }
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

// ----------------------------------------------------------- table set
struct Tables {
    header: HuffTable,
    gates: Vec<HuffTable>, // ctx = w*12 + min(gi,11), w in 0..=32
}

#[inline]
fn ctx_of(w: u32, gi: u32) -> usize {
    (w.min(32) as usize) * 12 + gi.min(GI_CLAMP) as usize
}

#[inline]
fn header_sym(g: u32, w: u32, chain: u32) -> u32 {
    (g << 7) | (w << 1) | chain
}

fn save_tables(t: &Tables, path: &str) {
    let mut out: Vec<u8> = Vec::new();
    let dump = |tab: &HuffTable, out: &mut Vec<u8>| {
        out.extend((tab.syms.len() as u32).to_le_bytes());
        for &s in &tab.syms {
            let (_, len) = tab.enc[&s];
            out.extend(s.to_le_bytes());
            out.push(len as u8);
        }
    };
    dump(&t.header, &mut out);
    out.extend((t.gates.len() as u32).to_le_bytes());
    for g in &t.gates {
        dump(g, &mut out);
    }
    std::fs::write(path, out).expect("write tables");
}

fn load_tables(path: &str) -> Tables {
    let data = std::fs::read(path).expect("read tables");
    let mut pos = 0usize;
    fn rd_u32(data: &[u8], pos: &mut usize) -> u32 {
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    }
    fn load_one(data: &[u8], pos: &mut usize) -> HuffTable {
        let n = rd_u32(data, pos) as usize;
        let mut sym_lens = Vec::with_capacity(n);
        for _ in 0..n {
            let s = rd_u32(data, pos);
            let l = data[*pos] as u32;
            *pos += 1;
            sym_lens.push((s, l));
        }
        rebuild_canonical(sym_lens)
    }
    let header = load_one(&data, &mut pos);
    let nctx = rd_u32(&data, &mut pos) as usize;
    let mut gates = Vec::with_capacity(nctx);
    for _ in 0..nctx {
        gates.push(load_one(&data, &mut pos));
    }
    Tables { header, gates }
}

// ---------------------------------------------------------------- values
fn parse_value(v: &[u8]) -> Option<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < v.len() {
        let len = v[pos] as usize;
        pos += 1;
        if pos + len > v.len() || len % 3 != 0 {
            return None;
        }
        out.push(&v[pos..pos + len]);
        pos += len;
    }
    if out.is_empty() { None } else { Some(out) }
}

fn encode_value(t: &Tables, w_out: &mut BitWriter, v: &[u8]) {
    match parse_value(v) {
        Some(blobs) => {
            let nb = blobs.len();
            for (bi, blob) in blobs.iter().enumerate() {
                let g = (blob.len() / 3) as u32;
                let w = blob.iter().copied().max().unwrap_or(0) as u32 + 1;
                let chain = (bi + 1 < nb) as u32;
                let hs = header_sym(g, w, chain);
                if g > 120 || w > 32 || !t.header.enc.contains_key(&hs) {
                    t.header.emit_escape(w_out);
                    w_out.put(g as u64, 8);
                    w_out.put(w.min(255) as u64, 8);
                    w_out.put(chain as u64, 1);
                } else {
                    t.header.encode(w_out, hs);
                }
                for (gi, gate) in blob.chunks(3).enumerate() {
                    let tab = &t.gates[ctx_of(w, gi as u32)];
                    if gate.iter().any(|&x| x >= 32) {
                        tab.emit_escape(w_out);
                        w_out.put(1, 1); // wide
                        for &x in gate {
                            w_out.put(x as u64, 8);
                        }
                        continue;
                    }
                    let triple = (gate[0] as u32) << 10 | (gate[1] as u32) << 5 | gate[2] as u32;
                    if !tab.encode(w_out, triple) {
                        tab.emit_escape(w_out);
                        w_out.put(0, 1); // narrow
                        w_out.put(triple as u64, 15);
                    }
                }
            }
        }
        None => {
            t.header.emit_escape(w_out);
            w_out.put(121, 8); // raw-value sentinel
            w_out.put(v.len() as u64, 16);
            for &b in v {
                w_out.put(b as u64, 8);
            }
        }
    }
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

// ------------------------------------------------------------- key split
#[inline]
fn split_key(k: &[u8]) -> (usize, u32, u64) {
    let hi = u64::from_be_bytes(k[0..8].try_into().unwrap());
    let lo = u64::from_be_bytes(k[8..16].try_into().unwrap());
    let shard = (hi >> 56) as usize;
    let bucket = ((hi >> 36) & 0xF_FFFF) as u32;
    let tail = ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52);
    (shard, bucket, tail)
}

// encode one bucket (tails + values) into bytes
fn encode_bucket(t: &Tables, tails: &[u64], vals: &[Vec<u8>]) -> Vec<u8> {
    let n = tails.len();
    let mut bw = BitWriter::new();
    let bits_n = 64 - (n as u64).leading_zeros();
    let l = TAIL_BITS.saturating_sub(bits_n);
    bw.put(n as u64, 16);
    bw.put(l as u64, 6);
    let mut prev_up = 0u64;
    for &tl in tails {
        let up = tl >> l;
        for _ in 0..(up - prev_up) {
            bw.put(0, 1);
        }
        bw.put(1, 1);
        prev_up = up;
    }
    if l > 0 {
        for &tl in tails {
            bw.put(tl & ((1u64 << l) - 1), l);
        }
    }
    for v in vals {
        encode_value(t, &mut bw, v);
    }
    bw.finish()
}

// ---------------------------------------------------------------- lmdb input
pub fn open_env(dir: &str) -> lmdb::Environment {
    lmdb::Environment::new()
        .set_flags(lmdb::EnvironmentFlags::READ_ONLY | lmdb::EnvironmentFlags::NO_LOCK)
        .set_max_readers(10000)
        .set_max_dbs(556)
        .set_map_size(800 * 1024 * 1024 * 1024)
        .open(std::path::Path::new(dir))
        .expect("open lmdb dir")
}

pub fn open_shards(env: &lmdb::Environment, prefix: &str) -> Vec<lmdb::Database> {
    (0u16..=255)
        .map(|s| {
            let name = format!("{prefix}{s:02x}");
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("open shard db {name}: {e:?}"))
        })
        .collect()
}

// ---------------------------------------------------------------- stages
pub fn stage_tables(lmdb_dir: &str, prefix: &str, out_dir: &str, per_range: usize) {
    let env = open_env(lmdb_dir);
    let dbs = open_shards(&env, prefix);
    let hdr = Mutex::new(HashMap::<u32, u64>::new());
    let gats = Mutex::new(HashMap::<(usize, u32), u64>::new());
    (0..256u32).into_par_iter().for_each(|r| {
        let mut lh: HashMap<u32, u64> = HashMap::new();
        let mut lg: HashMap<(usize, u32), u64> = HashMap::new();
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(dbs[r as usize]).expect("cursor");
        let mut n = 0usize;
        for (key, v) in cursor.iter() {
            assert_eq!(key[0], r as u8, "key outside its shard db");
            if let Some(blobs) = parse_value(&v) {
                let nb = blobs.len();
                for (bi, blob) in blobs.iter().enumerate() {
                    let g = (blob.len() / 3) as u32;
                    let w = blob.iter().copied().max().unwrap_or(0) as u32 + 1;
                    if g > 120 || w > 32 {
                        continue;
                    }
                    let chain = (bi + 1 < nb) as u32;
                    *lh.entry(header_sym(g, w, chain)).or_insert(0) += 1;
                    for (gi, gate) in blob.chunks(3).enumerate() {
                        if gate.iter().any(|&x| x >= 32) {
                            continue;
                        }
                        let triple =
                            (gate[0] as u32) << 10 | (gate[1] as u32) << 5 | gate[2] as u32;
                        *lg.entry((ctx_of(w, gi as u32), triple)).or_insert(0) += 1;
                    }
                }
            }
            n += 1;
            if n >= per_range {
                break;
            }
        }
        let mut h = hdr.lock().unwrap();
        for (k, v) in lh {
            *h.entry(k).or_insert(0) += v;
        }
        drop(h);
        let mut g = gats.lock().unwrap();
        for (k, v) in lg {
            *g.entry(k).or_insert(0) += v;
        }
    });

    let mut hdr = hdr.into_inner().unwrap();
    hdr.insert(ESC, 1);
    let gats = gats.into_inner().unwrap();
    let nctx = 33 * 12;
    let mut per_ctx: Vec<HashMap<u32, u64>> = vec![HashMap::new(); nctx];
    for ((c, s), f) in gats {
        per_ctx[c].insert(s, f);
    }
    std::fs::create_dir_all(out_dir).unwrap();
    let mut gate_tables = Vec::with_capacity(nctx);
    for m in per_ctx.iter_mut() {
        m.insert(ESC, 1);
        gate_tables.push(build_huffman(m));
    }
    let tables = Tables {
        header: build_huffman(&hdr),
        gates: gate_tables,
    };
    save_tables(&tables, &format!("{out_dir}/tables.bin"));
    println!(
        "tables built: header_syms={} gate_ctxs={} total_gate_syms={}",
        tables.header.syms.len(),
        tables.gates.len(),
        tables.gates.iter().map(|t| t.syms.len()).sum::<usize>()
    );
}

pub fn stage_write(lmdb_dir: &str, prefix: &str, out_dir: &str) {
    let tables = std::sync::Arc::new(load_tables(&format!("{out_dir}/tables.bin")));
    let env = open_env(lmdb_dir);
    let dbs = open_shards(&env, prefix);
    let total = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let head_len = 24usize + (BUCKETS + 1) * 5;

    (0..256usize).into_par_iter().for_each(|shard| {
        let t = &*tables;
        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(dbs[shard]).expect("cursor");

        let path = format!("{out_dir}/shard_{shard:02x}.frz");
        let mut file = std::fs::File::create(&path).unwrap();
        file.seek(SeekFrom::Start(head_len as u64)).unwrap();
        let mut out = std::io::BufWriter::with_capacity(4 << 20, &mut file);

        let mut offsets: Vec<u64> = Vec::with_capacity(BUCKETS + 1);
        offsets.push(0);
        let mut data_len = 0u64;
        let mut cur_bucket: u32 = 0;
        let mut tails: Vec<u64> = Vec::new();
        let mut vals: Vec<Vec<u8>> = Vec::new();
        let mut count = 0u64;

        macro_rules! flush_to {
            ($upto:expr) => {{
                if !tails.is_empty() {
                    let bytes = encode_bucket(t, &tails, &vals);
                    out.write_all(&bytes).unwrap();
                    data_len += bytes.len() as u64;
                    tails.clear();
                    vals.clear();
                }
                while (offsets.len() as u32) <= $upto {
                    offsets.push(data_len);
                }
            }};
        }

        for (key, v) in cursor.iter() {
            assert_eq!(key[0] as usize, shard, "key outside its shard db");
            let (_, bucket, tail) = split_key(key);
            if bucket != cur_bucket {
                flush_to!(bucket);
                cur_bucket = bucket;
            }
            tails.push(tail);
            vals.push(v.to_vec());
            count += 1;
        }
        flush_to!(BUCKETS as u32);
        out.flush().unwrap();
        drop(out);

        // backpatch header + offsets
        let mut head: Vec<u8> = Vec::with_capacity(head_len);
        head.extend(b"FRZTBL01");
        head.extend(count.to_le_bytes());
        head.extend(data_len.to_le_bytes());
        for &o in &offsets {
            head.extend(&o.to_le_bytes()[0..5]);
        }
        assert_eq!(head.len(), head_len);
        file.write_at(&head, 0).unwrap();
        file.sync_all().unwrap();

        total.fetch_add(count, Ordering::Relaxed);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 16 == 0 || d == 256 {
            println!(
                "[write] shards {}/256 entries={}",
                d,
                total.load(Ordering::Relaxed)
            );
        }
    });
    println!(
        "write done: total_entries={}",
        total.load(Ordering::Relaxed)
    );
}

pub fn stage_validate(lmdb_dir: &str, prefix: &str, out_dir: &str) {
    let tables = std::sync::Arc::new(load_tables(&format!("{out_dir}/tables.bin")));
    let env = open_env(lmdb_dir);
    let dbs = open_shards(&env, prefix);
    let mismatches = AtomicU64::new(0);
    let checked = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let head_len = 24usize + (BUCKETS + 1) * 5;

    (0..256usize).into_par_iter().for_each(|shard| {
        let t = &*tables;
        let path = format!("{out_dir}/shard_{shard:02x}.frz");
        let file = std::fs::File::open(&path).unwrap();
        let mut head = vec![0u8; head_len];
        file.read_exact_at(&mut head, 0).unwrap();
        assert_eq!(&head[0..8], b"FRZTBL01");
        let count = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let read_off = |i: usize| -> u64 {
            let mut b = [0u8; 8];
            b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
            u64::from_le_bytes(b)
        };

        let txn = env.begin_ro_txn().expect("ro txn");
        let mut cursor = txn.open_ro_cursor(dbs[shard]).expect("cursor");

        let mut cur_bucket = u32::MAX;
        let mut bucket_buf: Vec<u8> = Vec::new();
        let mut br_pos = 0usize; // decoded entries so far in bucket
        let mut btails: Vec<u64> = Vec::new();
        let mut reader: Option<BitReader> = None;
        let mut scratch: Vec<u8> = Vec::new();
        let mut n = 0u64;
        let mut bad = 0u64;

        // Work around borrow of bucket_buf: decode bucket fully on entry-by-entry
        // basis using an index-based reader recreated per bucket.
        for (key, v) in cursor.iter() {
            assert_eq!(key[0] as usize, shard, "key outside its shard db");
            let (_, bucket, tail) = split_key(key);
            if bucket != cur_bucket {
                if reader.is_some() && br_pos != btails.len() {
                    bad += 1; // frozen bucket had extra entries
                }
                reader = None;
                cur_bucket = bucket;
                let o0 = read_off(bucket as usize);
                let o1 = read_off(bucket as usize + 1);
                bucket_buf = vec![0u8; (o1 - o0) as usize];
                file.read_exact_at(&mut bucket_buf, head_len as u64 + o0)
                    .unwrap();
                // SAFETY: bucket_buf lives until reassigned; reader dropped first.
                let slice: &'static [u8] =
                    unsafe { std::slice::from_raw_parts(bucket_buf.as_ptr(), bucket_buf.len()) };
                let mut r = BitReader::new(slice);
                let bn = r.get(16) as usize;
                let l = r.get(6) as u32;
                btails.clear();
                let mut up = 0u64;
                for _ in 0..bn {
                    while r.get1() == 0 {
                        up += 1;
                    }
                    btails.push(up);
                }
                if l > 0 {
                    for i in 0..bn {
                        let low = r.get(l);
                        btails[i] = (btails[i] << l) | low;
                    }
                }
                br_pos = 0;
                reader = Some(r);
            }
            let r = reader.as_mut().unwrap();
            if br_pos >= btails.len() || btails[br_pos] != tail {
                bad += 1;
            }
            scratch.clear();
            decode_value(t, r, &mut scratch);
            if scratch.as_slice() != &v[..] {
                bad += 1;
            }
            br_pos += 1;
            n += 1;
        }
        if reader.is_some() && br_pos != btails.len() {
            bad += 1;
        }
        if n != count {
            bad += 1;
        }
        checked.fetch_add(n, Ordering::Relaxed);
        mismatches.fetch_add(bad, Ordering::Relaxed);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 32 == 0 || d == 256 {
            println!(
                "[validate] shards {}/256 checked={} mismatches={}",
                d,
                checked.load(Ordering::Relaxed),
                mismatches.load(Ordering::Relaxed)
            );
        }
    });
    let bad = mismatches.load(Ordering::Relaxed);
    println!(
        "validate done: checked={} mismatches={} -> {}",
        checked.load(Ordering::Relaxed),
        bad,
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    if bad != 0 {
        std::process::exit(1);
    }
}


// ------------------------------------------------- frozen -> frozen pool swap

/// Decoded [t,p,n] triples -> raw stored bytes (legacy-swapped-controls
/// layout: stored (t, c1, c2) decodes as (t, c2, c1), so write back (t, n, p)).
fn circuits_to_value(circuits: &[Vec<[u16; 3]>]) -> Vec<u8> {
    let mut v = Vec::new();
    for c in circuits {
        assert!(c.len() * 3 <= 255, "circuit too long for the value format");
        v.push((c.len() * 3) as u8);
        for &[t, p, n] in c {
            assert!(t < 256 && p < 256 && n < 256);
            v.push(t as u8);
            v.push(n as u8);
            v.push(p as u8);
        }
    }
    v
}

fn parse_sgdb_file(path: &str) -> Vec<Vec<[u16; 3]>> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = s.lines();
    let header = lines.next().expect("empty sgdb file");
    assert!(header.starts_with("sgdb1 "), "bad sgdb header in {path}");
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(';')
                .map(|t| {
                    let v: Vec<u16> = t.split(',').map(|x| x.parse().expect("bad triple")).collect();
                    [v[0], v[1], v[2]]
                })
                .collect()
        })
        .collect()
}

/// The deterministic reference spelling of a raw value: decode friends under
/// the swapped convention, take the lexicographically-least among the
/// minimal-size ones — the SAME rule mgdb_build used for its MANIFEST.
fn min_reference(v: &[u8]) -> Option<(usize, String)> {
    let blobs = parse_value(v)?;
    let mn = blobs.iter().map(|b| b.len() / 3).min()?;
    let mut best: Option<Vec<[u16; 3]>> = None;
    for b in blobs.iter().filter(|b| b.len() / 3 == mn) {
        let dec: Vec<[u16; 3]> = b
            .chunks(3)
            .map(|c| [c[0] as u16, c[2] as u16, c[1] as u16])
            .collect();
        if best.as_ref().is_none_or(|cur| dec < *cur) {
            best = Some(dec);
        }
    }
    let best = best?;
    let txt: Vec<String> = best.iter().map(|&[t, p, n]| format!("{t},{p},{n}")).collect();
    Some((mn, txt.join(";")))
}

/// Rebuild a curated frozen store with the M1/M2/M3 pools swapped in:
/// entries whose minimal retained friend has 1 gate get the SGDB pool
/// (`m1_pool` file), 2..=3 gates get their per-permutation pool from
/// `mgdb_dir` (MANIFEST.tsv maps minimal spelling -> file). Everything else
/// copies BYTE-IDENTICALLY (source tables are reused, untouched buckets are
/// copied without re-encoding). filters.bin and tables.bin are copied.
pub fn pool_swap(src_dir: &str, out_dir: &str, m1_pool: &str, mgdb_dir: &str) {
    pool_swap_upto(src_dir, out_dir, Some(m1_pool), mgdb_dir, 3)
}

/// Generalized pool swap: entries whose minimal retained friend has
/// 1..=max_min gates get their pool from `mgdb_dir`'s manifest (keyed by
/// minimal spelling); `m1_pool` (when Some) overrides the mn==1 entry with a
/// dedicated file. Layerable: running against an ALREADY-SWAPPED store is a
/// no-op for previously swapped entries (their minimal friend is now a big
/// pool circuit, above max_min), so an M4 layer can be applied on top of an
/// M1-M3 store without touching it.
pub fn pool_swap_upto(
    src_dir: &str,
    out_dir: &str,
    m1_pool: Option<&str>,
    mgdb_dir: &str,
    max_min: usize,
) {
    let tables = std::sync::Arc::new(load_tables(&format!("{src_dir}/tables.bin")));
    std::fs::create_dir_all(out_dir).unwrap();
    std::fs::copy(format!("{src_dir}/tables.bin"), format!("{out_dir}/tables.bin")).unwrap();
    if std::path::Path::new(&format!("{src_dir}/filters.bin")).exists() {
        std::fs::copy(format!("{src_dir}/filters.bin"), format!("{out_dir}/filters.bin")).unwrap();
    }

    // Pools: minimal-spelling text -> encoded raw value bytes.
    let m1_value =
        std::sync::Arc::new(m1_pool.map(|p| circuits_to_value(&parse_sgdb_file(p))));
    let mut pools: HashMap<String, Vec<u8>> = HashMap::new();
    let manifest = std::fs::read_to_string(format!("{mgdb_dir}/MANIFEST.tsv")).expect("MANIFEST");
    for line in manifest.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 5, "bad manifest line: {line}");
        let file = f[0];
        let minimal = f[4].to_string();
        let val = circuits_to_value(&parse_sgdb_file(&format!("{mgdb_dir}/{file}")));
        assert!(pools.insert(minimal, val).is_none(), "duplicate minimal spelling in manifest");
    }
    let pools = std::sync::Arc::new(pools);
    println!(
        "[pool-swap] pools loaded: m1 override {} bytes, {} manifest pools, max_min {max_min}",
        m1_value.as_ref().as_ref().map_or(0, |v| v.len()),
        pools.len()
    );

    let head_len = 24usize + (BUCKETS + 1) * 5;
    let swapped = AtomicU64::new(0);
    let swapped_m1 = AtomicU64::new(0);
    let missing = AtomicU64::new(0);
    let entries = AtomicU64::new(0);

    (0..256usize).into_par_iter().for_each(|shard| {
        let t = &*tables;
        let src_path = format!("{src_dir}/shard_{shard:02x}.frz");
        let file = std::fs::File::open(&src_path).unwrap();
        let mut head = vec![0u8; head_len];
        file.read_exact_at(&mut head, 0).unwrap();
        assert_eq!(&head[0..8], b"FRZTBL01");
        let count = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let read_off = |i: usize| -> u64 {
            let mut b = [0u8; 8];
            b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
            u64::from_le_bytes(b)
        };

        let out_path = format!("{out_dir}/shard_{shard:02x}.frz");
        let mut ofile = std::fs::File::create(&out_path).unwrap();
        ofile.seek(SeekFrom::Start(head_len as u64)).unwrap();
        let mut out = std::io::BufWriter::with_capacity(4 << 20, &mut ofile);
        let mut offsets: Vec<u64> = Vec::with_capacity(BUCKETS + 1);
        offsets.push(0);
        let mut data_len = 0u64;

        for bkt in 0..BUCKETS {
            let (o0, o1) = (read_off(bkt), read_off(bkt + 1));
            if o0 == o1 {
                offsets.push(data_len);
                continue;
            }
            let mut buf = vec![0u8; (o1 - o0) as usize];
            file.read_exact_at(&mut buf, head_len as u64 + o0).unwrap();
            // decode the bucket: tails + values
            let mut r = BitReader::new(&buf);
            let n = r.get(16) as usize;
            let l = r.get(6) as u32;
            let mut tails: Vec<u64> = Vec::with_capacity(n);
            let mut up = 0u64;
            for _ in 0..n {
                while r.get1() == 0 {
                    up += 1;
                }
                tails.push(up << l);
            }
            if l > 0 {
                for tl in tails.iter_mut() {
                    *tl |= r.get(l);
                }
            }
            let mut vals: Vec<Vec<u8>> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut v = Vec::new();
                decode_value(t, &mut r, &mut v);
                vals.push(v);
            }
            entries.fetch_add(n as u64, Ordering::Relaxed);
            // swap target values
            let mut touched = false;
            for v in vals.iter_mut() {
                if let Some((mn, minimal)) = min_reference(v) {
                    if mn == 1 && m1_value.as_ref().is_some() {
                        *v = m1_value.as_ref().as_ref().unwrap().clone();
                        touched = true;
                        swapped_m1.fetch_add(1, Ordering::Relaxed);
                    } else if mn <= max_min {
                        match pools.get(&minimal) {
                            Some(nv) => {
                                *v = nv.clone();
                                touched = true;
                                swapped.fetch_add(1, Ordering::Relaxed);
                            }
                            None => {
                                missing.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
            let bytes = if touched {
                encode_bucket(t, &tails, &vals)
            } else {
                buf // byte-identical copy
            };
            out.write_all(&bytes).unwrap();
            data_len += bytes.len() as u64;
            offsets.push(data_len);
        }
        out.flush().unwrap();
        drop(out);
        let mut newhead: Vec<u8> = Vec::with_capacity(head_len);
        newhead.extend(b"FRZTBL01");
        newhead.extend(count.to_le_bytes());
        newhead.extend(data_len.to_le_bytes());
        for &o in &offsets {
            newhead.extend(&o.to_le_bytes()[0..5]);
        }
        assert_eq!(newhead.len(), head_len);
        ofile.write_at(&newhead, 0).unwrap();
        ofile.sync_all().unwrap();
    });
    println!(
        "[pool-swap] done: {} entries walked, {} m2/m3 values swapped, {} m1 swapped, {} MISSING pools",
        entries.load(Ordering::Relaxed),
        swapped.load(Ordering::Relaxed),
        swapped_m1.load(Ordering::Relaxed),
        missing.load(Ordering::Relaxed)
    );
    assert_eq!(missing.load(Ordering::Relaxed), 0, "some M2/M3 entries had no pool");
}
