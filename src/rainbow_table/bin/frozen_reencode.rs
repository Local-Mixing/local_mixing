//! Frozen→frozen store converter: legacy-swapped-controls → native.
//!
//! The curated frozen store was built from a Rocks source whose VALUES carry
//! each gate's two controls in swapped order (the g57 triple [a,x,y] is
//! asymmetric — ¬x∧y — so the order is semantic, not cosmetic). The runtime
//! repairs this at decode time via `swap_value_controls` when
//! `FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls`. This tool bakes
//! that repair into the store itself: decode every value, apply the exact
//! shim transform, re-encode. The result reads correctly under `native`.
//!
//! Keys are untouched (the reader derives them natively; lookups already hit),
//! and `filters.bin` is keys-only, so it is copied byte-for-byte.
//!
//! The entropy tables are TRANSFORMED, not retrained: the control swap is a
//! context-preserving bijection on the 15-bit gate-symbol space
//! (`b0<<10|b1<<5|b2` → same b0, b1↔b2; the context `(w, gi)` is symmetric in
//! the controls because w = max(byte)+1). Mapping every non-ESC symbol through
//! the swap while KEEPING ITS CODE LENGTH yields tables exactly as optimal as
//! the originals — and makes every value's encoded bit length, every bucket's
//! byte length, the whole offset array, and every shard's file size
//! byte-identical to the old store. That identity is asserted throughout: any
//! encoder drift aborts at the first divergent bucket.
//!
//! MUST be built/run in release: `swap_value_controls` carries a
//! `debug_assert_eq!(len % 3, 0)` that would abort a debug build on stored
//! raw (unparseable) values, which the production reader — also release —
//! transforms without complaint.
//!
//! Subcommands:
//!   frozen_reencode tables <old_dir> <new_dir>   transform tables.bin
//!   frozen_reencode write  <old_dir> <new_dir>   re-encode all 256 shards
//!   frozen_reencode verify <old_dir> <new_dir>   native(new) == shim(old), all keys

use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicU64, Ordering};

const BUCKET_BITS: u32 = 20;
const BUCKETS: usize = 1 << BUCKET_BITS;
const TAIL_BITS: u32 = 48;
const GI_CLAMP: u32 = 11;
const ESC: u32 = u32::MAX;
const MAXLEN: usize = 40;

// ---------------------------------------------------------------- bit I/O
// Copied verbatim from frozen_table.rs (the gated Rocks→frozen writer).
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
    gates: Vec<HuffTable>,
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
            // The raw-value length field is 16 bits and BitWriter::put does
            // not mask excess bits; an oversized raw value would silently
            // corrupt the bitstream. Impossible in this bounded curated
            // store, but the assert is free.
            assert!(v.len() < 65536, "raw value exceeds 16-bit length field");
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

// Copied verbatim from src/replace/frozen.rs — this transform's output is,
// by definition, what the production shim serves today. The conversion target.
fn swap_value_controls(value: &mut [u8]) {
    let mut pos = 0usize;
    while pos < value.len() {
        let len = value[pos] as usize;
        pos += 1;
        debug_assert_eq!(len % 3, 0, "frozen: value chunk not gate-aligned");
        let end = (pos + len).min(value.len());
        while pos + 3 <= end {
            value.swap(pos + 1, pos + 2);
            pos += 3;
        }
        pos = end;
    }
}

// encode one bucket (tails + values) into bytes — verbatim from frozen_table.rs
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

// ------------------------------------------------------- the transform
/// The control swap as a bijection on the packed 15-bit gate symbol:
/// b0<<10 | b1<<5 | b2  →  b0<<10 | b2<<5 | b1. An involution; preserves the
/// encoding context (w = max byte + 1 is symmetric, gi positional).
#[inline]
fn swap_sym(s: u32) -> u32 {
    (s & 0x7C00) | ((s & 0x3E0) >> 5) | ((s & 0x1F) << 5)
}

/// Map every non-ESC gate symbol through the swap, KEEPING its code length,
/// then re-canonicalize. Header tables are untouched (g/w/chain are all
/// swap-invariant). Because frequency(swapped corpus, swapped symbol) equals
/// frequency(old corpus, old symbol), the result is exactly as optimal as the
/// original — and every encode event lands at the same bit cost, which is
/// what makes the whole store byte-count-identical.
fn transform_tables(old: &Tables) -> Tables {
    let transform_gate = |tab: &HuffTable| -> HuffTable {
        if tab.syms.is_empty() {
            return HuffTable::default();
        }
        let mut sym_lens: Vec<(u32, u32)> = tab
            .enc
            .iter()
            .map(|(&s, &(_, len))| (if s == ESC { ESC } else { swap_sym(s) }, len))
            .collect();
        sym_lens.sort();
        // Bijectivity check: the swap must not merge symbols.
        assert_eq!(sym_lens.len(), tab.enc.len(), "symbol swap collided");
        for w in sym_lens.windows(2) {
            assert_ne!(w[0].0, w[1].0, "duplicate symbol after swap");
        }
        let out = rebuild_canonical(sym_lens);
        // ESC must keep its old code length: escape emission cost is part of
        // the byte-count-identity invariant.
        if let (Some(&(_, old_len)), Some(&(_, new_len))) = (tab.enc.get(&ESC), out.enc.get(&ESC)) {
            assert_eq!(old_len, new_len, "ESC code length drifted");
        }
        out
    };
    let rebuild_header = |tab: &HuffTable| -> HuffTable {
        let sym_lens: Vec<(u32, u32)> = tab.enc.iter().map(|(&s, &(_, len))| (s, len)).collect();
        rebuild_canonical(sym_lens)
    };
    Tables {
        header: rebuild_header(&old.header),
        gates: old.gates.iter().map(transform_gate).collect(),
    }
}

// ------------------------------------------------------- shard plumbing
const HEAD_LEN: usize = 24 + (BUCKETS + 1) * 5;

fn load_shard_header(file: &std::fs::File) -> (u64, u64, Vec<u8>) {
    let mut head = vec![0u8; HEAD_LEN];
    file.read_exact_at(&mut head, 0).expect("read shard header");
    assert_eq!(&head[0..8], b"FRZTBL01", "bad shard magic");
    let count = u64::from_le_bytes(head[8..16].try_into().unwrap());
    let data_len = u64::from_le_bytes(head[16..24].try_into().unwrap());
    (count, data_len, head)
}

#[inline]
fn read_off(head: &[u8], i: usize) -> u64 {
    let mut b = [0u8; 8];
    b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
    u64::from_le_bytes(b)
}

/// Decode a bucket's tail section; returns (tails, reader positioned at the
/// first value).
fn decode_bucket_tails<'a>(buf: &'a [u8]) -> (Vec<u64>, BitReader<'a>) {
    let mut r = BitReader::new(buf);
    let n = r.get(16) as usize;
    let l = r.get(6) as u32;
    let mut tails = Vec::with_capacity(n);
    let mut up = 0u64;
    for _ in 0..n {
        while r.get1() == 0 {
            up += 1;
        }
        tails.push(up);
    }
    if l > 0 {
        for i in 0..n {
            let low = r.get(l);
            tails[i] = (tails[i] << l) | low;
        }
    }
    (tails, r)
}

// ---------------------------------------------------------------- stages
fn stage_tables(old_dir: &str, new_dir: &str) {
    let old = load_tables(&format!("{old_dir}/tables.bin"));
    let new = transform_tables(&old);
    std::fs::create_dir_all(new_dir).expect("mkdir new_dir");
    save_tables(&new, &format!("{new_dir}/tables.bin"));
    let old_sz = std::fs::metadata(format!("{old_dir}/tables.bin"))
        .unwrap()
        .len();
    let new_sz = std::fs::metadata(format!("{new_dir}/tables.bin"))
        .unwrap()
        .len();
    assert_eq!(old_sz, new_sz, "tables.bin size must be preserved");
    println!(
        "tables transformed: header_syms={} gate_ctxs={} size={} (identical to old)",
        new.header.syms.len(),
        new.gates.len(),
        new_sz
    );
}

fn stage_write(old_dir: &str, new_dir: &str) {
    let t_old = std::sync::Arc::new(load_tables(&format!("{old_dir}/tables.bin")));
    let t_new = std::sync::Arc::new(load_tables(&format!("{new_dir}/tables.bin")));
    let total = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    (0..256usize).into_par_iter().for_each(|shard| {
        let old_path = format!("{old_dir}/shard_{shard:02x}.frz");
        let old_file = std::fs::File::open(&old_path).expect("open old shard");
        let (old_count, old_data_len, old_head) = load_shard_header(&old_file);

        let new_path = format!("{new_dir}/shard_{shard:02x}.frz");
        let mut file = std::fs::File::create(&new_path).unwrap();
        file.seek(SeekFrom::Start(HEAD_LEN as u64)).unwrap();
        let mut out = std::io::BufWriter::with_capacity(4 << 20, &mut file);

        let mut offsets: Vec<u64> = Vec::with_capacity(BUCKETS + 1);
        offsets.push(0);
        let mut data_len = 0u64;
        let mut count = 0u64;
        let mut bucket_buf: Vec<u8> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();

        for b in 0..BUCKETS {
            let o0 = read_off(&old_head, b);
            let o1 = read_off(&old_head, b + 1);
            if o1 == o0 {
                offsets.push(data_len);
                continue;
            }
            bucket_buf.resize((o1 - o0) as usize, 0);
            old_file
                .read_exact_at(&mut bucket_buf, HEAD_LEN as u64 + o0)
                .expect("read old bucket");
            let (tails, mut r) = decode_bucket_tails(&bucket_buf);
            let mut vals: Vec<Vec<u8>> = Vec::with_capacity(tails.len());
            for _ in 0..tails.len() {
                scratch.clear();
                decode_value(&t_old, &mut r, &mut scratch);
                // The shim transform: this is exactly what the production
                // reader serves under legacy-swapped-controls.
                swap_value_controls(&mut scratch);
                vals.push(scratch.clone());
            }
            count += tails.len() as u64;
            let bytes = encode_bucket(&t_new, &tails, &vals);
            // The invariant that catches any encoder drift immediately.
            assert_eq!(
                bytes.len() as u64,
                o1 - o0,
                "shard {shard:02x} bucket {b}: re-encoded length diverged"
            );
            out.write_all(&bytes).unwrap();
            data_len += bytes.len() as u64;
            offsets.push(data_len);
        }
        out.flush().unwrap();
        drop(out);

        assert_eq!(count, old_count, "shard {shard:02x}: entry count drifted");
        assert_eq!(
            data_len, old_data_len,
            "shard {shard:02x}: data length drifted"
        );

        let mut head: Vec<u8> = Vec::with_capacity(HEAD_LEN);
        head.extend(b"FRZTBL01");
        head.extend(count.to_le_bytes());
        head.extend(data_len.to_le_bytes());
        for &o in &offsets {
            head.extend(&o.to_le_bytes()[0..5]);
        }
        assert_eq!(head.len(), HEAD_LEN);
        file.write_at(&head, 0).unwrap();
        file.sync_all().unwrap();

        let old_sz = std::fs::metadata(&old_path).unwrap().len();
        let new_sz = std::fs::metadata(&new_path).unwrap().len();
        assert_eq!(old_sz, new_sz, "shard {shard:02x}: file size drifted");

        total.fetch_add(count, Ordering::Relaxed);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 32 == 0 || d == 256 {
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

fn stage_verify(old_dir: &str, new_dir: &str) {
    let t_old = std::sync::Arc::new(load_tables(&format!("{old_dir}/tables.bin")));
    let t_new = std::sync::Arc::new(load_tables(&format!("{new_dir}/tables.bin")));
    let mismatches = AtomicU64::new(0);
    let checked = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    (0..256usize).into_par_iter().for_each(|shard| {
        let old_file =
            std::fs::File::open(format!("{old_dir}/shard_{shard:02x}.frz")).expect("old shard");
        let new_file =
            std::fs::File::open(format!("{new_dir}/shard_{shard:02x}.frz")).expect("new shard");
        let (old_count, _, old_head) = load_shard_header(&old_file);
        let (new_count, _, new_head) = load_shard_header(&new_file);
        let mut bad = 0u64;
        let mut n = 0u64;
        if old_count != new_count {
            bad += 1;
        }
        let mut ob: Vec<u8> = Vec::new();
        let mut nb: Vec<u8> = Vec::new();
        let mut ov: Vec<u8> = Vec::new();
        let mut nv: Vec<u8> = Vec::new();
        for b in 0..BUCKETS {
            let (oo0, oo1) = (read_off(&old_head, b), read_off(&old_head, b + 1));
            let (no0, no1) = (read_off(&new_head, b), read_off(&new_head, b + 1));
            if oo1 - oo0 != no1 - no0 {
                bad += 1;
                continue;
            }
            if oo1 == oo0 {
                continue;
            }
            ob.resize((oo1 - oo0) as usize, 0);
            nb.resize((no1 - no0) as usize, 0);
            old_file
                .read_exact_at(&mut ob, HEAD_LEN as u64 + oo0)
                .unwrap();
            new_file
                .read_exact_at(&mut nb, HEAD_LEN as u64 + no0)
                .unwrap();
            let (otails, mut or) = decode_bucket_tails(&ob);
            let (ntails, mut nr) = decode_bucket_tails(&nb);
            if otails != ntails {
                bad += 1;
                continue;
            }
            for _ in 0..otails.len() {
                ov.clear();
                nv.clear();
                decode_value(&t_old, &mut or, &mut ov);
                swap_value_controls(&mut ov); // what the shim serves today
                decode_value(&t_new, &mut nr, &mut nv);
                if ov != nv {
                    bad += 1;
                }
                n += 1;
            }
        }
        checked.fetch_add(n, Ordering::Relaxed);
        mismatches.fetch_add(bad, Ordering::Relaxed);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 32 == 0 || d == 256 {
            println!(
                "[verify] shards {}/256 checked={} mismatches={}",
                d,
                checked.load(Ordering::Relaxed),
                mismatches.load(Ordering::Relaxed)
            );
        }
    });
    let bad = mismatches.load(Ordering::Relaxed);
    println!(
        "verify done: checked={} mismatches={} -> {}",
        checked.load(Ordering::Relaxed),
        bad,
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    if bad != 0 {
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("tables") => stage_tables(&args[2], &args[3]),
        Some("write") => stage_write(&args[2], &args[3]),
        Some("verify") => stage_verify(&args[2], &args[3]),
        _ => {
            eprintln!("usage: frozen_reencode tables|write|verify <old_dir> <new_dir>");
            std::process::exit(2);
        }
    }
}
