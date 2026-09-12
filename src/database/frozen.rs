//! Frozen-table read path: a static, compressed point-lookup store used
//! directly by the replacement runtime.
//!
//! Runtime configuration is environment-driven:
//!   FROZEN_DB_DIR=<dir>       serve regular-shard lookups from <dir>
//!   FROZEN_CURATED_DIR=<dir>  serve curated-shard lookups from <dir>
//!   FROZEN_FILTER=1           also load <dir>/filters.bin (~25.5 GB RAM;
//!                             makes misses ~0.5us instead of one disk read)
//!
//! Layout (written by the superdb `frozen_table` builder, validated
//! byte-exact against the rocks source): 256 shard_XX.frz files, each
//! [magic][entry count][data len][2^20+1 x u40 bucket offsets][data];
//! bucket = Elias-Fano sorted 48-bit key tails + canonical-Huffman values
//! (context = (circuit width, gate index), symbol = whole gate triple).
//! `get` returns the exact legacy value bytes (length-prefixed 3-byte-gate
//! blobs), so consumers parse results without backend-specific decoding.

use std::os::unix::fs::FileExt;
use xorf::{BinaryFuse8, Filter};

const BUCKETS: usize = 1 << 20;
const GI_CLAMP: u32 = 11;
const ESC: u32 = u32::MAX;
const MAXLEN: usize = 40;

// ------------------------------------------------------------- bit reader
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
        let v = self.peek(bits);
        self.consume(bits);
        v
    }
    #[inline]
    fn get1(&mut self) -> u64 {
        self.get(1)
    }
    /// Refill so at least `bits` are buffered and return them without
    /// consuming. Reads past the end of the buffer yield zero bits, exactly
    /// like `get` always has.
    #[inline]
    fn peek(&mut self, bits: u32) -> u64 {
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
        self.acc & ((1u64 << bits) - 1)
    }
    /// Drop `bits` already buffered by a preceding `peek`.
    #[inline]
    fn consume(&mut self, bits: u32) {
        debug_assert!(self.nbits >= bits);
        self.acc >>= bits;
        self.nbits -= bits;
    }
    /// Advance the stream by `bits` without materializing values.
    fn skip(&mut self, mut bits: u64) {
        let take = bits.min(self.nbits as u64) as u32;
        self.consume(take);
        bits -= take as u64;
        self.pos += (bits / 8) as usize;
        let rem = (bits % 8) as u32;
        if rem > 0 {
            let _ = self.get(rem);
        }
    }
}

// ------------------------------------------------------ canonical Huffman
/// First-level decode LUT width: one `peek` of this many stream bits resolves
/// every code of length <= LUT_BITS in a single table hit; longer codes fall
/// back to the canonical per-length walk.
const LUT_BITS: u32 = 12;

#[derive(Default)]
struct HuffTable {
    syms: Vec<u32>,
    first_code: Vec<u64>,
    first_idx: Vec<usize>,
    count: Vec<usize>,
    single: bool,
    // lut[peek] = (symbol << 8) | code_len for codes of length <= LUT_BITS,
    // where `peek` holds the next LUT_BITS stream bits LSB-first (bit i of
    // `peek` is the i-th bit read; codes accumulate MSB-first, so bit i of the
    // code is stream bit code_len-1-i). Zero entry = no short code matches.
    lut: Vec<u64>,
}

fn rebuild_canonical(mut sym_lens: Vec<(u32, u32)>) -> HuffTable {
    let mut t = HuffTable::default();
    if sym_lens.is_empty() {
        return t;
    }
    if sym_lens.len() == 1 {
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
    t.lut = vec![0u64; 1 << LUT_BITS];
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
        t.syms.push(s);
        if l <= LUT_BITS {
            // Stream-order index: bit i of the peek is code bit l-1-i.
            let mut base = 0u64;
            for i in 0..l {
                base |= ((code >> (l - 1 - i)) & 1) << i;
            }
            let entry = ((s as u64) << 8) | l as u64;
            let mut fill = 0u64;
            while fill < (1u64 << (LUT_BITS - l)) {
                t.lut[(base | (fill << l)) as usize] = entry;
                fill += 1;
            }
        }
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
        let peek = r.peek(LUT_BITS);
        let entry = self.lut[peek as usize];
        if entry != 0 {
            r.consume((entry & 0xFF) as u32);
            return (entry >> 8) as u32;
        }
        // No code of length <= LUT_BITS matches these bits (the LUT covers all
        // extensions), so replay them into the canonical walk and continue.
        let mut code = 0u64;
        for i in 0..LUT_BITS {
            code = (code << 1) | ((peek >> i) & 1);
        }
        r.consume(LUT_BITS);
        for len in (LUT_BITS as usize + 1)..=MAXLEN {
            code = (code << 1) | r.get1();
            if self.count[len] > 0 {
                let fc = self.first_code[len];
                if code >= fc && code < fc + self.count[len] as u64 {
                    return self.syms[self.first_idx[len] + (code - fc) as usize];
                }
            }
        }
        panic!("frozen: corrupt huffman stream");
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
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("frozen: read {path}: {e}"));
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

// Decode a stored value into `out` as a chain of length-prefixed circuits
// ([len][len bytes]...). Each loop iteration emits exactly one circuit, so
// `max_circuits` bounds the decode: usize::MAX reconstructs the whole value
// and a finite cap stops after that many circuits on a clean boundary. Normal
// runtime lookups are deliberately unbounded; finite caps remain useful to
// diagnostic callers and focused decoder tests.
fn decode_value(t: &Tables, r: &mut BitReader, out: &mut Vec<u8>, max_circuits: usize) {
    let mut circuits = 0usize;
    loop {
        let hs = t.header.decode(r);
        let (g, w, chain);
        if hs == ESC {
            let ge = r.get(8) as u32;
            if ge == 121 {
                // Raw block: the value's bytes verbatim, themselves a chain of
                // length-prefixed circuits. Walk them so a bounded decode can
                // stop on a circuit boundary; usize::MAX reads all `len` bytes,
                // bit-for-bit as the original `for _ in 0..len` did.
                let len = r.get(16) as usize;
                // Unbounded, this branch emits exactly `len` bytes (one length
                // prefix plus its payload, repeatedly). Reserving consumes no
                // bits and writes no byte, so the decoded sequence and the
                // reader position are unchanged.
                out.reserve(len);
                let mut read = 0usize;
                while read < len {
                    let l = r.get(8) as u8;
                    read += 1;
                    out.push(l);
                    let take = (l as usize).min(len - read);
                    for _ in 0..take {
                        out.push(r.get(8) as u8);
                        read += 1;
                    }
                    circuits += 1;
                    if circuits >= max_circuits {
                        return;
                    }
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
        // Each circuit in the chain emits exactly 1 + 3g bytes.
        out.reserve(1 + 3 * g as usize);
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
        circuits += 1;
        if circuits >= max_circuits {
            return;
        }
        if chain == 0 {
            return;
        }
    }
}

/// Advance the reader past one stored value without materializing bytes.
/// Consumes exactly the bits `decode_value(.., usize::MAX)` would: predecessor
/// values in a bucket are decoded only to reach the target's bit offset.
fn skip_value(t: &Tables, r: &mut BitReader) {
    loop {
        let hs = t.header.decode(r);
        let (g, w, chain);
        if hs == ESC {
            let ge = r.get(8) as u32;
            if ge == 121 {
                let len = r.get(16) as u64;
                r.skip(len * 8);
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
        for gi in 0..g {
            let tab = &t.gates[ctx_of(w, gi)];
            let sym = tab.decode(r);
            if sym == ESC {
                if r.get1() == 1 {
                    r.skip(24);
                } else {
                    r.skip(15);
                }
            }
        }
        if chain == 0 {
            return;
        }
    }
}

// --------------------------------------------------------------- key math
#[inline]
pub fn split_key(k: &[u8]) -> (usize, u32, u64) {
    let hi = u64::from_be_bytes(k[0..8].try_into().unwrap());
    let lo = u64::from_be_bytes(k[8..16].try_into().unwrap());
    (
        (hi >> 56) as usize,
        ((hi >> 36) & 0xF_FFFF) as u32,
        ((hi & 0xF_FFFF_FFFF) << 12) | (lo >> 52),
    )
}
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
#[inline]
pub fn mix76(shard: usize, bucket: u32, tail: u64) -> u64 {
    splitmix64(tail ^ ((bucket as u64) << 44) ^ ((shard as u64) << 30))
}

// ---------------------------------------------------------------- store
struct FrozenShard {
    file: std::fs::File,
    // Raw on-disk u40 bucket offsets (5 bytes LE each), padded with 3 zero
    // bytes so any entry is one unaligned 8-byte load. Keeping them packed
    // saves ~3 MB resident per shard (~0.8 GB per store) versus Vec<u64> —
    // RAM that goes back to the page cache serving bucket preads.
    offs_raw: Vec<u8>,
    data_base: u64,
}

impl FrozenShard {
    #[inline]
    fn off(&self, i: usize) -> u64 {
        let p = i * 5;
        u64::from_le_bytes(self.offs_raw[p..p + 8].try_into().unwrap()) & 0xFF_FFFF_FFFF
    }
}

struct Frozen {
    shards: Vec<FrozenShard>,
    tables: Tables,
    filters: Option<Vec<BinaryFuse8>>,
    // Decoded-value cache for BIG-POOL keys (values with > 20 candidates —
    // only the M1/M2/M3 pool-swap entries qualify). Those ~1.7K hot keys are
    // hit constantly and their Huffman decode dominates lookup cost
    // (~40ms/move measured against ~7ms with the bounded store); caching the
    // decoded bytes once per key removes it. Bounded, never evicts (the
    // qualifying key population is small by construction).
    big_cache: std::sync::Mutex<std::collections::HashMap<[u8; 16], std::sync::Arc<Vec<u8>>>>,
    // Value convention: when true, every decoded gate triple [t, c1, c2] has
    // its two controls swapped (bytes 1 and 2) before the value is returned.
    // The historical pre-2ed0222a curated store used the swapped-controls
    // convention (its keys were canonicalized with the b/c-swapped polynomial
    // of pre-2ed0222a, so its values read back swapped relative to native) —
    // current native builds do not need this. This is the store-side half
    // of FROZEN_*_VALUE_CONVENTION for historical store compatibility.
    swap_ctrls: bool,
}

/// A QC lookup was incomplete; this must not be reported as a missing DB key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QcLookupLimit {
    BucketBytes { bytes: u64, limit: usize },
    DecodeBytes { limit: usize },
    InvalidData(&'static str),
    ReadFailure,
}

// QC has its own strict reader: the ordinary sampler keeps its established
// decoder/cache path. EOF is an error here, and decoded bytes (including
// skipped predecessor values) share one budget, even for zero-bit symbols.
struct QcReader<'a> {
    bytes: &'a [u8],
    bit: usize,
    decoded: usize,
    limit: usize,
}

impl<'a> QcReader<'a> {
    fn new(bytes: &'a [u8], limit: usize) -> Self {
        Self {
            bytes,
            bit: 0,
            decoded: 0,
            limit,
        }
    }

    fn get(&mut self, bits: usize) -> Result<u64, QcLookupLimit> {
        if bits > 63 || bits > self.bytes.len().saturating_mul(8).saturating_sub(self.bit) {
            return Err(QcLookupLimit::InvalidData("truncated bit stream"));
        }
        let mut value = 0;
        for shift in 0..bits {
            value |= (((self.bytes[self.bit / 8] >> (self.bit % 8)) & 1) as u64) << shift;
            self.bit += 1;
        }
        Ok(value)
    }

    fn skip(&mut self, bits: usize) -> Result<(), QcLookupLimit> {
        if bits > self.bytes.len().saturating_mul(8).saturating_sub(self.bit) {
            return Err(QcLookupLimit::InvalidData("truncated bit stream"));
        }
        self.bit += bits;
        Ok(())
    }

    fn charge(&mut self, bytes: usize) -> Result<(), QcLookupLimit> {
        if bytes > self.limit.saturating_sub(self.decoded) {
            return Err(QcLookupLimit::DecodeBytes { limit: self.limit });
        }
        self.decoded += bytes;
        Ok(())
    }

    fn symbol(&mut self, table: &HuffTable) -> Result<u32, QcLookupLimit> {
        if table.single {
            return table
                .syms
                .first()
                .copied()
                .ok_or(QcLookupLimit::InvalidData("empty single-symbol table"));
        }
        let mut code = 0u64;
        for len in 1..=MAXLEN {
            code = (code << 1) | self.get(1)?;
            let count = table.count.get(len).copied().unwrap_or(0);
            if count > 0 {
                let first = table.first_code[len];
                if code >= first && code - first < count as u64 {
                    return table
                        .syms
                        .get(table.first_idx[len] + (code - first) as usize)
                        .copied()
                        .ok_or(QcLookupLimit::InvalidData("invalid Huffman index"));
                }
            }
        }
        Err(QcLookupLimit::InvalidData("invalid Huffman symbol"))
    }
}

fn qc_cached_prefix(value: &[u8], cap: usize, byte_limit: usize) -> Result<Vec<u8>, QcLookupLimit> {
    let mut end = 0usize;
    for _ in 0..cap {
        if end == value.len() {
            break;
        }
        let len = value[end] as usize;
        if len % 3 != 0 || len + 1 > value.len() - end {
            return Err(QcLookupLimit::InvalidData("invalid cached circuit length"));
        }
        if len + 1 > byte_limit.saturating_sub(end) {
            return Err(QcLookupLimit::DecodeBytes { limit: byte_limit });
        }
        end += len + 1;
    }
    Ok(value[..end].to_vec())
}

fn decode_value_qc(
    tables: &Tables,
    reader: &mut QcReader<'_>,
    cap: usize,
    materialize: bool,
) -> Result<Vec<u8>, QcLookupLimit> {
    let mut output = Vec::new();
    let mut circuits = 0;
    while circuits < cap {
        let header = reader.symbol(&tables.header)?;
        let (gates, width, chain) = if header == ESC {
            let gates = reader.get(8)? as usize;
            if gates == 121 {
                let mut remaining = reader.get(16)? as usize;
                if !materialize {
                    reader.charge(remaining)?;
                    reader.skip(remaining * 8)?;
                    return Ok(output);
                }
                // Raw values can contain many circuits. Never reserve their
                // entire advertised length when only a prefix was requested.
                while remaining > 0 {
                    let len = reader.get(8)? as usize;
                    if len % 3 != 0 || len + 1 > remaining {
                        return Err(QcLookupLimit::InvalidData("invalid raw circuit length"));
                    }
                    reader.charge(len + 1)?;
                    output.push(len as u8);
                    for _ in 0..len {
                        output.push(reader.get(8)? as u8);
                    }
                    remaining -= len + 1;
                    circuits += 1;
                    if circuits >= cap {
                        return Ok(output);
                    }
                }
                return Ok(output);
            }
            (gates, reader.get(8)? as u32, reader.get(1)? != 0)
        } else {
            (
                (header >> 7) as usize,
                (header >> 1) & 0x3f,
                header & 1 != 0,
            )
        };
        if gates > 85 {
            return Err(QcLookupLimit::InvalidData(
                "circuit exceeds one-byte length",
            ));
        }
        reader.charge(1 + 3 * gates)?;
        if materialize {
            output.push((3 * gates) as u8);
        }
        for index in 0..gates {
            let table = tables
                .gates
                .get(ctx_of(width, index as u32))
                .ok_or(QcLookupLimit::InvalidData("missing gate context"))?;
            let symbol = reader.symbol(table)?;
            let triple = if symbol == ESC {
                if reader.get(1)? != 0 {
                    [
                        reader.get(8)? as u8,
                        reader.get(8)? as u8,
                        reader.get(8)? as u8,
                    ]
                } else {
                    let packed = reader.get(15)? as u32;
                    [
                        (packed >> 10) as u8,
                        ((packed >> 5) & 31) as u8,
                        (packed & 31) as u8,
                    ]
                }
            } else {
                [
                    ((symbol >> 10) & 31) as u8,
                    ((symbol >> 5) & 31) as u8,
                    (symbol & 31) as u8,
                ]
            };
            if materialize {
                output.extend_from_slice(&triple);
            }
        }
        circuits += 1;
        if !chain {
            return Ok(output);
        }
    }
    Ok(output)
}

/// Swap the two controls of every gate in a legacy value chain
/// (`[len][len bytes]*`, gates are 3-byte triples `[t, c1, c2]`).
fn swap_value_controls(v: &mut [u8]) {
    let mut pos = 0usize;
    while pos < v.len() {
        let len = v[pos] as usize;
        pos += 1;
        debug_assert_eq!(len % 3, 0, "frozen: value chunk not gate-aligned");
        let end = (pos + len).min(v.len());
        while pos + 3 <= end {
            v.swap(pos + 1, pos + 2);
            pos += 3;
        }
        pos = end;
    }
}

#[derive(bincode2::Decode)]
#[bincode(crate = "bincode2")]
struct FiltersFile {
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

impl Frozen {
    fn open(dir: &str, swap_ctrls: bool, filters_setting: FilterSetting) -> Frozen {
        let tables = load_tables(&format!("{dir}/tables.bin"));
        let head_len = 24usize + (BUCKETS + 1) * 5;
        let mut shards = Vec::with_capacity(256);
        let mut table_entry_count = 0u64;
        for s in 0..256usize {
            let path = format!("{dir}/shard_{s:02x}.frz");
            let file =
                std::fs::File::open(&path).unwrap_or_else(|e| panic!("frozen: open {path}: {e}"));
            let mut head = vec![0u8; head_len];
            file.read_exact_at(&mut head, 0)
                .expect("frozen: shard header");
            assert_eq!(&head[0..8], b"FRZTBL01", "frozen: bad magic in {path}");
            let shard_entries = u64::from_le_bytes(head[8..16].try_into().unwrap());
            table_entry_count = table_entry_count
                .checked_add(shard_entries)
                .expect("frozen: total shard entry count overflow");
            let mut offs_raw = head.split_off(24);
            offs_raw.extend_from_slice(&[0u8; 3]);
            shards.push(FrozenShard {
                file,
                offs_raw,
                data_base: head_len as u64,
            });
        }
        // Optional in-RAM miss filter (~25.5 GB). Opt-in per process.
        let filters = if filters_setting.enabled() {
            let path = format!("{dir}/filters.bin");
            match std::fs::File::open(&path) {
                Ok(f) => {
                    let t0 = std::time::Instant::now();
                    let mut r = std::io::BufReader::with_capacity(8 << 20, f);
                    let ff: FiltersFile =
                        bincode2::decode_from_std_read(&mut r, bincode2::config::standard())
                            .expect("frozen: filters.bin decode");
                    assert_eq!(
                        ff.filters.len(),
                        256,
                        "frozen: filters.bin has {} shard filters, expected 256",
                        ff.filters.len()
                    );
                    assert_eq!(
                        ff.table_entry_count, table_entry_count,
                        "frozen: filters.bin entry count {} does not match frozen shard total {}",
                        ff.table_entry_count, table_entry_count
                    );
                    eprintln!(
                        "[frozen] filters.bin loaded in {:.1}s",
                        t0.elapsed().as_secs_f64()
                    );
                    Some(ff.filters)
                }
                Err(e) => {
                    eprintln!(
                        "[frozen] FROZEN_FILTER=1 but no filters.bin ({e}); running unfiltered"
                    );
                    None
                }
            }
        } else {
            None
        };
        Frozen {
            shards,
            tables,
            filters,
            swap_ctrls,
            big_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Exact point lookup; returns legacy value bytes, byte-identical to the
    /// source replacement value for this key.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.get_capped(key, usize::MAX)
    }

    fn get_qc(
        &self,
        key: &[u8; 16],
        cap: usize,
        max_bucket_bytes: usize,
    ) -> Result<Option<Vec<u8>>, QcLookupLimit> {
        if let Some(value) = self.big_cache.lock().unwrap().get(key) {
            // Cached values already have the convention fix-up. Copy only the
            // requested records, without first cloning the potentially huge pool.
            return qc_cached_prefix(value, cap, max_bucket_bytes).map(Some);
        }
        let (shard, bucket, tail) = split_key(key);
        if let Some(filters) = &self.filters {
            if !filters[shard].contains(&mix76(shard, bucket, tail)) {
                return Ok(None);
            }
        }
        let shard = self
            .shards
            .get(shard)
            .ok_or(QcLookupLimit::InvalidData("missing shard"))?;
        let first_offset = shard.off(bucket as usize);
        let end_offset = shard.off(bucket as usize + 1);
        let bytes = end_offset
            .checked_sub(first_offset)
            .ok_or(QcLookupLimit::InvalidData("decreasing bucket offsets"))?;
        if bytes == 0 {
            return Ok(None);
        }
        if bytes > max_bucket_bytes as u64 {
            return Err(QcLookupLimit::BucketBytes {
                bytes,
                limit: max_bucket_bytes,
            });
        }
        let file_offset = shard
            .data_base
            .checked_add(first_offset)
            .ok_or(QcLookupLimit::InvalidData("bucket offset overflow"))?;
        let mut buffer = vec![0; bytes as usize];
        shard
            .file
            .read_exact_at(&mut buffer, file_offset)
            .map_err(|_| QcLookupLimit::ReadFailure)?;
        let mut reader = QcReader::new(&buffer, max_bucket_bytes);
        let count = reader.get(16)? as usize;
        let low_bits = reader.get(6)? as usize;
        let target_upper = tail >> low_bits;
        let target_low = if low_bits == 0 {
            0
        } else {
            tail & ((1u64 << low_bits) - 1)
        };
        let (mut first, mut last, mut upper) = (None, 0, 0);
        for index in 0..count {
            while reader.get(1)? == 0 {
                upper += 1;
            }
            if upper == target_upper {
                first.get_or_insert(index);
                last = index;
            }
        }
        let Some(first) = first else {
            return Ok(None);
        };
        let index = if low_bits == 0 {
            first
        } else {
            reader.skip(first * low_bits)?;
            let mut found = None;
            for index in first..=last {
                if reader.get(low_bits)? == target_low {
                    found = Some(index);
                    break;
                }
            }
            let Some(found) = found else {
                return Ok(None);
            };
            reader.skip((count - 1 - found) * low_bits)?;
            found
        };
        for _ in 0..index {
            decode_value_qc(&self.tables, &mut reader, usize::MAX, false)?;
        }
        let mut value = decode_value_qc(&self.tables, &mut reader, cap, true)?;
        if self.swap_ctrls {
            swap_value_controls(&mut value);
        }
        // A bounded prefix must never replace an unbounded sampler cache value.
        Ok(Some(value))
    }

    /// Like `get`, but decodes at most `cap` circuits of the target value.
    /// Predecessor values in the same bucket are still decoded in full -- that
    /// is how the bit-reader advances to the target -- but the 20-bit bucket
    /// index makes buckets sparse (idx is almost always 0), so in practice only
    /// the target is decoded. Production lookups pass `usize::MAX`; this helper
    /// is retained for diagnostics that intentionally inspect only a prefix.
    fn get_capped(&self, key: &[u8], cap: usize) -> Option<Vec<u8>> {
        debug_assert_eq!(key.len(), 16);
        let key16: [u8; 16] = key.try_into().ok()?;
        if let Some(v) = self.big_cache.lock().unwrap().get(&key16) {
            return Some((**v).clone());
        }
        let (shard, bucket, tail) = split_key(key);
        if let Some(filters) = &self.filters {
            if !filters[shard].contains(&mix76(shard, bucket, tail)) {
                return None;
            }
        }
        let sh = &self.shards[shard];
        let o0 = sh.off(bucket as usize);
        let o1 = sh.off(bucket as usize + 1);
        if o0 == o1 {
            return None;
        }
        // Reuse a per-thread bucket buffer: probes are frequent and buckets
        // average ~1 KB, so a fresh allocation per probe is pure churn.
        thread_local! {
            static BUCKET_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        BUCKET_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.resize((o1 - o0) as usize, 0);
            sh.file.read_exact_at(&mut buf, sh.data_base + o0).ok()?;
            let mut r = BitReader::new(&buf);
            let n = r.get(16) as usize;
            let l = r.get(6) as u32;
            // Elias-Fano scan without materializing all uppers: the uppers are
            // nondecreasing, so entries matching the target's upper bits form
            // one contiguous index range [first, last].
            let target_up = tail >> l;
            let low_mask = if l == 0 { 0 } else { (1u64 << l) - 1 };
            let target_low = tail & low_mask;
            let mut first: Option<usize> = None;
            let mut last = 0usize;
            let mut up = 0u64;
            for i in 0..n {
                while r.get1() == 0 {
                    up += 1;
                }
                if up == target_up {
                    if first.is_none() {
                        first = Some(i);
                    }
                    last = i;
                }
            }
            let first = first?;
            // Lows are fixed-width: jump straight to the candidate range and
            // compare only those entries. The first match wins, as before.
            let mut idx: Option<usize> = None;
            if l == 0 {
                idx = Some(first);
            } else {
                r.skip(first as u64 * l as u64);
                for i in first..=last {
                    let low = r.get(l);
                    if low == target_low {
                        idx = Some(i);
                        break;
                    }
                }
                let idx = idx?;
                // Position the reader at the values region (after all n lows).
                r.skip((n - 1 - idx) as u64 * l as u64);
            }
            let idx = idx?;
            for _ in 0..idx {
                skip_value(&self.tables, &mut r);
            }
            let mut out = Vec::new();
            decode_value(&self.tables, &mut r, &mut out, cap);
            // Store-side convention fix-up at the single choke point:
            // covers all four emission paths (incl. verbatim raw blocks).
            if self.swap_ctrls {
                swap_value_controls(&mut out);
            }
            // BIG-POOL CACHE: values carrying more than the historical bounded
            // contract (20 candidates) are the M1..M4 pool entries of a
            // pool-swapped store. Those few hot keys are hit constantly and
            // their Huffman decode dominates lookup cost (measured 39ms/move
            // uncached against ~7ms cached, and unbounded decode makes that
            // worse); cache the decoded bytes once per key. Bounded at 4096
            // keys and only ever populated by values no ordinary entry
            // reaches, so a normal store never touches it. Cached AFTER the
            // convention fix-up, and only for a full (uncapped) decode.
            if cap == usize::MAX {
                let mut cands = 0usize;
                let mut pos = 0usize;
                while pos < out.len() && cands <= 20 {
                    cands += 1;
                    pos += 1 + out[pos] as usize;
                }
                if cands > 20 {
                    let mut c = self.big_cache.lock().unwrap();
                    if c.len() < 4096 {
                        c.insert(key16, std::sync::Arc::new(out.clone()));
                    }
                }
            }
            Some(out)
        })
    }
}

/// Diagnostic sequential scan: decode every entry value in one shard, calling
/// `f(value)` with the legacy value bytes for each. Walks the same bucket
/// layout `get` point-reads, needing only the shard file and tables.bin, so
/// census tools (degree/gate-count histograms) can sample the store without
/// keys or filters. Shards partition keys by hash, so any one shard is an
/// unbiased ~1/256 sample of the whole store.
pub fn scan_shard(dir: &str, shard: usize, f: &mut dyn FnMut(&[u8])) {
    let tables = load_tables(&format!("{dir}/tables.bin"));
    let head_len = 24usize + (BUCKETS + 1) * 5;
    let path = format!("{dir}/shard_{shard:02x}.frz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("frozen: open {path}: {e}"));
    let mut head = vec![0u8; head_len];
    file.read_exact_at(&mut head, 0)
        .expect("frozen: shard header");
    assert_eq!(&head[0..8], b"FRZTBL01", "frozen: bad magic in {path}");
    let mut offs = Vec::with_capacity(BUCKETS + 1);
    for i in 0..=BUCKETS {
        let mut b = [0u8; 8];
        b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
        offs.push(u64::from_le_bytes(b));
    }
    let data_base = head_len as u64;
    let mut buf = Vec::new();
    let mut out = Vec::new();
    for bkt in 0..BUCKETS {
        let (o0, o1) = (offs[bkt], offs[bkt + 1]);
        if o0 == o1 {
            continue;
        }
        buf.clear();
        buf.resize((o1 - o0) as usize, 0);
        file.read_exact_at(&mut buf, data_base + o0)
            .expect("frozen: bucket read");
        let mut r = BitReader::new(&buf);
        let n = r.get(16) as usize;
        let l = r.get(6) as u32;
        for _ in 0..n {
            while r.get1() == 0 {}
        }
        if l > 0 {
            for _ in 0..n {
                r.get(l);
            }
        }
        for _ in 0..n {
            out.clear();
            decode_value(&tables, &mut r, &mut out, usize::MAX);
            f(&out);
        }
    }
}

/// As [`scan_shard`], but also yields each entry's bucket index and 48-bit
/// key tail, reconstructing the Elias-Fano uppers the value-only scan skips.
/// With the shard index these form the store's full 76 key bits, enough to
/// test membership against another store via `mix76`.
pub fn scan_shard_entries(dir: &str, shard: usize, f: &mut dyn FnMut(u32, u64, &[u8])) {
    let tables = load_tables(&format!("{dir}/tables.bin"));
    let head_len = 24usize + (BUCKETS + 1) * 5;
    let path = format!("{dir}/shard_{shard:02x}.frz");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("frozen: open {path}: {e}"));
    let mut head = vec![0u8; head_len];
    file.read_exact_at(&mut head, 0)
        .expect("frozen: shard header");
    assert_eq!(&head[0..8], b"FRZTBL01", "frozen: bad magic in {path}");
    let mut offs = Vec::with_capacity(BUCKETS + 1);
    for i in 0..=BUCKETS {
        let mut b = [0u8; 8];
        b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
        offs.push(u64::from_le_bytes(b));
    }
    let data_base = head_len as u64;
    let mut buf = Vec::new();
    let mut out = Vec::new();
    let mut tails: Vec<u64> = Vec::new();
    for bkt in 0..BUCKETS {
        let (o0, o1) = (offs[bkt], offs[bkt + 1]);
        if o0 == o1 {
            continue;
        }
        buf.clear();
        buf.resize((o1 - o0) as usize, 0);
        file.read_exact_at(&mut buf, data_base + o0)
            .expect("frozen: bucket read");
        let mut r = BitReader::new(&buf);
        let n = r.get(16) as usize;
        let l = r.get(6) as u32;
        tails.clear();
        let mut up = 0u64;
        for _ in 0..n {
            while r.get1() == 0 {
                up += 1;
            }
            tails.push(up << l);
        }
        if l > 0 {
            for tail in tails.iter_mut() {
                *tail |= r.get(l);
            }
        }
        for tail in &tails {
            out.clear();
            decode_value(&tables, &mut r, &mut out, usize::MAX);
            f(bkt as u32, *tail, &out);
        }
    }
}

/// Interpretation of the two stored G57 control columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ValueConvention {
    #[default]
    Native,
    LegacySwappedControls,
}

impl ValueConvention {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LegacySwappedControls => "legacy-swapped-controls",
        }
    }
}

/// Resolved immutable-store decoding options; defaults use native values and no filters.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrozenOpenOptions {
    pub regular_convention: ValueConvention,
    pub curated_convention: ValueConvention,
    pub load_filters: bool,
}

#[derive(Clone, Copy)]
enum FilterSetting {
    LegacyEnvironment,
    Resolved(bool),
}

impl FilterSetting {
    fn enabled(self) -> bool {
        match self {
            Self::LegacyEnvironment => super::legacy_environment::filters_enabled(),
            Self::Resolved(enabled) => enabled,
        }
    }
}

/// Native runtime handle for the immutable replacement stores.
///
/// `open`/`from_env` require the regular store. The curated store is optional
/// so commands that only compress against the regular table do not need to
/// open it. (Deviation from the shuffletests original: `regular` is an Option
/// so `FrozenDb::empty()` can model this branch's legacy DB-less modes — the
/// paths that used to pass empty shard-db slices, e.g. the unsamf-cycle tests
/// and interleave's no-lookup arm.)
pub struct FrozenDb {
    regular: Option<Frozen>,
    curated: Option<Frozen>,
}

impl FrozenDb {
    /// Open explicit directories with historical environment overrides.
    /// See `open_with_options` for a fully resolved storage boundary.
    pub fn open(regular_dir: &str, curated_dir: Option<&str>) -> Self {
        let regular_convention =
            super::legacy_environment::value_convention("FROZEN_REGULAR_VALUE_CONVENTION");
        let curated_convention =
            super::legacy_environment::value_convention("FROZEN_CURATED_VALUE_CONVENTION");
        Self::open_resolved(
            regular_dir,
            curated_dir,
            regular_convention,
            curated_convention,
            FilterSetting::LegacyEnvironment,
        )
    }

    /// Open immutable stores without consulting environment configuration.
    pub fn open_with_options(
        regular_dir: &str,
        curated_dir: Option<&str>,
        options: FrozenOpenOptions,
    ) -> Self {
        Self::open_resolved(
            regular_dir,
            curated_dir,
            options.regular_convention,
            options.curated_convention,
            FilterSetting::Resolved(options.load_filters),
        )
    }

    fn open_resolved(
        regular_dir: &str,
        curated_dir: Option<&str>,
        regular_convention: ValueConvention,
        curated_convention: ValueConvention,
        filters: FilterSetting,
    ) -> Self {
        let regular = Some(Self::open_store(
            "regular",
            regular_dir,
            regular_convention == ValueConvention::LegacySwappedControls,
            filters,
        ));
        let curated = curated_dir.map(|dir| {
            Self::open_store(
                "curated",
                dir,
                curated_convention == ValueConvention::LegacySwappedControls,
                filters,
            )
        });
        let reg_name = regular_convention.as_str();
        let cur_name = curated_convention.as_str();
        eprintln!(
            "[frozen] value conventions: regular={reg_name}, curated={}",
            if curated.is_some() { cur_name } else { "-" }
        );
        Self { regular, curated }
    }

    /// A handle with no stores: every lookup misses. Stands in for the legacy
    /// empty-shard-slice convention where a caller ran without any DB.
    pub fn empty() -> Self {
        Self {
            regular: None,
            curated: None,
        }
    }

    /// Open stores from `FROZEN_DB_DIR` and optional `FROZEN_CURATED_DIR`.
    /// The regular store is required because there is no legacy fallback.
    pub fn from_env() -> Self {
        let (regular, curated) = super::legacy_environment::frozen_directories();
        Self::open(&regular, curated.as_deref())
    }

    fn open_store(label: &str, dir: &str, swap_ctrls: bool, filters: FilterSetting) -> Frozen {
        let t0 = std::time::Instant::now();
        let store = Frozen::open(dir, swap_ctrls, filters);
        eprintln!(
            "[frozen] {label}={dir} opened in {:.1}s (filter {})",
            t0.elapsed().as_secs_f64(),
            if store.filters.is_some() { "on" } else { "off" }
        );
        store
    }

    #[inline]
    pub fn get_regular(&self, key: &[u8; 16]) -> Option<Vec<u8>> {
        self.regular.as_ref()?.get(key)
    }

    #[inline]
    pub fn get_curated(&self, key: &[u8; 16]) -> Option<Vec<u8>> {
        self.curated.as_ref()?.get(key)
    }

    /// QC-only prefix lookup. Bounds both the compressed bucket allocation and
    /// decoded legacy bytes examined (including predecessor values) by
    /// `max_bucket_bytes`. Returns at most `max_candidates` complete records,
    /// also when the full value is already cached. Ask for one extra record to
    /// detect candidate truncation. Errors distinguish incomplete lookup from
    /// a missing key; ordinary sampler lookups retain their existing behavior.
    pub fn get_regular_qc(
        &self,
        key: &[u8; 16],
        max_candidates: usize,
        max_bucket_bytes: usize,
    ) -> Result<Option<Vec<u8>>, QcLookupLimit> {
        match &self.regular {
            Some(store) => store.get_qc(key, max_candidates, max_bucket_bytes),
            None => Ok(None),
        }
    }

    /// The curated-store counterpart to [`Self::get_regular_qc`].
    pub fn get_curated_qc(
        &self,
        key: &[u8; 16],
        max_candidates: usize,
        max_bucket_bytes: usize,
    ) -> Result<Option<Vec<u8>>, QcLookupLimit> {
        match &self.curated {
            Some(store) => store.get_qc(key, max_candidates, max_bucket_bytes),
            None => Ok(None),
        }
    }

    pub fn has_curated(&self) -> bool {
        self.curated.is_some()
    }
}

#[cfg(test)]
#[path = "../../tests/database/frozen/tests.rs"]
mod tests;
