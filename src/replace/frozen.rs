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
fn split_key(k: &[u8]) -> (usize, u32, u64) {
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
fn mix76(shard: usize, bucket: u32, tail: u64) -> u64 {
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
    // Value convention: when true, every decoded gate triple [t, c1, c2] has
    // its two controls swapped (bytes 1 and 2) before the value is returned.
    // The curated store was BUILT under the legacy swapped-controls
    // convention (its keys were canonicalized with the b/c-swapped polynomial
    // of pre-2ed0222a, so its values read back swapped relative to native) —
    // this is the store-side half of FROZEN_*_VALUE_CONVENTION.
    swap_ctrls: bool,
}

/// Per-store value convention, from `FROZEN_REGULAR_VALUE_CONVENTION` /
/// `FROZEN_CURATED_VALUE_CONVENTION`. `native` (default) returns values as
/// stored; `legacy-swapped-controls` swaps each gate's two controls at decode.
fn value_convention(var: &str) -> (&'static str, bool) {
    match std::env::var(var).as_deref() {
        Err(_) | Ok("native") => ("native", false),
        Ok("legacy-swapped-controls") => ("legacy-swapped-controls", true),
        Ok(other) => panic!(
            "{var}={other}: unknown value convention (native | legacy-swapped-controls)"
        ),
    }
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
    #[allow(dead_code)]
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

impl Frozen {
    fn open(dir: &str, swap_ctrls: bool) -> Frozen {
        let tables = load_tables(&format!("{dir}/tables.bin"));
        let head_len = 24usize + (BUCKETS + 1) * 5;
        let mut shards = Vec::with_capacity(256);
        for s in 0..256usize {
            let path = format!("{dir}/shard_{s:02x}.frz");
            let file =
                std::fs::File::open(&path).unwrap_or_else(|e| panic!("frozen: open {path}: {e}"));
            let mut head = vec![0u8; head_len];
            file.read_exact_at(&mut head, 0)
                .expect("frozen: shard header");
            assert_eq!(&head[0..8], b"FRZTBL01", "frozen: bad magic in {path}");
            let mut offs_raw = head.split_off(24);
            offs_raw.extend_from_slice(&[0u8; 3]);
            shards.push(FrozenShard {
                file,
                offs_raw,
                data_base: head_len as u64,
            });
        }
        // Optional in-RAM miss filter (~25.5 GB). Opt-in per process.
        let filters = if std::env::var("FROZEN_FILTER").map(|v| v == "1") == Ok(true) {
            let path = format!("{dir}/filters.bin");
            match std::fs::File::open(&path) {
                Ok(f) => {
                    let t0 = std::time::Instant::now();
                    let mut r = std::io::BufReader::with_capacity(8 << 20, f);
                    let ff: FiltersFile =
                        bincode2::decode_from_std_read(&mut r, bincode2::config::standard())
                            .expect("frozen: filters.bin decode");
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
        }
    }

    /// Exact point lookup; returns legacy value bytes, byte-identical to the
    /// source replacement value for this key.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.get_capped(key, usize::MAX)
    }

    /// Like `get`, but decodes at most `cap` circuits of the target value.
    /// Predecessor values in the same bucket are still decoded in full -- that
    /// is how the bit-reader advances to the target -- but the 20-bit bucket
    /// index makes buckets sparse (idx is almost always 0), so in practice only
    /// the target is decoded. Production lookups pass `usize::MAX`; this helper
    /// is retained for diagnostics that intentionally inspect only a prefix.
    fn get_capped(&self, key: &[u8], cap: usize) -> Option<Vec<u8>> {
        debug_assert_eq!(key.len(), 16);
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
    file.read_exact_at(&mut head, 0).expect("frozen: shard header");
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
    /// Open stores from explicit directories.
    pub fn open(regular_dir: &str, curated_dir: Option<&str>) -> Self {
        let (reg_name, reg_swap) = value_convention("FROZEN_REGULAR_VALUE_CONVENTION");
        let (cur_name, cur_swap) = value_convention("FROZEN_CURATED_VALUE_CONVENTION");
        let regular = Some(Self::open_store("regular", regular_dir, reg_swap));
        let curated = curated_dir.map(|dir| Self::open_store("curated", dir, cur_swap));
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
        let regular = std::env::var("FROZEN_DB_DIR")
            .expect("FROZEN_DB_DIR is required; the runtime is frozen-store only");
        let curated = std::env::var("FROZEN_CURATED_DIR").ok();
        Self::open(&regular, curated.as_deref())
    }

    fn open_store(label: &str, dir: &str, swap_ctrls: bool) -> Frozen {
        let t0 = std::time::Instant::now();
        let store = Frozen::open(dir, swap_ctrls);
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

    pub fn has_curated(&self) -> bool {
        self.curated.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_key_recovers_frozen_address_fields() {
        let shard = 0xabu64;
        let bucket = 0x54321u64;
        let tail = 0x1234_5678_9abcu64;
        let hi = (shard << 56) | (bucket << 36) | (tail >> 12);
        let lo = (tail & 0xfff) << 52;
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&hi.to_be_bytes());
        key[8..].copy_from_slice(&lo.to_be_bytes());

        assert_eq!(split_key(&key), (shard as usize, bucket as u32, tail));
    }

    #[test]
    fn runtime_handle_is_thread_shareable() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrozenDb>();
    }

    // legacy-swapped-controls: bytes 1 and 2 of every 3-byte gate swap, per
    // circuit chunk of the [len][len bytes]* value chain, targets untouched.
    #[test]
    fn swap_value_controls_swaps_per_gate_within_chunks() {
        let mut v = vec![
            6, /* two gates */ 10, 1, 2, 20, 3, 4, //
            3, /* one gate */ 30, 5, 6,
        ];
        swap_value_controls(&mut v);
        assert_eq!(v, vec![6, 10, 2, 1, 20, 4, 3, 3, 30, 6, 5]);
    }

    // Reference implementation of the pre-LUT canonical walk; the LUT fast
    // path must decode identical symbols and consume identical bit counts.
    fn reference_decode(t: &HuffTable, r: &mut BitReader) -> u32 {
        if t.single {
            return t.syms[0];
        }
        let mut code = 0u64;
        for len in 1..=MAXLEN {
            code = (code << 1) | r.get1();
            if t.count[len] > 0 {
                let fc = t.first_code[len];
                if code >= fc && code < fc + t.count[len] as u64 {
                    return t.syms[t.first_idx[len] + (code - fc) as usize];
                }
            }
        }
        panic!("corrupt reference stream");
    }

    struct BitWriter {
        bytes: Vec<u8>,
        acc: u64,
        nbits: u32,
    }
    impl BitWriter {
        fn new() -> Self {
            BitWriter {
                bytes: Vec::new(),
                acc: 0,
                nbits: 0,
            }
        }
        fn push_bit(&mut self, b: u64) {
            self.acc |= (b & 1) << self.nbits;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.acc as u8);
                self.acc = 0;
                self.nbits = 0;
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.nbits > 0 {
                self.bytes.push(self.acc as u8);
            }
            self.bytes
        }
    }

    #[test]
    fn opt_equiv_lut_decode_matches_reference_walk() {
        // Complete canonical codes, including lengths beyond LUT_BITS so the
        // fallback path is exercised.
        let mut deep: Vec<(u32, u32)> = (0..12u32).map(|i| (100 + i, i + 1)).collect();
        deep.push((900, 13));
        deep.push((901, 13));
        let alphabets: Vec<Vec<(u32, u32)>> = vec![
            vec![(7, 1), (8, 2), (9, 3), (10, 3)],
            vec![(1, 2), (2, 2), (3, 2), (4, 2)],
            deep,
        ];
        for sym_lens in alphabets {
            let table = rebuild_canonical(sym_lens.clone());
            // Recompute (code, len) per symbol exactly as rebuild_canonical does.
            let mut sorted = sym_lens.clone();
            sorted.sort_by_key(|&(s, l)| (l, s));
            let mut codes = Vec::new();
            let mut code = 0u64;
            let mut prev_len = sorted[0].1;
            for &(s, l) in &sorted {
                if l > prev_len {
                    code <<= l - prev_len;
                    prev_len = l;
                }
                codes.push((s, code, l));
                code += 1;
            }
            // Deterministic pseudo-random symbol sequence.
            let mut state = 0x1234_5678_9abc_def0u64;
            let mut seq = Vec::new();
            for _ in 0..500 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                seq.push(codes[(state >> 33) as usize % codes.len()]);
            }
            let mut w = BitWriter::new();
            for &(_, c, l) in &seq {
                for i in (0..l).rev() {
                    w.push_bit((c >> i) & 1);
                }
            }
            let stream = w.finish();
            let mut fast = BitReader::new(&stream);
            let mut refr = BitReader::new(&stream);
            for &(s, _, _) in &seq {
                assert_eq!(table.decode(&mut fast), s);
                assert_eq!(reference_decode(&table, &mut refr), s);
                // peek refills eagerly, so raw (pos, nbits) may differ while
                // the logical bit offset must not.
                assert_eq!(
                    fast.pos * 8 - fast.nbits as usize,
                    refr.pos * 8 - refr.nbits as usize,
                    "bit positions diverged"
                );
            }
        }
    }

    #[test]
    fn bounded_raw_decode_stops_on_a_circuit_boundary() {
        let tables = Tables {
            header: rebuild_canonical(vec![(ESC, 1)]),
            gates: Vec::new(),
        };
        // Raw marker, eight bytes, then two one-gate circuit chunks.
        let encoded = [121, 8, 0, 3, 10, 1, 2, 3, 20, 3, 4];
        let mut reader = BitReader::new(&encoded);
        let mut output = Vec::new();
        decode_value(&tables, &mut reader, &mut output, 1);
        assert_eq!(output, [3, 10, 1, 2]);
    }

    #[test]
    fn unbounded_raw_decode_returns_more_than_the_old_curated_cap() {
        let tables = Tables {
            header: rebuild_canonical(vec![(ESC, 1)]),
            gates: Vec::new(),
        };
        let mut value = Vec::new();
        for i in 0..300u16 {
            value.extend_from_slice(&[3, (i & 0xff) as u8, 1, 2]);
        }
        let mut encoded = Vec::with_capacity(value.len() + 3);
        encoded.push(121);
        encoded.extend_from_slice(&(value.len() as u16).to_le_bytes());
        encoded.extend_from_slice(&value);

        let mut reader = BitReader::new(&encoded);
        let mut output = Vec::new();
        decode_value(&tables, &mut reader, &mut output, usize::MAX);
        assert_eq!(output, value);
        assert_eq!(output.chunks_exact(4).len(), 300);
    }
}
