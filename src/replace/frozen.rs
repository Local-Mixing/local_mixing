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

// ------------------------------------------------------ canonical Huffman
#[derive(Default)]
struct HuffTable {
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
    offs: Vec<u64>,
    data_base: u64,
}

struct Frozen {
    shards: Vec<FrozenShard>,
    tables: Tables,
    filters: Option<Vec<BinaryFuse8>>,
}

#[derive(bincode2::Decode)]
#[bincode(crate = "bincode2")]
struct FiltersFile {
    #[allow(dead_code)]
    table_entry_count: u64,
    filters: Vec<BinaryFuse8>,
}

impl Frozen {
    fn open(dir: &str) -> Frozen {
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
            let mut offs = Vec::with_capacity(BUCKETS + 1);
            for i in 0..=BUCKETS {
                let mut b = [0u8; 8];
                b[0..5].copy_from_slice(&head[24 + i * 5..24 + i * 5 + 5]);
                offs.push(u64::from_le_bytes(b));
            }
            shards.push(FrozenShard {
                file,
                offs,
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
        }
    }

    /// Exact point lookup; returns legacy value bytes, byte-identical to the
    /// source replacement value for this key.
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        debug_assert_eq!(key.len(), 16);
        let (shard, bucket, tail) = split_key(key);
        if let Some(filters) = &self.filters {
            if !filters[shard].contains(&mix76(shard, bucket, tail)) {
                return None;
            }
        }
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
            if ((uppers[i] << l) | low) == tail && idx.is_none() {
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
        let regular = Some(Self::open_store("regular", regular_dir));
        let curated = curated_dir.map(|dir| Self::open_store("curated", dir));
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

    fn open_store(label: &str, dir: &str) -> Frozen {
        let t0 = std::time::Instant::now();
        let store = Frozen::open(dir);
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
}
