//! The deliverable: the `anf_dump_eval.py` body (byte for byte), container
//! v1 around it, CRC-32, a validating flat-arena decoder and its evaluator
//! (docs/STAGE4_RUST_DESIGN.md §4.12, §6).
//!
//! Body = concatenation over windows of `<I len>` + blob, where blob is
//!
//! ```text
//! <HH f c>            f = support size (wires in some monomial), c = changed wires
//! <H w> * f           support, ascending
//! per changed wire, ascending:  <HI w count>  then count masks of ceil(max(f,1)/8)
//!                     bytes, little-endian, over LOCAL support indices,
//!                     monomials in ascending global value
//! ```
//!
//! Windows are `[0] + sorted(0 < b < m) + [m]` (no dedup) and each comes from
//! `window_anf(…, EMIT_BUDGET)`. The oracle uses budget 1<<30, where no cap
//! fires; the port's ceiling is lower (a resource bound), and since a cap only
//! ever aborts, every window it emits is the oracle's canonical one; a window
//! above it fails with `OverBudget` (exit 6).
use super::anf::{MulStats, WindowAnf, window_anf};
use super::mono::{MonoWidthFn, mono_words, with_mono_width};
use super::oracle_io::window_cuts;
use super::types::{CapSemantics, Stage4Error};
use crate::circuit::xgate::XGate;
use rayon::prelude::*;
use std::io::Write;
use xxhash_rust::xxh3::Xxh3;

pub const MAGIC: [u8; 8] = *b"LMTDPANF";
pub const FORMAT_VERSION: u16 = 1;
pub const HEADER_BYTES: u16 = 48;
/// Inputs on wires 0..n, answer on wires n..2n, every other wire 0.
pub const IO_LAYOUT_CLASSIC_ZERO_SLICE: u32 = 1;
/// The serialization budget. `anf_dump_eval.py` uses 1<<30 (no cap fires,
/// but a pathological window can then hold billions of monomials); the port
/// emits at the largest accepted split budget instead. A window the repair's
/// split accepted fits its split budget (caps at most these) or has at most
/// 2 gates, so it emits byte-identically; anything larger is refused.
pub const EMIT_BUDGET: usize = super::types::MAX_MONOMIAL_BUDGET;

// ---------------------------------------------------------------- CRC-32

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static CRC_TABLE: [u32; 256] = crc_table();

/// Incremental zlib/IEEE CRC-32 (`zlib.crc32`).
#[derive(Clone, Copy, Debug)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Crc32(0xFFFF_FFFF)
    }
}

impl Crc32 {
    pub fn update(&mut self, bytes: &[u8]) {
        let mut c = self.0;
        for &b in bytes {
            c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }
    pub fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut c = Crc32::default();
    c.update(bytes);
    c.finish()
}

// ---------------------------------------------------------------- container

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerHeader {
    pub source_wires: u32,
    pub state_wires: u32,
    pub io_layout: u32,
    pub window_count: u32,
    pub body_bytes: u64,
    pub body_crc32: u32,
}

impl ContainerHeader {
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES as usize] {
        let mut b = [0u8; HEADER_BYTES as usize];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        b[10..12].copy_from_slice(&HEADER_BYTES.to_le_bytes());
        b[12..16].copy_from_slice(&0u32.to_le_bytes()); // flags
        b[16..20].copy_from_slice(&self.source_wires.to_le_bytes());
        b[20..24].copy_from_slice(&self.state_wires.to_le_bytes());
        b[24..28].copy_from_slice(&self.io_layout.to_le_bytes());
        b[28..32].copy_from_slice(&self.window_count.to_le_bytes());
        b[32..40].copy_from_slice(&self.body_bytes.to_le_bytes());
        b[40..44].copy_from_slice(&self.body_crc32.to_le_bytes());
        let hc = crc32_ieee(&b[0..44]);
        b[44..48].copy_from_slice(&hc.to_le_bytes());
        b
    }

    /// Parse and validate the 48-byte header (magic, version, size, flags,
    /// header CRC, io layout, state width).
    pub fn parse(b: &[u8]) -> Result<ContainerHeader, Stage4Error> {
        let bad = |m: &str| Err(Stage4Error::Input(format!("container: {m}")));
        if b.len() < HEADER_BYTES as usize {
            return bad("shorter than the 48-byte header");
        }
        if b[0..8] != MAGIC {
            return bad("bad magic (not an LMTDPANF container)");
        }
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        if u16_at(8) != FORMAT_VERSION {
            return bad(&format!("unsupported format_version {}", u16_at(8)));
        }
        if u16_at(10) != HEADER_BYTES {
            return bad(&format!("header_bytes {} != 48", u16_at(10)));
        }
        if u32_at(12) != 0 {
            return bad(&format!("unknown flags {:#x}", u32_at(12)));
        }
        if crc32_ieee(&b[0..44]) != u32_at(44) {
            return bad("header CRC mismatch");
        }
        let h = ContainerHeader {
            source_wires: u32_at(16),
            state_wires: u32_at(20),
            io_layout: u32_at(24),
            window_count: u32_at(28),
            body_bytes: u64::from_le_bytes(b[32..40].try_into().unwrap()),
            body_crc32: u32_at(40),
        };
        if h.io_layout != IO_LAYOUT_CLASSIC_ZERO_SLICE {
            return bad(&format!("unknown io_layout {}", h.io_layout));
        }
        if h.state_wires == 0 || h.state_wires > 65536 {
            return bad(&format!("state_wires {} out of range", h.state_wires));
        }
        if (h.source_wires as u64) * 2 > h.state_wires as u64 {
            return bad("source_wires does not fit the zero-slice layout");
        }
        Ok(h)
    }
}

pub fn write_container_header(h: &ContainerHeader, out: &mut impl Write) -> std::io::Result<()> {
    out.write_all(&h.to_bytes())
}

// ---------------------------------------------------------------- encoder

/// Counters of one emission.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmitReport {
    pub windows: usize,
    pub monomials: u64,
    pub max_window_monomials: usize,
    pub empty_windows: usize,
    pub body_bytes: u64,
    pub body_crc32: u32,
    pub body_xxh3: u128,
}

fn enc_err(m: String) -> Stage4Error {
    Stage4Error::Internal(format!("deliverable encoding: {m}"))
}

/// One window's blob (without its `<I len>` prefix), from its ANF.
pub fn encode_window<const W: usize>(anf: &WindowAnf<W>) -> Result<Vec<u8>, Stage4Error> {
    let support = anf.support();
    let f = support.len();
    let c = anf.changed.len();
    if f > u16::MAX as usize || c > u16::MAX as usize {
        return Err(enc_err(format!("support {f} or changed {c} exceeds 65535")));
    }
    let fb = f.max(1).div_ceil(8);
    let mut local = vec![u16::MAX; 64 * W];
    for (i, &w) in support.iter().enumerate() {
        local[w as usize] = i as u16;
    }
    let mut b = Vec::with_capacity(4 + 2 * f + c * 6 + anf.total * fb);
    b.extend_from_slice(&(f as u16).to_le_bytes());
    b.extend_from_slice(&(c as u16).to_le_bytes());
    for &w in &support {
        b.extend_from_slice(&w.to_le_bytes());
    }
    let mut mask = vec![0u8; fb];
    for (w, poly) in &anf.changed {
        let count =
            u32::try_from(poly.len()).map_err(|_| enc_err(format!("wire {w}: count overflow")))?;
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        for m in poly {
            mask.iter_mut().for_each(|x| *x = 0);
            for bit in m.bits() {
                let l = local[bit] as usize;
                mask[l >> 3] |= 1 << (l & 7);
            }
            b.extend_from_slice(&mask);
        }
    }
    if b.len() > u32::MAX as usize {
        return Err(enc_err(format!(
            "window blob of {} bytes exceeds u32",
            b.len()
        )));
    }
    Ok(b)
}

/// Canonical ANF of one window `[lo, hi)` for emission; `OverBudget` (exit 6)
/// when it exceeds [`EMIT_BUDGET`].
pub fn emit_anf<const W: usize>(
    gates: &[XGate],
    lo: usize,
    hi: usize,
) -> Result<WindowAnf<W>, Stage4Error> {
    let mut stats = MulStats::default();
    window_anf::<W>(
        &gates[lo..hi],
        EMIT_BUDGET,
        CapSemantics::CPython39,
        &mut stats,
    )
    .map_err(|_| Stage4Error::OverBudget {
        windows: vec![(lo, hi)],
    })
}

struct EncodeBody<'a, O: Write> {
    gates: &'a [XGate],
    cuts: &'a [usize],
    batch: usize,
    out: &'a mut O,
}

impl<O: Write> MonoWidthFn<Result<EmitReport, Stage4Error>> for EncodeBody<'_, O> {
    fn call<const W: usize>(self) -> Result<EmitReport, Stage4Error> {
        let nwin = self.cuts.len() - 1;
        let mut rep = EmitReport {
            windows: nwin,
            ..Default::default()
        };
        let mut crc = Crc32::default();
        let mut xxh = Xxh3::new();
        let mut start = 0;
        while start < nwin {
            let end = (start + self.batch).min(nwin);
            let blobs: Vec<Result<(Vec<u8>, usize), Stage4Error>> = (start..end)
                .into_par_iter()
                .map(|wi| {
                    let anf = emit_anf::<W>(self.gates, self.cuts[wi], self.cuts[wi + 1])?;
                    Ok((encode_window(&anf)?, anf.total))
                })
                .collect();
            for r in blobs {
                let (blob, total) = r?;
                let len = (blob.len() as u32).to_le_bytes();
                for part in [&len[..], &blob[..]] {
                    crc.update(part);
                    xxh.update(part);
                    self.out
                        .write_all(part)
                        .map_err(|e| Stage4Error::io("<deliverable body>", e))?;
                }
                rep.body_bytes += (4 + blob.len()) as u64;
                rep.monomials += total as u64;
                rep.max_window_monomials = rep.max_window_monomials.max(total);
                if total == 0 {
                    rep.empty_windows += 1;
                }
            }
            start = end;
        }
        rep.body_crc32 = crc.finish();
        rep.body_xxh3 = xxh.digest128();
        Ok(rep)
    }
}

/// Stream the `anf_dump_eval.py` body of `(gates, bounds)` to `out`, windows
/// in order, computing CRC-32 and xxh3-128 on the fly. `m_header` is the
/// circuit's header m (DE's `total`); `nw` is the state width (sets the
/// monomial width). Windows are encoded `batch` at a time in parallel.
pub fn encode_body(
    gates: &[XGate],
    bounds: &[i64],
    m_header: usize,
    nw: usize,
    batch: usize,
    out: &mut impl Write,
) -> Result<EmitReport, Stage4Error> {
    if m_header != gates.len() {
        return Err(Stage4Error::Input(format!(
            "header m={m_header} but {} gates",
            gates.len()
        )));
    }
    let maxw = gates
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    let nw = nw.max(maxw);
    if mono_words(nw).is_none() {
        return Err(Stage4Error::Usage(format!(
            "stage 4 supports at most 1024 state wires, got {nw}"
        )));
    }
    let cuts = window_cuts(bounds, m_header);
    with_mono_width(
        nw,
        EncodeBody {
            gates,
            cuts: &cuts,
            batch: batch.max(1),
            out,
        },
    )
}

// ---------------------------------------------------------------- decoder

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// The raw body (`deliverable.bin`). It carries no width: the caller
    /// states the source width; the state is taken to be 4·n wires unless
    /// `state_wires` is given.
    Raw { source_wires: u32, state_wires: u32 },
    /// Container v1 (`deliverable.tdp4`).
    Container,
}

#[derive(Clone, Copy, Debug)]
struct WindowIdx {
    sup_start: u32,
    sup_len: u16,
    fb: u16,
    wire_start: u32,
    wire_len: u16,
}

#[derive(Clone, Copy, Debug)]
struct WireIdx {
    wire: u16,
    count: u32,
    offset: u64,
}

/// A decoded deliverable as flat arenas (no per-monomial allocation).
#[derive(Clone, Debug)]
pub struct Deliverable {
    pub header: Option<ContainerHeader>,
    pub source_wires: u32,
    pub state_wires: u32,
    bytes: Vec<u8>,
    windows: Vec<WindowIdx>,
    wires: Vec<WireIdx>,
    supports: Vec<u16>,
    pub monomials: u64,
    pub body_bytes: u64,
}

/// One decoded window.
pub struct WindowView<'a> {
    pub support: &'a [u16],
    /// Bytes per local mask (`ceil(max(f,1)/8)`).
    pub mask_bytes: usize,
    d: &'a Deliverable,
    wires: &'a [WireIdx],
}

impl<'a> WindowView<'a> {
    pub fn changed_len(&self) -> usize {
        self.wires.len()
    }
    /// (wire, its masks as one byte slice of `count * mask_bytes`).
    pub fn changed(&self) -> impl Iterator<Item = (u16, &'a [u8])> + '_ {
        let fb = self.mask_bytes;
        let bytes = &self.d.bytes;
        self.wires.iter().map(move |w| {
            let o = w.offset as usize;
            (w.wire, &bytes[o..o + w.count as usize * fb])
        })
    }
}

fn dec_err(wi: usize, m: impl std::fmt::Display) -> Stage4Error {
    Stage4Error::Input(format!("deliverable window {wi}: {m}"))
}

impl Deliverable {
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    pub fn window(&self, i: usize) -> WindowView<'_> {
        let w = self.windows[i];
        WindowView {
            support: &self.supports
                [w.sup_start as usize..w.sup_start as usize + w.sup_len as usize],
            mask_bytes: w.fb as usize,
            d: self,
            wires: &self.wires[w.wire_start as usize..w.wire_start as usize + w.wire_len as usize],
        }
    }

    /// Evaluate on 1024 bit-sliced states: `state[w][j]` holds lanes
    /// 64j..64j+63 of wire w. Each window maps its input snapshot to new
    /// values of its changed wires.
    pub fn eval_lanes16(&self, state: &mut [[u64; 16]]) {
        let mut snap: Vec<[u64; 16]> = Vec::new();
        let mut newv: Vec<[u64; 16]> = Vec::new();
        for i in 0..self.windows.len() {
            let win = self.window(i);
            snap.clear();
            snap.extend(win.support.iter().map(|&w| state[w as usize]));
            newv.clear();
            let fb = win.mask_bytes;
            for (_, masks) in win.changed() {
                let mut acc = [0u64; 16];
                for m in masks.chunks_exact(fb) {
                    let mut term = [!0u64; 16];
                    for (bi, &byte) in m.iter().enumerate() {
                        let mut b = byte;
                        while b != 0 {
                            let l = bi * 8 + b.trailing_zeros() as usize;
                            b &= b - 1;
                            let s = &snap[l];
                            for k in 0..16 {
                                term[k] &= s[k];
                            }
                        }
                    }
                    for k in 0..16 {
                        acc[k] ^= term[k];
                    }
                }
                newv.push(acc);
            }
            for ((w, _), v) in win.changed().zip(newv.iter()) {
                state[w as usize] = *v;
            }
        }
    }

    /// Evaluate one state held as a little-endian limb array, one bit per wire.
    pub fn eval_limbs(&self, state: &mut [u64]) {
        let get = |s: &[u64], w: usize| (s[w >> 6] >> (w & 63)) & 1;
        let mut snap: Vec<u64> = Vec::new();
        let mut newv: Vec<u64> = Vec::new();
        for i in 0..self.windows.len() {
            let win = self.window(i);
            snap.clear();
            snap.extend(win.support.iter().map(|&w| get(state, w as usize)));
            newv.clear();
            let fb = win.mask_bytes;
            for (_, masks) in win.changed() {
                let mut acc = 0u64;
                for m in masks.chunks_exact(fb) {
                    let mut term = 1u64;
                    for (bi, &byte) in m.iter().enumerate() {
                        let mut b = byte;
                        while b != 0 {
                            term &= snap[bi * 8 + b.trailing_zeros() as usize];
                            b &= b - 1;
                        }
                    }
                    acc ^= term;
                }
                newv.push(acc);
            }
            for ((w, _), v) in win.changed().zip(newv.iter()) {
                let w = w as usize;
                state[w >> 6] = (state[w >> 6] & !(1u64 << (w & 63))) | (v << (w & 63));
            }
        }
    }
}

/// Decode and validate a deliverable. `Format::Raw` needs the widths;
/// `Format::Container` requires the magic. Validates (§6): CRCs and lengths
/// (container), blob lengths, support strictly ascending and below the state
/// width, changed wires strictly ascending and below the state width, masks
/// strictly ascending, no mask bit at or above f, every support index used,
/// f = 0 iff c = 0.
pub fn decode(bytes: &[u8], format: Format) -> Result<Deliverable, Stage4Error> {
    let (header, body, source_wires, state_wires) = match format {
        Format::Container => {
            let h = ContainerHeader::parse(bytes)?;
            let body = &bytes[HEADER_BYTES as usize..];
            if body.len() as u64 != h.body_bytes {
                return Err(Stage4Error::Input(format!(
                    "container: body is {} bytes, header says {}",
                    body.len(),
                    h.body_bytes
                )));
            }
            if crc32_ieee(body) != h.body_crc32 {
                return Err(Stage4Error::Input("container: body CRC mismatch".into()));
            }
            (Some(h), body, h.source_wires, h.state_wires)
        }
        Format::Raw {
            source_wires,
            state_wires,
        } => {
            if bytes.len() >= 8 && bytes[0..8] == MAGIC {
                return Err(Stage4Error::Input(
                    "raw format requested but the file starts with the container magic".into(),
                ));
            }
            (None, bytes, source_wires, state_wires)
        }
    };
    if state_wires == 0 || state_wires > 65536 {
        return Err(Stage4Error::Usage(format!(
            "state width {state_wires} out of range"
        )));
    }
    let mut d = Deliverable {
        header,
        source_wires,
        state_wires,
        bytes: Vec::new(),
        windows: Vec::new(),
        wires: Vec::new(),
        supports: Vec::new(),
        monomials: 0,
        body_bytes: body.len() as u64,
    };
    let mut pos = 0usize;
    let rd = |p: usize, n: usize, wi: usize, blob: &[u8]| -> Result<(), Stage4Error> {
        if p + n > blob.len() {
            Err(dec_err(wi, "blob truncated"))
        } else {
            Ok(())
        }
    };
    while pos < body.len() {
        let wi = d.windows.len();
        if pos + 4 > body.len() {
            return Err(dec_err(wi, "truncated length prefix"));
        }
        let len = u32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > body.len() {
            return Err(dec_err(
                wi,
                format!("blob of {len} bytes runs past the body"),
            ));
        }
        let blob = &body[pos..pos + len];
        pos += len;
        rd(0, 4, wi, blob)?;
        let f = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        let c = u16::from_le_bytes([blob[2], blob[3]]) as usize;
        let mut q = 4;
        rd(q, 2 * f, wi, blob)?;
        let sup_start = d.supports.len();
        for i in 0..f {
            let w = u16::from_le_bytes([blob[q + 2 * i], blob[q + 2 * i + 1]]);
            if w as u32 >= state_wires {
                return Err(dec_err(wi, format!("support wire {w} beyond the state")));
            }
            if i > 0 && w <= d.supports[sup_start + i - 1] {
                return Err(dec_err(wi, "support not strictly ascending"));
            }
            d.supports.push(w);
        }
        q += 2 * f;
        if (f == 0) != (c == 0) {
            return Err(dec_err(wi, format!("f={f} but c={c}")));
        }
        let fb = f.max(1).div_ceil(8);
        let wire_start = d.wires.len();
        let mut used = vec![0u8; fb];
        let mut prev_w: Option<u16> = None;
        for _ in 0..c {
            rd(q, 6, wi, blob)?;
            let w = u16::from_le_bytes([blob[q], blob[q + 1]]);
            let count = u32::from_le_bytes(blob[q + 2..q + 6].try_into().unwrap());
            q += 6;
            if w as u32 >= state_wires {
                return Err(dec_err(wi, format!("changed wire {w} beyond the state")));
            }
            if prev_w.is_some_and(|p| w <= p) {
                return Err(dec_err(wi, "changed wires not strictly ascending"));
            }
            prev_w = Some(w);
            let nbytes = count as usize * fb;
            rd(q, nbytes, wi, blob)?;
            let masks = &blob[q..q + nbytes];
            let mut prev: Option<&[u8]> = None;
            for m in masks.chunks_exact(fb) {
                if let Some(p) = prev {
                    // little-endian: compare from the most significant byte
                    if m.iter().rev().cmp(p.iter().rev()) != std::cmp::Ordering::Greater {
                        return Err(dec_err(
                            wi,
                            format!("masks of wire {w} not strictly ascending"),
                        ));
                    }
                }
                prev = Some(m);
                for (u, &b) in used.iter_mut().zip(m) {
                    *u |= b;
                }
            }
            if !f.is_multiple_of(8) || f == 0 {
                let top = fb - 1;
                let allowed: u8 = if f == 0 {
                    0
                } else {
                    (1u16 << (f % 8)).wrapping_sub(1) as u8
                };
                if masks.chunks_exact(fb).any(|m| m[top] & !allowed != 0) {
                    return Err(dec_err(wi, format!("wire {w}: mask bit at or above f={f}")));
                }
            }
            d.wires.push(WireIdx {
                wire: w,
                count,
                offset: d.bytes.len() as u64,
            });
            d.bytes.extend_from_slice(masks);
            d.monomials += count as u64;
            q += nbytes;
        }
        if q != len {
            return Err(dec_err(
                wi,
                format!("{} trailing bytes in the blob", len - q),
            ));
        }
        let all_used = (0..f).all(|l| used[l >> 3] >> (l & 7) & 1 == 1);
        if !all_used {
            return Err(dec_err(wi, "a support wire appears in no monomial"));
        }
        d.windows.push(WindowIdx {
            sup_start: sup_start as u32,
            sup_len: f as u16,
            fb: fb as u16,
            wire_start: wire_start as u32,
            wire_len: c as u16,
        });
    }
    if let Some(h) = header
        && h.window_count as usize != d.windows.len()
    {
        return Err(Stage4Error::Input(format!(
            "container: {} windows, header says {}",
            d.windows.len(),
            h.window_count
        )));
    }
    Ok(d)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/serialize.rs"]
mod tests;
