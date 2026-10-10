// File I/O for post-mixed circuits.
//
// mpmct1 format (plain text, one gate per line):
//   mpmct1 <num_wires> <num_gates>
//   <target> <comp:0|1> <k> <wire> <pol:0|1> ... (k pairs)
//
// g57 input uses the existing base-83 CircuitSeq format.
use crate::circuit::CircuitSeq;
use crate::circuit::xgate::{Lits, XGate, sort_lits};
use std::io::{self, BufWriter, Write};

pub fn read_g57_file(path: &str) -> io::Result<Vec<XGate>> {
    // base-83 is ASCII, so the bytes go straight to the parser without the
    // UTF-8 validation pass `read_to_string` would run over the whole file.
    let raw = std::fs::read(path)?;
    let c = CircuitSeq::from_bytes(&raw);
    Ok(c.gates.iter().map(|&g| XGate::from_g57(g)).collect())
}

pub fn write_mpmct(path: &str, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    write_mpmct_to(&mut out, gates, num_wires)?;
    out.flush()
}

/// [`write_mpmct`] through `<path>.tmp`: written, fsynced, then renamed over
/// `path` (and the directory synced, best effort). A kill or a crash leaves
/// the previous file or no file at `path`, never a truncated one.
pub fn write_mpmct_atomic(path: &str, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    let tmp = format!("{path}.tmp");
    let result = (|| {
        let file = std::fs::File::create(&tmp)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        write_mpmct_to(&mut out, gates, num_wires)?;
        let file = out.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
        return result;
    }
    let parent = std::path::Path::new(path)
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."));
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

fn write_mpmct_to(mut out: impl Write, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    writeln!(out, "mpmct1 {} {}", num_wires, gates.len())?;
    // Every field is a small unsigned integer, so the line is assembled with a
    // direct decimal writer instead of going through `write!`'s formatting
    // machinery once per field.
    let mut line = Vec::with_capacity(64);
    for g in gates {
        line.clear();
        push_dec(&mut line, g.target as u32);
        line.push(b' ');
        line.push(b'0' + g.comp as u8);
        line.push(b' ');
        push_dec(&mut line, g.ctrls.len() as u32);
        for &(w, p) in &g.ctrls {
            line.push(b' ');
            push_dec(&mut line, w as u32);
            line.push(b' ');
            line.push(b'0' + p as u8);
        }
        line.push(b'\n');
        out.write_all(&line)?;
    }
    Ok(())
}

/// Append `v` in decimal. Small-value fast path first: targets, polarities and
/// control counts are almost always one or two digits.
#[inline]
fn push_dec(out: &mut Vec<u8>, v: u32) {
    if v < 10 {
        out.push(b'0' + v as u8);
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut v = v;
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    out.extend_from_slice(&buf[i..]);
}

/// Whitespace-separated unsigned decimal fields within one line.
///
/// The previous reader went through `str::split_whitespace` plus
/// `str::parse::<u32>` for every field, which is ~190 ns per gate on a
/// production artifact; the file is pure ASCII digits, so a byte scan does the
/// same job without the UTF-8 validation of `read_to_string` on top.
struct Fields<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Fields<'a> {
    fn new(b: &'a [u8]) -> Self {
        Fields { b, i: 0 }
    }

    fn skip_space(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    /// Next whitespace-delimited token, verbatim.
    fn word(&mut self) -> Option<&'a [u8]> {
        self.skip_space();
        if self.i >= self.b.len() {
            return None;
        }
        let start = self.i;
        while self.i < self.b.len() && !self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
        Some(&self.b[start..self.i])
    }

    /// Next token as u32. `None` for a missing or non-numeric token, matching
    /// the old `parse::<u32>().ok()`.
    fn u32(&mut self) -> Option<u32> {
        let w = self.word()?;
        if w.is_empty() {
            return None;
        }
        let mut v: u32 = 0;
        for &c in w {
            if !c.is_ascii_digit() {
                return None;
            }
            v = v.wrapping_mul(10).wrapping_add((c - b'0') as u32);
        }
        Some(v)
    }
}

// ---- packed canonical circuit formats: anf1 and esop1 -------------------------
//
// One generalized gate per line. In `anf1` the activation function is spelled
// as its algebraic normal form (XOR of positive monomials), the unique
// representation of a Boolean function:
//
//   anf1 <wires> <gates>
//   <target> <n_terms> [<degree> <w_1> ... <w_degree>]*
//
// In `esop1` it is an exclusive sum of mixed-polarity cubes, produced from the
// ANF by a fixed deterministic compaction (implemented in the historic fcompress
// stage, branch ran/archive/curated-and-postmix) -- a function of the
// activation function alone, hence also one spelling per function, at ~2.3x
// fewer terms than the ANF:
//
//   esop1 <wires> <gates>
//   <target> <n_terms> [<width> <w_1> <p_1> ... <w_width> <p_width>]*
//
// Literals are (wire, polarity 0/1) as in mpmct1. Wires ascend inside a
// term, terms are sorted by (size, literals), and an empty term is the
// constant 1. Any mpmct1 reader loads both transparently (`read_mpmct`), as
// one cube gate per term.
//
// Only the READ side lives here: the writers had a single producer, the
// historic fcompress stage, and went to the archive branch with it. The
// reader stays so that packed outputs of archive-built runs still load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedGate {
    pub target: u16,
    // Sorted canonically; each term is an ascending (wire, polarity) list,
    // empty = the constant 1. All-positive terms in the anf1 form.
    pub terms: Vec<Vec<(u16, bool)>>,
}

impl PackedGate {
    /// Exact expansion into XGates: one conjunction (or a bare X for the
    /// constant) per term, all on the target.
    pub fn expand(&self) -> Vec<XGate> {
        self.terms
            .iter()
            .map(|t| {
                XGate::conj(self.target, t.iter().copied()).expect("a term has distinct wires")
            })
            .collect()
    }
}

pub fn expand_packed(packed: &[PackedGate]) -> Vec<XGate> {
    packed.iter().flat_map(PackedGate::expand).collect()
}

/// Read an anf1 or esop1 file (dispatching on the header).
pub fn read_packed(path: &str) -> io::Result<(Vec<PackedGate>, usize)> {
    let raw = std::fs::read(path)?;
    read_packed_bytes(&raw)
}

fn read_packed_bytes(raw: &[u8]) -> io::Result<(Vec<PackedGate>, usize)> {
    let mut lines = raw.split(|&c| c == b'\n');
    let header = lines
        .next()
        .ok_or_else(|| io::Error::other("empty packed circuit file"))?;
    let mut hp = Fields::new(header);
    let anf = match hp.word() {
        Some(b"anf1") => true,
        Some(b"esop1") => false,
        _ => return Err(io::Error::other("missing anf1/esop1 header")),
    };
    let bad_header = || io::Error::other("bad packed-circuit header");
    let num_wires = hp.u32().ok_or_else(bad_header)? as usize;
    let num_gates = hp.u32().ok_or_else(bad_header)? as usize;
    let mut packed = Vec::with_capacity(num_gates);
    for line in lines {
        let mut f = Fields::new(line);
        f.skip_space();
        if f.i >= line.len() {
            continue;
        }
        let bad = || io::Error::other(format!("bad packed gate line: {}", show(line)));
        let target = f.u32().ok_or_else(bad)? as u16;
        let n = f.u32().ok_or_else(bad)? as usize;
        let mut terms = Vec::with_capacity(n);
        for _ in 0..n {
            let k = f.u32().ok_or_else(bad)? as usize;
            let mut t = Vec::with_capacity(k);
            for _ in 0..k {
                let w = f.u32().ok_or_else(bad)? as u16;
                let p = if anf {
                    true
                } else {
                    f.u32().ok_or_else(bad)? != 0
                };
                t.push((w, p));
            }
            terms.push(t);
        }
        if f.word().is_some() {
            return Err(bad());
        }
        packed.push(PackedGate { target, terms });
    }
    if packed.len() != num_gates {
        return Err(io::Error::other(format!(
            "packed gate count mismatch: header {} vs {}",
            num_gates,
            packed.len()
        )));
    }
    Ok((packed, num_wires))
}

/// Read any circuit format as XGates: an mpmct1 file verbatim, an anf1 or
/// esop1 file as the term expansion of its packed gates (exact, one cube per
/// term). Every mpmct1 consumer accepts packed circuits through this.
pub fn read_mpmct(path: &str) -> io::Result<(Vec<XGate>, usize)> {
    let raw = std::fs::read(path)?;
    // `str::lines` semantics: split on '\n', a trailing '\r' is whitespace and
    // is skipped by the field scanner.
    let mut lines = raw.split(|&c| c == b'\n');
    let header = lines
        .next()
        .ok_or_else(|| io::Error::other("empty mpmct file"))?;
    let mut hp = Fields::new(header);
    match hp.word() {
        Some(b"mpmct1") => {}
        Some(b"anf1") | Some(b"esop1") => {
            let (packed, wires) = read_packed_bytes(&raw)?;
            return Ok((expand_packed(&packed), wires));
        }
        _ => return Err(io::Error::other("missing mpmct1 header")),
    }
    let bad_header = || io::Error::other("bad mpmct header");
    let num_wires = hp.u32().ok_or_else(bad_header)? as usize;
    let num_gates = hp.u32().ok_or_else(bad_header)? as usize;

    let mut gates = Vec::with_capacity(num_gates);
    for line in lines {
        let mut f = Fields::new(line);
        f.skip_space();
        if f.i >= line.len() {
            continue; // blank line
        }
        let bad = || io::Error::other(format!("bad mpmct gate line: {}", show(line)));
        let target = f.u32().ok_or_else(bad)? as u16;
        let comp = f.u32().ok_or_else(bad)? != 0;
        let k = f.u32().ok_or_else(bad)? as usize;
        let mut ctrls: Lits = Lits::new();
        for _ in 0..k {
            let w = f.u32().ok_or_else(bad)? as u16;
            let p = f.u32().ok_or_else(bad)? != 0;
            ctrls.push((w, p));
        }
        sort_lits(&mut ctrls);
        // Enforce the XGate invariant (xgate.rs): controls sorted by wire, at
        // most one literal per wire, never the gate's own target. No legitimate
        // writer produces violations (from_g57 normalizes x==y to an X gate;
        // conj merges/rejects), and accepting them would silently corrupt the
        // sorted linear scan in XGate::collides and GateMask's one-bit-per-wire
        // assumption — so reject rather than normalize.
        for i in 0..ctrls.len() {
            if ctrls[i].0 == target {
                return Err(io::Error::other(format!(
                    "mpmct control on its own target: {}",
                    show(line)
                )));
            }
            if i > 0 && ctrls[i - 1].0 == ctrls[i].0 {
                return Err(io::Error::other(format!(
                    "mpmct duplicate or contradictory control wire: {}",
                    show(line)
                )));
            }
        }
        gates.push(XGate {
            target,
            comp,
            ctrls,
        });
    }
    if gates.len() != num_gates {
        return Err(io::Error::other(format!(
            "mpmct gate count mismatch: header {} vs {}",
            num_gates,
            gates.len()
        )));
    }
    Ok((gates, num_wires))
}

fn show(line: &[u8]) -> String {
    String::from_utf8_lossy(line).trim_end().to_string()
}

#[cfg(test)]
#[path = "../../tests/unit/circuit/formats/tests.rs"]
mod tests;

mod g57;
