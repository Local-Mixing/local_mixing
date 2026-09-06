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
// ANF by a fixed deterministic compaction (postprocessing::compress::compact)
// -- a function of the activation function alone, hence also one spelling
// per function, at ~2.3x fewer terms than the ANF:
//
//   esop1 <wires> <gates>
//   <target> <n_terms> [<width> <w_1> <p_1> ... <w_width> <p_width>]*
//
// Literals are (wire, polarity 0/1) as in mpmct1. Wires ascend inside a
// term, terms are sorted by (size, literals), and an empty term is the
// constant 1. Any mpmct1 reader loads both transparently (`read_mpmct`), as
// one cube gate per term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedGate {
    pub target: u16,
    // Sorted canonically; each term is an ascending (wire, polarity) list,
    // empty = the constant 1. All-positive terms in the anf1 form.
    pub terms: Vec<Vec<(u16, bool)>>,
}

impl PackedGate {
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    pub fn is_anf(&self) -> bool {
        self.terms.iter().all(|t| t.iter().all(|l| l.1))
    }

    /// Canonical term order: by size, then the literal list.
    pub fn sort_terms(&mut self) {
        self.terms
            .sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    }

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

/// Write packed gates: `anf1` when every term is positive (asserted), else
/// `esop1`. Returns the header word used.
pub fn write_packed(path: &str, packed: &[PackedGate], num_wires: usize, anf: bool) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    writeln!(out, "{} {} {}", if anf { "anf1" } else { "esop1" }, num_wires, packed.len())?;
    let mut line = Vec::with_capacity(256);
    for g in packed {
        line.clear();
        push_dec(&mut line, g.target as u32);
        line.push(b' ');
        push_dec(&mut line, g.terms.len() as u32);
        for t in &g.terms {
            line.push(b' ');
            push_dec(&mut line, t.len() as u32);
            for &(w, p) in t {
                line.push(b' ');
                push_dec(&mut line, w as u32);
                if anf {
                    assert!(p, "anf1 terms are positive monomials");
                } else {
                    line.push(b' ');
                    line.push(b'0' + p as u8);
                }
            }
        }
        line.push(b'\n');
        out.write_all(&line)?;
    }
    out.flush()
}

pub fn write_anf1(path: &str, packed: &[PackedGate], num_wires: usize) -> io::Result<()> {
    write_packed(path, packed, num_wires, true)
}

pub fn write_esop1(path: &str, packed: &[PackedGate], num_wires: usize) -> io::Result<()> {
    write_packed(path, packed, num_wires, false)
}

/// Read an anf1 or esop1 file (dispatching on the header).
pub fn read_packed(path: &str) -> io::Result<(Vec<PackedGate>, usize)> {
    let raw = std::fs::read(path)?;
    read_packed_bytes(&raw)
}

pub fn read_anf1(path: &str) -> io::Result<(Vec<PackedGate>, usize)> {
    read_packed(path)
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
                let p = if anf { true } else { f.u32().ok_or_else(bad)? != 0 };
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
mod tests {
    use super::*;

    #[test]
    fn mpmct_writer_matches_golden_bytes_and_roundtrips() {
        let gates = vec![
            XGate::x_gate(12),
            XGate {
                target: 7,
                comp: true,
                ctrls: [(2, false), (9, true)].into_iter().collect(),
            },
            XGate::conj(3, [(1, false), (15, true)]).unwrap(),
        ];
        let path = std::env::temp_dir().join(format!(
            "local_mixing_mpmct_golden_{}.txt",
            std::process::id()
        ));
        let path_str = path.to_str().expect("temporary path is UTF-8");

        write_mpmct(path_str, &gates, 16).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes,
            b"mpmct1 16 3\n12 0 0\n7 1 2 2 0 9 1\n3 0 2 1 0 15 1\n"
        );

        let (roundtrip, num_wires) = read_mpmct(path_str).unwrap();
        assert_eq!(num_wires, 16);
        assert_eq!(roundtrip, gates);
        std::fs::remove_file(path).ok();
    }

    fn read_str(body: &str, tag: &str) -> io::Result<(Vec<XGate>, usize)> {
        let path = std::env::temp_dir().join(format!(
            "local_mixing_mpmct_{tag}_{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, body).unwrap();
        let out = read_mpmct(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        out
    }

    // The byte reader replaced a `lines()` + `split_whitespace()` + `parse()`
    // one. These are the input shapes that reader tolerated.
    #[test]
    fn opt_equiv_read_mpmct_keeps_whitespace_and_blank_line_tolerance() {
        let want = vec![
            XGate::x_gate(12),
            XGate {
                target: 7,
                comp: true,
                ctrls: [(2, false), (9, true)].into_iter().collect(),
            },
        ];
        // Blank lines, CRLF, leading/inner/trailing spaces and tabs, a
        // trailing newline, and out-of-order controls that must be sorted.
        for (tag, body) in [
            ("plain", "mpmct1 16 2\n12 0 0\n7 1 2 2 0 9 1\n"),
            ("noeol", "mpmct1 16 2\n12 0 0\n7 1 2 2 0 9 1"),
            ("blank", "mpmct1 16 2\n12 0 0\n\n   \n7 1 2 2 0 9 1\n\n"),
            ("crlf", "mpmct1 16 2\r\n12 0 0\r\n7 1 2 2 0 9 1\r\n"),
            (
                "spacey",
                "mpmct1  16   2\n  12 0 0  \n\t7  1 2   2 0 9 1\t\n",
            ),
            ("unsorted", "mpmct1 16 2\n12 0 0\n7 1 2 9 1 2 0\n"),
        ] {
            let (got, wires) = read_str(body, tag).unwrap_or_else(|e| panic!("{tag}: {e}"));
            assert_eq!(wires, 16, "{tag}");
            assert_eq!(got, want, "{tag}");
        }
    }

    #[test]
    fn read_mpmct_rejects_malformed_input() {
        assert!(read_str("", "empty").is_err());
        assert!(read_str("g57 16 2\n", "wrongtag").is_err());
        assert!(read_str("mpmct1 16\n", "shorthdr").is_err());
        assert!(read_str("mpmct1 x 2\n", "nonnumhdr").is_err());
        // k=2 announced but only one control pair present.
        assert!(read_str("mpmct1 16 1\n7 1 2 2 0\n", "shortgate").is_err());
        // Header count disagrees with the body.
        assert!(read_str("mpmct1 16 5\n12 0 0\n", "countdrift").is_err());
    }
}
