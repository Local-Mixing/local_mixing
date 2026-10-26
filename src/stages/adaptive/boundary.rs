//! Classification of C's births against the public quadratic family (port
//! of the sampled half of the server prototype's `boundary.py`).
//!
//! A birth `c_g` is `public_quadratic_boundary` when it is a Boolean
//! polynomial of degree at most 2 in the public INPUT bits, or separately in
//! the public OUTPUT bits of C (no mixed input/output products). Such births
//! are not "deeper": they are computable by anyone holding the public
//! interface, so their exposure in the delivered circuit is not a leak.
//!
//! Input side: the unique degree-<=2 candidate is interpolated from C's
//! values at the points {0, every unit vector, every pair} (Moebius
//! inversion), then tested on `probe_inputs` fresh random inputs. Output
//! side: the same with C run BACKWARD from the output coordinates (the
//! gates after the birth are undone in reverse order, so the recorded value
//! is the birth as a function of the outputs).
//!
//! The prototype PROVED every surviving candidate with an exact SAT miter.
//! This port has no SAT solver: a candidate that agrees with C on every
//! probe input is ACCEPTED ON SAMPLES, not proved. A mismatch is still an
//! exact non-membership certificate (the degree-<=2 interpolant is unique).
//! With 8192 probes a false acceptance needs a nonzero polynomial of degree
//! above 2 that vanishes on all of them; the births of a random C differ
//! from their interpolant on about half of the inputs.
use super::census::{apply_words, invalid, parse_bool, parse_num};
use crate::circuit::xgate::XGate;
use crate::stages::packing::gf2::bit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Fresh random probe inputs per side.
pub const DEFAULT_PROBE_INPUTS: usize = 8192;
/// Probe seed (the prototype's `random.Random(196731 + n)`; `StdRng` here).
pub const PROBE_SEED: u64 = 196_731;

const HEADER: &str = "source_gate\tsource_wire\tinput_quadratic\toutput_quadratic\tis_output";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryRow {
    pub source_gate: usize,
    pub source_wire: u16,
    /// Degree <= 2 in the public inputs (on samples).
    pub input_quadratic: bool,
    /// Degree <= 2 in the public outputs (on samples).
    pub output_quadratic: bool,
    /// The birth is the last write of its wire (a final output of C).
    pub is_output: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Boundary {
    pub rows: Vec<BoundaryRow>,
    /// Births quadratic on the input OR the output side (outputs of C
    /// included; `Source::new` takes this as its `excluded`).
    pub excluded: BTreeSet<usize>,
    pub probe_inputs: usize,
}

/// The value born by every gate of `c` on the lane inputs `inputs`
/// (`[wire < n][word]`), forward (as a function of the inputs) or backward
/// (`reverse`: as a function of the outputs, undoing the later gates).
fn traces(c: &[XGate], inputs: &[Vec<u64>], reverse: bool) -> Vec<Vec<u64>> {
    let words = inputs.first().map_or(0, Vec::len);
    let mut state = inputs.to_vec();
    let mut acc = vec![0u64; words];
    let mut born = Vec::with_capacity(c.len());
    if reverse {
        for g in c.iter().rev() {
            born.push(state[g.target as usize].clone());
            apply_words(g, &mut state, &mut acc);
        }
        born.reverse();
    } else {
        for g in c {
            apply_words(g, &mut state, &mut acc);
            born.push(state[g.target as usize].clone());
        }
    }
    born
}

/// Classify every birth of `c` (gates over wires `0..n`).
pub fn classify(c: &[XGate], n: usize, probe_inputs: usize) -> Boundary {
    assert!(n >= 1, "an empty public interface");
    assert!(
        c.iter().all(|g| (g.max_wire() as usize) < n),
        "C uses a wire beyond n = {n}"
    );
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();
    let points = 1 + n + pairs.len();
    let pw = points.div_ceil(64);
    // Interpolation points as lanes: point 0 = 0, point 1+i = e_i, point
    // 1+n+k = e_i + e_j for pair k.
    let mut inputs = vec![vec![0u64; pw]; n];
    for (i, lane) in inputs.iter_mut().enumerate() {
        lane[(1 + i) / 64] |= 1 << ((1 + i) % 64);
    }
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let p = 1 + n + k;
        inputs[i][p / 64] |= 1 << (p % 64);
        inputs[j][p / 64] |= 1 << (p % 64);
    }
    let fw = probe_inputs.div_ceil(64).max(1);
    let mut rng = StdRng::seed_from_u64(PROBE_SEED + n as u64);
    let mut fresh = vec![vec![0u64; fw]; n];
    for lane in fresh.iter_mut() {
        for x in lane.iter_mut() {
            *x = rng.random::<u64>();
        }
        if !probe_inputs.is_multiple_of(64) {
            lane[probe_inputs / 64] &= (1u64 << (probe_inputs % 64)) - 1;
        }
    }
    let tail_mask = |word: usize| -> u64 {
        if word + 1 == fw && !probe_inputs.is_multiple_of(64) {
            (1u64 << (probe_inputs % 64)) - 1
        } else {
            !0u64
        }
    };

    let mut rows: Vec<BoundaryRow> = c
        .iter()
        .enumerate()
        .map(|(g, x)| BoundaryRow {
            source_gate: g,
            source_wire: x.target,
            input_quadratic: false,
            output_quadratic: false,
            is_output: false,
        })
        .collect();
    for reverse in [false, true] {
        let tt = traces(c, &inputs, reverse);
        let ft = traces(c, &fresh, reverse);
        for (g, row) in rows.iter_mut().enumerate() {
            let truth = &tt[g];
            let c0 = bit(truth, 0);
            let units: Vec<bool> = (0..n).map(|i| bit(truth, 1 + i)).collect();
            let lin: Vec<usize> = (0..n).filter(|&i| units[i] ^ c0).collect();
            let quad: Vec<(usize, usize)> = pairs
                .iter()
                .enumerate()
                .filter(|&(k, &(i, j))| bit(truth, 1 + n + k) ^ units[i] ^ units[j] ^ c0)
                .map(|(_, &p)| p)
                .collect();
            let quadratic = (0..fw).all(|word| {
                let mut pred = 0u64.wrapping_sub(c0 as u64);
                for &i in &lin {
                    pred ^= fresh[i][word];
                }
                for &(i, j) in &quad {
                    pred ^= fresh[i][word] & fresh[j][word];
                }
                (pred ^ ft[g][word]) & tail_mask(word) == 0
            });
            if reverse {
                row.output_quadratic = quadratic;
            } else {
                row.input_quadratic = quadratic;
            }
        }
    }
    let mut last: BTreeMap<u16, usize> = BTreeMap::new();
    for (g, x) in c.iter().enumerate() {
        last.insert(x.target, g);
    }
    for row in rows.iter_mut() {
        row.is_output = last[&row.source_wire] == row.source_gate;
    }
    let excluded = rows
        .iter()
        .filter(|r| r.input_quadratic || r.output_quadratic)
        .map(|r| r.source_gate)
        .collect();
    Boundary {
        rows,
        excluded,
        probe_inputs,
    }
}

/// `# probe_inputs=N`, a header, one tab-separated line per row.
pub fn write_file(b: &Boundary, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    writeln!(w, "# probe_inputs={}", b.probe_inputs)?;
    writeln!(w, "{HEADER}")?;
    for r in &b.rows {
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}",
            r.source_gate, r.source_wire, r.input_quadratic, r.output_quadratic, r.is_output
        )?;
    }
    w.flush()
}

/// The inverse of [`write_file`] (`excluded` is rebuilt from the rows).
pub fn read_file(path: &Path) -> io::Result<Boundary> {
    let f = BufReader::new(fs::File::open(path)?);
    let mut b = Boundary::default();
    let mut header_seen = false;
    for (k, line) in f.lines().enumerate() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            if let Some(v) = rest.trim().strip_prefix("probe_inputs=") {
                b.probe_inputs = parse_num(v.trim(), "probe_inputs")?;
            }
            continue;
        }
        if !header_seen {
            if line != HEADER {
                return Err(invalid(format!("{}: unexpected header", path.display())));
            }
            header_seen = true;
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 5 {
            return Err(invalid(format!(
                "{}: line {}: 5 columns expected",
                path.display(),
                k + 1
            )));
        }
        b.rows.push(BoundaryRow {
            source_gate: parse_num(f[0], "source_gate")?,
            source_wire: parse_num(f[1], "source_wire")?,
            input_quadratic: parse_bool(f[2], "input_quadratic")?,
            output_quadratic: parse_bool(f[3], "output_quadratic")?,
            is_output: parse_bool(f[4], "is_output")?,
        });
    }
    if !header_seen {
        return Err(invalid(format!("{}: no header", path.display())));
    }
    b.excluded = b
        .rows
        .iter()
        .filter(|r| r.input_quadratic || r.output_quadratic)
        .map(|r| r.source_gate)
        .collect();
    Ok(b)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/adaptive/boundary.rs"]
mod tests;
