//! Circuits and bounds exactly as the stage-4 oracle reads and writes them
//! (docs/STAGE4_RUST_DESIGN.md §4.2).
//!
//! The Python `load` keeps controls in file order and accepts malformed
//! gates; `circuit::formats::read_mpmct` sorts and validates. This reader is
//! STRICT: it accepts only files on which `XGate` equality coincides with the
//! Python tuple (controls strictly ascending, none on the target, comp and
//! polarity in {0, 1}, header m equal to the gate count, no extra tokens,
//! plain decimal numbers). Anything else is an `Input` error, never silently
//! canonicalized.
use super::types::Stage4Error;
use crate::circuit::xgate::{Lits, XGate};
use smallvec::SmallVec;
use std::io::{self, Write};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct OracleCircuit {
    /// `hdr[1]` (order_repair, verify_equiv use it).
    pub header_n: usize,
    /// `hdr[2]` (anf_dump_eval uses it as m).
    pub header_m: usize,
    /// `max(header_n, maxwire + 1)` (window_pack3, anf_compose use it).
    pub computed_n: usize,
    pub gates: Vec<XGate>,
}

fn num(tok: &str, what: &str, line: usize) -> Result<usize, Stage4Error> {
    if tok.is_empty() || tok.len() > 18 || !tok.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Stage4Error::Input(format!(
            "line {line}: {what} {tok:?} is not a plain decimal number"
        )));
    }
    Ok(tok.parse().expect("checked digits"))
}

fn wire(tok: &str, what: &str, line: usize) -> Result<u16, Stage4Error> {
    let v = num(tok, what, line)?;
    u16::try_from(v)
        .map_err(|_| Stage4Error::Input(format!("line {line}: {what} {v} exceeds 65535")))
}

fn bit(tok: &str, what: &str, line: usize) -> Result<bool, Stage4Error> {
    match tok {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(Stage4Error::Input(format!(
            "line {line}: {what} {tok:?} is not 0 or 1"
        ))),
    }
}

/// Parse mpmct1 text (`mpmct1 n m`, then `t comp k c1 p1 ... ck pk` per gate;
/// blank lines are skipped as the oracle does).
pub fn parse_oracle_mpmct(text: &str) -> Result<OracleCircuit, Stage4Error> {
    let mut lines = text.lines().enumerate();
    let (_, hdr) = lines
        .next()
        .ok_or_else(|| Stage4Error::Input("empty circuit file".into()))?;
    let h: Vec<&str> = hdr.split_whitespace().collect();
    if h.len() != 3 || h[0] != "mpmct1" {
        return Err(Stage4Error::Input(format!("bad mpmct1 header {hdr:?}")));
    }
    let header_n = num(h[1], "header n", 1)?;
    let header_m = num(h[2], "header m", 1)?;
    let mut gates = Vec::with_capacity(header_m.min(1 << 26));
    let mut maxw = 0usize;
    for (i, line) in lines {
        let ln = i + 1;
        let q: Vec<&str> = line.split_whitespace().collect();
        if q.is_empty() {
            continue;
        }
        if q.len() < 3 {
            return Err(Stage4Error::Input(format!(
                "line {ln}: truncated gate {line:?}"
            )));
        }
        let target = wire(q[0], "target", ln)?;
        let comp = bit(q[1], "comp", ln)?;
        let k = num(q[2], "k", ln)?;
        if q.len() != 3 + 2 * k {
            return Err(Stage4Error::Input(format!(
                "line {ln}: {} tokens for k={k} (expected {})",
                q.len(),
                3 + 2 * k
            )));
        }
        let mut ctrls: Lits = SmallVec::new();
        for j in 0..k {
            let c = wire(q[3 + 2 * j], "control", ln)?;
            let p = bit(q[4 + 2 * j], "polarity", ln)?;
            if c == target {
                return Err(Stage4Error::Input(format!(
                    "line {ln}: control on the target {c}"
                )));
            }
            if let Some(&(prev, _)) = ctrls.last()
                && c <= prev
            {
                return Err(Stage4Error::Input(format!(
                    "line {ln}: controls not strictly ascending ({prev} then {c})"
                )));
            }
            ctrls.push((c, p));
            maxw = maxw.max(c as usize);
        }
        maxw = maxw.max(target as usize);
        gates.push(XGate {
            target,
            comp,
            ctrls,
        });
    }
    if gates.len() != header_m {
        return Err(Stage4Error::Input(format!(
            "header m={header_m} but {} gate lines",
            gates.len()
        )));
    }
    let computed_n = if gates.is_empty() {
        header_n
    } else {
        header_n.max(maxw + 1)
    };
    Ok(OracleCircuit {
        header_n,
        header_m,
        computed_n,
        gates,
    })
}

pub fn read_oracle_mpmct(path: &Path) -> Result<OracleCircuit, Stage4Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Stage4Error::Input(format!("{}: {e}", path.display())))?;
    parse_oracle_mpmct(&text).map_err(|e| match e {
        Stage4Error::Input(m) => Stage4Error::Input(format!("{}: {m}", path.display())),
        other => other,
    })
}

/// Bounds as whitespace-separated integers, input order kept. Negative
/// values are accepted (the oracle's `int()` does, and every consumer keeps
/// only `0 < b < m`).
pub fn parse_bounds(text: &str) -> Result<Vec<i64>, Stage4Error> {
    text.split_whitespace()
        .map(|t| {
            let digits = t.strip_prefix('-').unwrap_or(t);
            if digits.is_empty() || digits.len() > 18 || !digits.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(Stage4Error::Input(format!(
                    "bound {t:?} is not a decimal integer"
                )));
            }
            Ok(t.parse::<i64>().expect("checked digits"))
        })
        .collect()
}

pub fn read_bounds(path: &Path) -> Result<Vec<i64>, Stage4Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Stage4Error::Input(format!("{}: {e}", path.display())))?;
    parse_bounds(&text).map_err(|e| match e {
        Stage4Error::Input(m) => Stage4Error::Input(format!("{}: {m}", path.display())),
        other => other,
    })
}

/// The oracle's window cuts: `[0] + sorted(b for b in bounds if 0 < b < m) + [m]`
/// (no dedup: a repeated bound is an empty window).
pub fn window_cuts(bounds: &[i64], m: usize) -> Vec<usize> {
    let mut inner: Vec<usize> = bounds
        .iter()
        .filter(|&&b| b > 0 && (b as u64) < m as u64)
        .map(|&b| b as usize)
        .collect();
    inner.sort_unstable();
    let mut cuts = Vec::with_capacity(inner.len() + 2);
    cuts.push(0);
    cuts.extend(inner);
    cuts.push(m);
    cuts
}

/// One gate line as both oracle writers print it, without the newline:
/// `t comp k c1 p1 ...` (`k=0` gives `"t c 0"`).
fn gate_line(g: &XGate) -> String {
    let mut s = format!("{} {} {}", g.target, g.comp as u8, g.ctrls.len());
    for &(c, p) in &g.ctrls {
        s.push_str(&format!(" {} {}", c, p as u8));
    }
    s
}

/// window_pack3 output (WP:476-481): no trailing space.
pub fn write_pack_mpmct(w: &mut impl Write, n: usize, gates: &[XGate]) -> io::Result<()> {
    writeln!(w, "mpmct1 {} {}", n, gates.len())?;
    for g in gates {
        writeln!(w, "{}", gate_line(g))?;
    }
    Ok(())
}

/// order_repair.save: a gate with k=0 is written `"t c 0 \n"` (trailing space).
pub fn write_repair_mpmct(w: &mut impl Write, n: usize, gates: &[XGate]) -> io::Result<()> {
    writeln!(w, "mpmct1 {} {}", n, gates.len())?;
    for g in gates {
        if g.ctrls.is_empty() {
            writeln!(w, "{} ", gate_line(g))?;
        } else {
            writeln!(w, "{}", gate_line(g))?;
        }
    }
    Ok(())
}

/// `"\n".join(map(str, b)) + "\n"` ("\n" when empty).
pub fn write_bounds(w: &mut impl Write, b: &[usize]) -> io::Result<()> {
    let s: Vec<String> = b.iter().map(|x| x.to_string()).collect();
    writeln!(w, "{}", s.join("\n"))
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/oracle_io.rs"]
mod tests;
