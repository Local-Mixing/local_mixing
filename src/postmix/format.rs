// File I/O for post-mixed circuits.
//
// mpmct1 format (plain text, one gate per line):
//   mpmct1 <num_wires> <num_gates>
//   <target> <comp:0|1> <k> <wire> <pol:0|1> ... (k pairs)
//
// g57 input uses the existing base-83 CircuitSeq format.
use super::xgate::{Lits, XGate};
use crate::circuit::CircuitSeq;
use std::io::{self, Write};

pub fn read_g57_file(path: &str) -> io::Result<Vec<XGate>> {
    let s = std::fs::read_to_string(path)?;
    let c = CircuitSeq::from_string(&s);
    Ok(c.gates.iter().map(|&g| XGate::from_g57(g)).collect())
}

pub fn write_mpmct(path: &str, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    let mut out = String::with_capacity(gates.len() * 24 + 32);
    out.push_str(&format!("mpmct1 {} {}\n", num_wires, gates.len()));
    for g in gates {
        out.push_str(&format!("{} {} {}", g.target, g.comp as u8, g.ctrls.len()));
        for &(w, p) in &g.ctrls {
            out.push_str(&format!(" {} {}", w, p as u8));
        }
        out.push('\n');
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())
}

pub fn read_mpmct(path: &str) -> io::Result<(Vec<XGate>, usize)> {
    let s = std::fs::read_to_string(path)?;
    let mut lines = s.lines();
    let header = lines.next().ok_or_else(|| io::Error::other("empty mpmct file"))?;
    let mut hp = header.split_whitespace();
    if hp.next() != Some("mpmct1") {
        return Err(io::Error::other("missing mpmct1 header"));
    }
    let parse = |t: Option<&str>| -> io::Result<usize> {
        t.and_then(|x| x.parse().ok()).ok_or_else(|| io::Error::other("bad mpmct header"))
    };
    let num_wires = parse(hp.next())?;
    let num_gates = parse(hp.next())?;
    let mut gates = Vec::with_capacity(num_gates);
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut t = line.split_whitespace().map(|x| x.parse::<u32>());
        let mut next = || -> io::Result<u32> {
            t.next()
                .and_then(|r| r.ok())
                .ok_or_else(|| io::Error::other(format!("bad mpmct gate line: {line}")))
        };
        let target = next()? as u16;
        let comp = next()? != 0;
        let k = next()? as usize;
        let mut ctrls: Lits = Lits::new();
        for _ in 0..k {
            let w = next()? as u16;
            let p = next()? != 0;
            ctrls.push((w, p));
        }
        ctrls.sort_unstable();
        gates.push(XGate { target, comp, ctrls });
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
