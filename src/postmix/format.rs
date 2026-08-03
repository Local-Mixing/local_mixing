// File I/O for post-mixed circuits.
//
// mpmct1 format (plain text, one gate per line):
//   mpmct1 <num_wires> <num_gates>
//   <target> <comp:0|1> <k> <wire> <pol:0|1> ... (k pairs)
//
// g57 input uses the existing base-83 CircuitSeq format.
use super::xgate::{Lits, XGate};
use crate::circuit::CircuitSeq;
use std::io::{self, BufWriter, Write};

const MPMCT_WRITE_BUFFER_BYTES: usize = 1 << 20;

pub fn read_g57_file(path: &str) -> io::Result<Vec<XGate>> {
    let s = std::fs::read_to_string(path)?;
    let c = CircuitSeq::from_string(&s);
    Ok(c.gates.iter().map(|&g| XGate::from_g57(g)).collect())
}

pub fn write_mpmct(path: &str, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut out = BufWriter::with_capacity(MPMCT_WRITE_BUFFER_BYTES, file);
    write_mpmct_to(&mut out, gates, num_wires)?;
    out.flush()
}

fn write_mpmct_to(out: &mut impl Write, gates: &[XGate], num_wires: usize) -> io::Result<()> {
    writeln!(out, "mpmct1 {} {}", num_wires, gates.len())?;
    for g in gates {
        write!(out, "{} {} {}", g.target, g.comp as u8, g.ctrls.len())?;
        for &(w, p) in &g.ctrls {
            write!(out, " {} {}", w, p as u8)?;
        }
        out.write_all(b"\n")?;
    }
    Ok(())
}

pub fn mpmct_string(gates: &[XGate], num_wires: usize) -> String {
    let mut out = String::with_capacity(gates.len() * 24 + 32);
    out.push_str(&format!("mpmct1 {} {}\n", num_wires, gates.len()));
    for g in gates {
        out.push_str(&format!("{} {} {}", g.target, g.comp as u8, g.ctrls.len()));
        for &(w, p) in &g.ctrls {
            out.push_str(&format!(" {} {}", w, p as u8));
        }
        out.push('\n');
    }
    out
}

pub fn read_mpmct(path: &str) -> io::Result<(Vec<XGate>, usize)> {
    let s = std::fs::read_to_string(path)?;
    let mut lines = s.lines();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::other("empty mpmct file"))?;
    let mut hp = header.split_whitespace();
    if hp.next() != Some("mpmct1") {
        return Err(io::Error::other("missing mpmct1 header"));
    }
    let parse = |t: Option<&str>| -> io::Result<usize> {
        t.and_then(|x| x.parse().ok())
            .ok_or_else(|| io::Error::other("bad mpmct header"))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(target: u16, comp: bool, controls: &[(u16, bool)]) -> XGate {
        let mut ctrls: Lits = controls.iter().copied().collect();
        ctrls.sort_unstable();
        XGate {
            target,
            comp,
            ctrls,
        }
    }

    #[test]
    fn streaming_mpmct_writer_is_byte_exact_and_roundtrips() {
        let gates = vec![
            XGate::x_gate(0),
            gate(9, true, &[(7, true), (1, false)]),
            gate(
                20,
                false,
                &[
                    (8, false),
                    (2, true),
                    (6, true),
                    (4, true),
                    (5, false),
                    (3, false),
                ],
            ),
            gate(
                31,
                true,
                &[
                    (16, false),
                    (10, false),
                    (15, true),
                    (12, false),
                    (14, false),
                    (11, true),
                    (13, true),
                ],
            ),
        ];
        let expected = b"mpmct1 32 4\n\
0 0 0\n\
9 1 2 1 0 7 1\n\
20 0 6 2 1 3 0 4 1 5 0 6 1 8 0\n\
31 1 7 10 0 11 1 12 0 13 1 14 0 15 1 16 0\n";
        let path = std::env::temp_dir().join(format!(
            "local_mixing_mpmct_writer_{}.mpmct1",
            std::process::id()
        ));
        let path_str = path.to_str().expect("temporary path is UTF-8");

        write_mpmct(path_str, &gates, 32).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let (roundtripped, num_wires) = read_mpmct(path_str).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(bytes, expected);
        assert_eq!(bytes, mpmct_string(&gates, 32).as_bytes());
        assert_eq!(num_wires, 32);
        assert_eq!(roundtripped, gates);
    }
}
