use local_mixing::replace::transpositions::Transpositions;
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::io::{Seek, SeekFrom, Write};

type Gate = [u16; 3];

#[derive(Clone, Copy, Debug)]
enum Op {
    Swap { lo: u16, hi: u16, neg: u16 },
    Not { wire: u16 },
}

struct Cnf {
    out: fs::File,
    clauses: u64,
}

fn usage(program: &str) -> ! {
    eprintln!(
        "usage: {program} <circuit> <out.cnf> [low_bits=64] [total_wires=128] [flip_bit=63] [input_bit=0] [samples=200000]"
    );
    std::process::exit(2);
}

fn val(c: char) -> Option<u16> {
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
        .chars()
        .position(|x| x == c)
        .map(|x| x as u16)
}

fn parse(path: &str) -> Vec<Gate> {
    let input = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    let mut gates = Vec::new();
    let mut wires = Vec::new();
    let mut overflow = 0u16;
    for ch in input.chars() {
        match ch {
            ';' => {
                if !wires.is_empty() {
                    assert_eq!(wires.len(), 3, "bad gate in {path}");
                    gates.push([wires[0], wires[1], wires[2]]);
                    wires.clear();
                }
                overflow = 0;
            }
            '~' => overflow += 1,
            c if c.is_whitespace() => {}
            c => {
                let base = val(c).unwrap_or_else(|| panic!("bad wire char {c:?}"));
                wires.push(base + 83 * overflow);
                overflow = 0;
            }
        }
    }
    gates
}

fn localize(window: &[Gate], local_to_phys: &[u16]) -> Option<Vec<Gate>> {
    let mut phys_to_local = HashMap::new();
    for (local, &phys) in local_to_phys.iter().enumerate() {
        phys_to_local.insert(phys, local as u16);
    }
    let mut out = Vec::with_capacity(window.len());
    for &[a, b, c] in window {
        out.push([
            *phys_to_local.get(&a)?,
            *phys_to_local.get(&b)?,
            *phys_to_local.get(&c)?,
        ]);
    }
    Some(out)
}

fn permutations(values: &[u16], out: &mut Vec<Vec<u16>>) {
    fn rec(values: &mut [u16], out: &mut Vec<Vec<u16>>, i: usize) {
        if i == values.len() {
            out.push(values.to_vec());
            return;
        }
        for j in i..values.len() {
            values.swap(i, j);
            rec(values, out, i + 1);
            values.swap(i, j);
        }
    }
    let mut values = values.to_vec();
    rec(&mut values, out, 0);
}

fn build_templates(samples: usize) -> (BTreeSet<Vec<Gate>>, BTreeSet<Vec<Gate>>) {
    let mut swaps = BTreeSet::new();
    let mut nots = BTreeSet::new();
    for neg in 0..=3u16 {
        for _ in 0..samples {
            swaps.insert(Transpositions::gen_gates_swap(3, (1, 2, neg)));
            swaps.insert(Transpositions::gen_gates_swap(4, (1, 2, neg)));
        }
    }
    for _ in 0..samples {
        nots.insert(Transpositions::gen_gates_not(4, 1));
    }
    (swaps, nots)
}

fn eval_gate(mut state: usize, gate: Gate) -> usize {
    let c1 = (state >> gate[1]) & 1;
    let c2 = (state >> gate[2]) & 1;
    if (c1 | (1 ^ c2)) != 0 {
        state ^= 1usize << gate[0];
    }
    state
}

fn classify_local_swap(gates: &[Gate]) -> Option<u16> {
    let n = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|x| x as usize + 1)?;
    if !(3..=4).contains(&n) {
        return None;
    }
    let mut xors = vec![0u8; n];
    let mut from = vec![usize::MAX; n];
    let zero_out = gates.iter().fold(0usize, |state, &gate| eval_gate(state, gate));
    for out in 0..n {
        let bit0 = ((zero_out >> out) & 1) as u8;
        let mut found = None;
        for input in 0..n {
            let ok = (0..(1usize << n)).all(|x| {
                let y = gates.iter().fold(x, |state, &gate| eval_gate(state, gate));
                ((y >> out) & 1) == (((x >> input) & 1) ^ bit0 as usize)
            });
            if ok {
                if found.is_some() {
                    return None;
                }
                found = Some(input);
            }
        }
        from[out] = found?;
        xors[out] = bit0;
    }
    if from[1] != 2 || from[2] != 1 {
        return None;
    }
    for i in 0..n {
        if i != 1 && i != 2 && xors[i] != 0 {
            return None;
        }
    }
    Some(match (xors[1] != 0, xors[2] != 0) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    })
}

fn match_exact(window: &[Gate], swaps: &BTreeSet<Vec<Gate>>, nots: &BTreeSet<Vec<Gate>>) -> Option<Op> {
    let mut used = Vec::new();
    for gate in window {
        for &w in gate {
            if !used.contains(&w) {
                used.push(w);
            }
        }
    }
    if !(3..=4).contains(&used.len()) {
        return None;
    }
    let mut perms = Vec::new();
    permutations(&used, &mut perms);
    for p in perms {
        if let Some(local) = localize(window, &p) {
            if swaps.contains(&local) {
                let neg = classify_local_swap(&local)?;
                return Some(Op::Swap {
                    lo: p[1],
                    hi: p[2],
                    neg,
                });
            }
            if nots.contains(&local) {
                return Some(Op::Not { wire: p[1] });
            }
        }
    }
    None
}

fn apply_neg(mask: &mut [u8], lo: usize, hi: usize, neg: u16) {
    mask.swap(lo, hi);
    match neg {
        0 => {}
        1 => mask[lo] ^= 1,
        2 => mask[hi] ^= 1,
        3 => {
            mask[lo] ^= 1;
            mask[hi] ^= 1;
        }
        _ => unreachable!(),
    }
}

impl Cnf {
    fn new(path: &str) -> Self {
        let mut out = fs::File::create(path).unwrap_or_else(|e| panic!("failed to create {path}: {e}"));
        writeln!(out, "p cnf 0000000000 0000000000").unwrap();
        Self { out, clauses: 0 }
    }

    fn clause(&mut self, lits: &[i32]) {
        for lit in lits {
            write!(self.out, "{lit} ").unwrap();
        }
        writeln!(self.out, "0").unwrap();
        self.clauses += 1;
    }

    fn finish(mut self, path: &str, vars: i32) {
        self.out.flush().unwrap();
        let mut header = format!("p cnf {vars} {}", self.clauses);
        assert!(header.len() <= 27, "CNF header too long");
        while header.len() < 27 {
            header.push(' ');
        }
        header.push('\n');
        self.out.seek(SeekFrom::Start(0)).unwrap();
        self.out.write_all(header.as_bytes()).unwrap();
        eprintln!("vars {vars} clauses {}", self.clauses);
        eprintln!("wrote {path}");
    }
}

fn emit_r57(cnf: &mut Cnf, cur: &mut [i32], vars: &mut i32, a: usize, b: usize, c: usize, bf: u8, cf: u8) {
    let a0 = cur[a];
    let b0 = cur[b];
    let c0 = cur[c];
    *vars += 1;
    let z = *vars;
    match (bf, cf) {
        (0, 0) => {
            cnf.clause(&[b0, -c0, -a0, z]);
            cnf.clause(&[b0, -c0, a0, -z]);
            cnf.clause(&[-b0, -a0, -z]);
            cnf.clause(&[-b0, a0, z]);
            cnf.clause(&[c0, -a0, -z]);
            cnf.clause(&[c0, a0, z]);
        }
        (1, 1) => {
            cnf.clause(&[c0, -b0, -a0, z]);
            cnf.clause(&[c0, -b0, a0, -z]);
            cnf.clause(&[-c0, -a0, -z]);
            cnf.clause(&[-c0, a0, z]);
            cnf.clause(&[b0, -a0, -z]);
            cnf.clause(&[b0, a0, z]);
        }
        (0, 1) => {
            cnf.clause(&[b0, c0, -a0, z]);
            cnf.clause(&[b0, c0, a0, -z]);
            cnf.clause(&[-b0, -a0, -z]);
            cnf.clause(&[-b0, a0, z]);
            cnf.clause(&[-c0, -a0, -z]);
            cnf.clause(&[-c0, a0, z]);
        }
        (1, 0) => {
            cnf.clause(&[-b0, -c0, -a0, z]);
            cnf.clause(&[-b0, -c0, a0, -z]);
            cnf.clause(&[b0, -a0, -z]);
            cnf.clause(&[b0, a0, z]);
            cnf.clause(&[c0, -a0, -z]);
            cnf.clause(&[c0, a0, z]);
        }
        _ => unreachable!(),
    }
    cur[a] = z;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage(&args[0]);
    }
    let circuit_path = &args[1];
    let cnf_path = &args[2];
    let low_bits: usize = args.get(3).map_or(Ok(64), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
    let total_wires: usize = args.get(4).map_or(Ok(128), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
    let flip_bit: usize = args.get(5).map_or(Ok(63), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
    let input_bit: u8 = args.get(6).map_or(Ok(0), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));
    let samples: usize = args.get(7).map_or(Ok(200_000), |s| s.parse()).unwrap_or_else(|_| usage(&args[0]));

    let (swaps, nots) = build_templates(samples);
    eprintln!("templates swaps={} nots={}", swaps.len(), nots.len());
    let gates = parse(circuit_path);
    eprintln!("input gates {}", gates.len());

    let mut cnf = Cnf::new(cnf_path);
    let mut vars = total_wires as i32;
    let mut cur: Vec<i32> = (1..=vars).collect();
    for w in low_bits..total_wires {
        cnf.clause(&[-cur[w]]);
    }

    let mut label: Vec<u16> = (0..total_wires as u16).collect();
    let mut flip = vec![0u8; total_wires];
    let mut skipped_swaps = 0usize;
    let mut skipped_nots = 0usize;
    let mut emitted = 0usize;
    let mut dirty_controls = 0usize;
    let mut i = 0usize;
    while i < gates.len() {
        let mut matched = None;
        for len in (6..=12).rev() {
            if i + len <= gates.len() {
                if let Some(op) = match_exact(&gates[i..i + len], &swaps, &nots) {
                    matched = Some((len, op));
                    break;
                }
            }
        }
        if let Some((len, Op::Swap { lo, hi, neg })) = matched {
            label.swap(lo as usize, hi as usize);
            apply_neg(&mut flip, lo as usize, hi as usize, neg);
            skipped_swaps += 1;
            i += len;
            continue;
        }
        if let Some((len, Op::Not { wire })) = matched {
            flip[wire as usize] ^= 1;
            skipped_nots += 1;
            i += len;
            continue;
        }

        let [a, b, c] = gates[i];
        if flip[b as usize] != 0 || flip[c as usize] != 0 {
            dirty_controls += 1;
        }
        emit_r57(
            &mut cnf,
            &mut cur,
            &mut vars,
            label[a as usize] as usize,
            label[b as usize] as usize,
            label[c as usize] as usize,
            flip[b as usize],
            flip[c as usize],
        );
        emitted += 1;
        i += 1;
    }

    for w in 0..low_bits {
        let x = (w + 1) as i32;
        let y = cur[label[w] as usize];
        let yf = flip[w] != 0;
        if w == flip_bit {
            let want_input = input_bit != 0;
            cnf.clause(&[if want_input { x } else { -x }]);
            let want_output = !want_input;
            let want_y = want_output ^ yf;
            cnf.clause(&[if want_y { y } else { -y }]);
        } else if yf {
            cnf.clause(&[-x, -y]);
            cnf.clause(&[x, y]);
        } else {
            cnf.clause(&[-x, y]);
            cnf.clause(&[x, -y]);
        }
    }

    let final_nonid = label
        .iter()
        .enumerate()
        .filter(|(i, w)| *i != **w as usize)
        .count();
    let final_flips = flip.iter().filter(|&&x| x != 0).count();
    eprintln!(
        "emitted {emitted} skipped_swaps {skipped_swaps} skipped_nots {skipped_nots} dirty_controls {dirty_controls} final_nonid {final_nonid} final_flips {final_flips}"
    );
    cnf.finish(cnf_path, vars);
}
