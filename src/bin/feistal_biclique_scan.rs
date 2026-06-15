use local_mixing::replace::transpositions::Transpositions;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::{env, fs};

type Gate = [u16; 3];

#[derive(Clone, Copy, Debug)]
enum Op {
    Swap { lo: u16, hi: u16, neg: u16 },
    Not { wire: u16 },
}

#[derive(Clone, Copy, Debug)]
struct LGate {
    pos: usize,
    a: u16,
    b: u16,
    c: u16,
    bf: u8,
    cf: u8,
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
    let zero_out = gates.iter().fold(0usize, |state, &gate| eval_gate(state, gate));
    let mut xors = vec![0u8; n];
    let mut from = vec![usize::MAX; n];
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

fn peel(gates: &[Gate], samples: usize) -> Vec<LGate> {
    let n = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    let (swaps, nots) = build_templates(samples);
    let mut label: Vec<u16> = (0..n as u16).collect();
    let mut flip = vec![0u8; n];
    let mut logical = Vec::new();
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
            i += len;
            continue;
        }
        if let Some((len, Op::Not { wire })) = matched {
            flip[wire as usize] ^= 1;
            i += len;
            continue;
        }
        let [a, b, c] = gates[i];
        logical.push(LGate {
            pos: logical.len(),
            a: label[a as usize],
            b: label[b as usize],
            c: label[c as usize],
            bf: flip[b as usize],
            cf: flip[c as usize],
        });
        i += 1;
    }
    logical
}

fn choose3(vals: &[u16]) -> Vec<[u16; 3]> {
    let mut out = Vec::new();
    for i in 0..vals.len() {
        for j in i + 1..vals.len() {
            for k in j + 1..vals.len() {
                out.push([vals[i], vals[j], vals[k]]);
            }
        }
    }
    out
}

fn count_window(logical: &[LGate], span: usize, clean_only: bool) -> (usize, Vec<String>) {
    let mut by_target: HashMap<u16, VecDeque<LGate>> = HashMap::new();
    let mut count = 0usize;
    let mut examples = Vec::new();
    for &g in logical {
        let q = by_target.entry(g.a).or_default();
        while q.front().is_some_and(|old| g.pos - old.pos >= span) {
            q.pop_front();
        }
        q.push_back(g);
        if q.len() < 9 {
            continue;
        }
        let filtered: Vec<_> = q
            .iter()
            .copied()
            .filter(|x| !clean_only || (x.bf == 0 && x.cf == 0))
            .collect();
        if filtered.len() < 9 {
            continue;
        }
        let mut bs: Vec<u16> = filtered.iter().map(|x| x.b).collect();
        let mut cs: Vec<u16> = filtered.iter().map(|x| x.c).collect();
        bs.sort_unstable();
        bs.dedup();
        cs.sort_unstable();
        cs.dedup();
        if bs.len() < 3 || cs.len() < 3 {
            continue;
        }
        let pairs: HashSet<(u16, u16)> = filtered.iter().map(|x| (x.b, x.c)).collect();
        let btriples = choose3(&bs);
        let ctriples = choose3(&cs);
        let mut found = None;
        'outer: for bt in &btriples {
            for ct in &ctriples {
                if bt
                    .iter()
                    .all(|&b| ct.iter().all(|&c| pairs.contains(&(b, c))))
                {
                    found = Some((*bt, *ct));
                    break 'outer;
                }
            }
        }
        if let Some((bt, ct)) = found {
            count += 1;
            if examples.len() < 8 {
                let start = filtered.iter().map(|x| x.pos).min().unwrap_or(g.pos);
                examples.push(format!("end={} start={} target={} b={:?} c={:?}", g.pos, start, g.a, bt, ct));
            }
        }
    }
    (count, examples)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: feistal_biclique_scan <circuit> [samples=200000]");
        std::process::exit(2);
    }
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let gates = parse(&args[1]);
    let logical = peel(&gates, samples);
    println!("input_gates {}", gates.len());
    println!("logical_after_samf_peel {}", logical.len());
    for span in [16usize, 32, 64, 128, 256, 512, 1024, 2048] {
        for clean_only in [true, false] {
            let (count, examples) = count_window(&logical, span, clean_only);
            println!(
                "span {} clean_only {} sg3_biclique_candidates {}",
                span, clean_only, count
            );
            for ex in examples {
                println!("  example {ex}");
            }
        }
    }
}
