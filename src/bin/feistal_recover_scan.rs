use local_mixing::replace::transpositions::Transpositions;
use std::collections::{BTreeSet, HashMap};
use std::{env, fs};

type Gate = [u16; 3];

#[derive(Clone, Copy, Debug)]
enum Op {
    Swap { lo: u16, hi: u16, neg: u16 },
    Not { wire: u16 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LGate {
    a: u16,
    b: u16,
    c: u16,
    bf: u8,
    cf: u8,
}

const RG1: [Gate; 6] = [
    [1, 2, 3],
    [0, 3, 2],
    [3, 1, 0],
    [2, 0, 1],
    [0, 3, 2],
    [1, 2, 3],
];

const RG2: [Gate; 6] = [
    [0, 3, 2],
    [1, 0, 2],
    [2, 0, 3],
    [2, 3, 0],
    [1, 3, 2],
    [3, 0, 2],
];

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
                    assert_eq!(wires.len(), 3, "bad gate near {}", gates.len());
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

fn normalize_plain(gates: &[Gate]) -> Vec<Gate> {
    let mut map = HashMap::new();
    let mut next = 0u16;
    gates
        .iter()
        .map(|gate| {
            let mut out = [0u16; 3];
            for (i, &w) in gate.iter().enumerate() {
                out[i] = *map.entry(w).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                });
            }
            out
        })
        .collect()
}

fn count_plain_template(logical: &[LGate], template: &[Gate]) -> usize {
    if logical.len() < template.len() {
        return 0;
    }
    let tmpl = normalize_plain(template);
    let len = template.len();
    let mut count = 0;
    for i in 0..=logical.len() - len {
        if logical[i..i + len].iter().any(|g| g.bf != 0 || g.cf != 0) {
            continue;
        }
        let window: Vec<Gate> = logical[i..i + len].iter().map(|g| [g.a, g.b, g.c]).collect();
        if normalize_plain(&window) == tmpl {
            count += 1;
        }
    }
    count
}

fn count_sg3(logical: &[LGate]) -> usize {
    let mut count = 0usize;
    for win in logical.windows(9) {
        if win.iter().any(|g| g.bf != 0 || g.cf != 0) {
            continue;
        }
        let target = win[0].a;
        if win.iter().any(|g| g.a != target) {
            continue;
        }
        let bs = [win[0].b, win[3].b, win[6].b];
        let cs = [win[0].c, win[1].c, win[2].c];
        if bs.iter().collect::<std::collections::HashSet<_>>().len() != 3
            || cs.iter().collect::<std::collections::HashSet<_>>().len() != 3
        {
            continue;
        }
        let mut ok = true;
        for bi in 0..3 {
            for ci in 0..3 {
                let g = win[3 * bi + ci];
                if g.b != bs[bi] || g.c != cs[ci] {
                    ok = false;
                }
            }
        }
        if ok {
            count += 1;
        }
    }
    count
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: feistal_recover_scan <circuit> [samples=200000]");
        std::process::exit(2);
    }
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let gates = parse(&args[1]);
    let n = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    let (swaps, nots) = build_templates(samples);
    eprintln!("templates swaps={} nots={}", swaps.len(), nots.len());

    let mut label: Vec<u16> = (0..n as u16).collect();
    let mut flip = vec![0u8; n];
    let mut logical = Vec::new();
    let mut skipped_swaps = 0usize;
    let mut skipped_nots = 0usize;
    let mut dirty = 0usize;
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
        let lg = LGate {
            a: label[a as usize],
            b: label[b as usize],
            c: label[c as usize],
            bf: flip[b as usize],
            cf: flip[c as usize],
        };
        if lg.bf != 0 || lg.cf != 0 {
            dirty += 1;
        }
        logical.push(lg);
        i += 1;
    }

    let flips = flip.iter().filter(|&&x| x != 0).count();
    let nonid = label
        .iter()
        .enumerate()
        .filter(|(idx, w)| *idx != **w as usize)
        .count();
    println!("input_gates {}", gates.len());
    println!("emitted_logical {}", logical.len());
    println!("skipped_swaps {}", skipped_swaps);
    println!("skipped_nots {}", skipped_nots);
    println!("dirty_controls {}", dirty);
    println!("final_nonid {}", nonid);
    println!("final_flips {}", flips);
    println!("sg3_windows {}", count_sg3(&logical));
    println!("rg1_windows {}", count_plain_template(&logical, &RG1));
    println!("rg2_windows {}", count_plain_template(&logical, &RG2));
}
