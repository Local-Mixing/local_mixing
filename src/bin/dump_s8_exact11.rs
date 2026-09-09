//! Enumerate an exact-length 11-gate 4-wire r57 circuit for every `π ∈ S_8`
//! realized as `π ⊗ I` (ancilla = wire 3).

use std::fs::File;
use std::io::{BufWriter, Write};

use local_mixing::circuit::{base_gates, Gate};
use rustc_hash::FxHashMap;

const S8_ORDER: usize = 40320;

fn identity_perm16() -> u64 {
    let mut packed = 0u64;
    for i in 0..16u64 {
        packed |= i << (4 * i);
    }
    packed
}

fn apply_gate_perm16(packed: u64, gate: [u16; 3]) -> u64 {
    let mut out = 0u64;
    for i in 0..16 {
        let x = ((packed >> (4 * i)) & 0xF) as usize;
        let y = Gate::evaluate_index(x, gate) as u64;
        out |= y << (4 * i);
    }
    out
}

fn compose_perm16(p: u64, q: u64) -> u64 {
    let mut out = 0u64;
    for i in 0..16u64 {
        let mid = (q >> (4 * i)) & 0xF;
        let dest = (p >> (4 * mid)) & 0xF;
        out |= dest << (4 * i);
    }
    out
}

fn invert_perm16(packed: u64) -> u64 {
    let mut out = 0u64;
    for i in 0..16u64 {
        let dest = (packed >> (4 * i)) & 0xF;
        out |= i << (4 * dest);
    }
    out
}

fn induced_pi_otimes_i(packed: u64) -> Option<[u8; 8]> {
    let mut pi = [0u8; 8];
    let mut seen = 0u8;
    for i in 0..8 {
        let lo = ((packed >> (4 * i)) & 0xF) as u8;
        let hi = ((packed >> (4 * (i + 8))) & 0xF) as u8;
        if lo >= 8 || hi != lo + 8 {
            return None;
        }
        seen |= 1 << lo;
        pi[i] = lo;
    }
    (seen == 0xFF).then_some(pi)
}

fn pack_pi_otimes_i(pi: [u8; 8]) -> u64 {
    let mut packed = 0u64;
    for i in 0..8 {
        let v = pi[i] as u64;
        packed |= v << (4 * i);
        packed |= (v + 8) << (4 * (i + 8));
    }
    packed
}

fn lehmer_rank_s8(images: [u8; 8]) -> usize {
    let mut remaining = 0xFFu16;
    let mut rank = 0usize;
    for (i, &value) in images.iter().enumerate() {
        let bit = 1u16 << value;
        let smaller = (remaining & (bit - 1)).count_ones() as usize;
        rank = rank * (8 - i) + smaller;
        remaining &= !bit;
    }
    rank
}

fn unrank_s8(mut rank: usize, out: &mut [u8; 8]) {
    let mut items: Vec<u8> = (0..8).collect();
    for i in 0..8 {
        let f = factorial(7 - i);
        let idx = rank / f;
        rank %= f;
        out[i] = items.remove(idx);
    }
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn s8_perm_is_even(images: [u8; 8]) -> bool {
    let mut seen = [false; 8];
    let mut even = true;
    for start in 0..8 {
        if seen[start] {
            continue;
        }
        let mut len = 0usize;
        let mut j = start;
        while !seen[j] {
            seen[j] = true;
            j = images[j] as usize;
            len += 1;
        }
        if len % 2 == 0 {
            even = !even;
        }
    }
    even
}

/// Canonical disjoint-cycle notation, 0-based, 1-cycles omitted.
fn reduced_cycle_form(pi: [u8; 8]) -> String {
    let mut seen = [false; 8];
    let mut parts = Vec::new();
    for start in 0..8 {
        if seen[start] {
            continue;
        }
        let mut cyc = Vec::new();
        let mut j = start;
        while !seen[j] {
            seen[j] = true;
            cyc.push(j);
            j = pi[j] as usize;
        }
        if cyc.len() > 1 {
            parts.push(format!(
                "({})",
                cyc.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
    }
    if parts.is_empty() {
        "()".to_string()
    } else {
        parts.concat()
    }
}

fn eval_word(word: &[u8], gens: &[[u16; 3]]) -> u64 {
    let mut p = identity_perm16();
    for &gi in word {
        p = apply_gate_perm16(p, gens[gi as usize]);
    }
    p
}

fn pad_to_11(word: &[u8]) -> [u8; 11] {
    let d = word.len();
    assert!(d <= 11 && (11 - d) % 2 == 0, "cannot pad length {d} to 11");
    let mut out = [0u8; 11];
    out[..d].copy_from_slice(word);
    let h = if d > 0 && word[d - 1] == 0 { 1u8 } else { 0u8 };
    let mut i = d;
    while i < 11 {
        out[i] = h;
        out[i + 1] = h;
        i += 2;
    }
    out
}

fn format_circuit(word: &[u8; 11], gens: &[[u16; 3]]) -> String {
    word.iter()
        .map(|&gi| {
            let [a, b, c] = gens[gi as usize];
            format!("{a}{b}{c}")
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn exact_layer<const K: usize>(gens: &[[u16; 3]]) -> FxHashMap<u64, [u8; K]> {
    let mut prev: FxHashMap<u64, [u8; 11]> = FxHashMap::default();
    prev.insert(identity_perm16(), [0u8; 11]);
    for depth in 0..K {
        let mut next = FxHashMap::with_capacity_and_hasher(
            prev.len() * gens.len(),
            Default::default(),
        );
        for (p, word) in &prev {
            for (gi, &gate) in gens.iter().enumerate() {
                let q = apply_gate_perm16(*p, gate);
                next.entry(q).or_insert_with(|| {
                    let mut w = *word;
                    w[depth] = gi as u8;
                    w
                });
            }
        }
        prev = next;
    }
    prev.into_iter()
        .map(|(p, w)| {
            let mut short = [0u8; K];
            short.copy_from_slice(&w[..K]);
            (p, short)
        })
        .collect()
}

fn six_words_for_all_gens(gens: &[[u16; 3]]) -> Vec<[u8; 6]> {
    // 6-word for (0,1,2), then relabel wires.
    let proto_gates: [[u16; 3]; 6] = [
        [0, 1, 3],
        [0, 2, 1],
        [2, 1, 3],
        [0, 1, 2],
        [0, 2, 1],
        [2, 1, 3],
    ];
    let gate_index = |g: [u16; 3]| -> u8 {
        gens.iter()
            .position(|&h| h == g)
            .unwrap_or_else(|| panic!("missing generator {g:?}")) as u8
    };
    let proto: [u8; 6] = proto_gates.map(gate_index);
    let proto_idx = gate_index([0, 1, 2]);
    let id = identity_perm16();
    let want = apply_gate_perm16(id, [0, 1, 2]);
    assert_eq!(eval_word(&proto, gens), want, "prototype 6-word != (0,1,2)");
    let seven: Vec<u8> = proto.iter().copied().chain([proto_idx]).collect();
    assert_eq!(eval_word(&seven, gens), id, "6-word ++ G is not a 7-identity");

    let mut out = vec![[0u8; 6]; gens.len()];
    for (gi, &[a, b, c]) in gens.iter().enumerate() {
        let d = 6 - a - b - c;
        let sigma = [a, b, c, d];
        let relabel = |g: [u16; 3]| -> [u16; 3] {
            [sigma[g[0] as usize], sigma[g[1] as usize], sigma[g[2] as usize]]
        };
        let word = proto_gates.map(|g| gate_index(relabel(g)));
        let got = eval_word(&word, gens);
        let expect = apply_gate_perm16(id, [a, b, c]);
        assert_eq!(got, expect, "relabeled 6-word failed for {:?}", [a, b, c]);
        out[gi] = word;
    }
    out
}

fn replace_first_then_pad(word: &[u8], six: &[[u8; 6]]) -> [u8; 11] {
    assert!(!word.is_empty());
    let mut v = Vec::with_capacity(word.len() + 5);
    v.extend_from_slice(&six[word[0] as usize]);
    v.extend_from_slice(&word[1..]);
    pad_to_11(&v)
}

fn main() {
    let gens = base_gates(4);
    assert_eq!(gens.len(), 24);
    let gens3: Vec<u8> = gens
        .iter()
        .enumerate()
        .filter(|(_, g)| g.iter().all(|&w| w < 3))
        .map(|(i, _)| i as u8)
        .collect();
    assert_eq!(gens3.len(), 6);

    let six = six_words_for_all_gens(&gens);
    let g012 = gens
        .iter()
        .position(|&g| g == [0, 1, 2])
        .expect("missing (0,1,2)") as u8;

    let mut three_wire: Vec<Option<Vec<u8>>> = vec![None; S8_ORDER];
    three_wire[0] = Some(Vec::new());
    let mut packed_at = vec![0u64; S8_ORDER];
    packed_at[0] = identity_perm16();
    let mut frontier = vec![0usize];
    let mut qhead = 0usize;
    while qhead < frontier.len() {
        let idx = frontier[qhead];
        qhead += 1;
        let d = three_wire[idx].as_ref().unwrap().len();
        if d >= 10 {
            continue;
        }
        let p = packed_at[idx];
        for &gi in &gens3 {
            let child = apply_gate_perm16(p, gens[gi as usize]);
            let pi = induced_pi_otimes_i(child).expect("3-wire gate left π⊗I");
            let nidx = lehmer_rank_s8(pi);
            if three_wire[nidx].is_none() {
                let mut w = three_wire[idx].clone().unwrap();
                w.push(gi);
                three_wire[nidx] = Some(w);
                packed_at[nidx] = child;
                frontier.push(nidx);
            }
        }
    }
    let n3 = three_wire.iter().filter(|w| w.is_some()).count();
    assert_eq!(n3, S8_ORDER, "3-wire BFS missed some of S8");

    let mut circuits: Vec<Option<[u8; 11]>> = vec![None; S8_ORDER];
    let mut source = vec![""; S8_ORDER];
    let mut n_odd = 0usize;
    let mut n_id = 0usize;
    let mut n_even_short = 0usize;

    for idx in 0..S8_ORDER {
        let mut pi = [0u8; 8];
        unrank_s8(idx, &mut pi);
        let w = three_wire[idx].as_ref().unwrap();
        let d = w.len();
        if idx == 0 {
            let mut seven = Vec::from(six[g012 as usize]);
            seven.push(g012);
            circuits[idx] = Some(pad_to_11(&seven));
            source[idx] = "identity-7+pad";
            n_id += 1;
            continue;
        }
        if !s8_perm_is_even(pi) {
            assert!(d % 2 == 1 && d <= 9, "odd π has 3-wire length {d}");
            circuits[idx] = Some(pad_to_11(w));
            source[idx] = "odd-3wire+pad";
            n_odd += 1;
            continue;
        }
        if d <= 6 {
            assert!(d % 2 == 0 && d >= 2);
            circuits[idx] = Some(replace_first_then_pad(w, &six));
            source[idx] = "even-3wire+6word";
            n_even_short += 1;
        }
    }

    let leftover: Vec<usize> = (0..S8_ORDER)
        .filter(|&i| circuits[i].is_none())
        .collect();
    eprintln!(
        "constructed {} (odd {}) (id {}) (even≤6 {}); leftover {}",
        S8_ORDER - leftover.len(),
        n_odd,
        n_id,
        n_even_short,
        leftover.len()
    );

    if !leftover.is_empty() {
        eprintln!("building exact L2, L3, L4…");
        let l2 = exact_layer::<2>(&gens);
        let l3 = exact_layer::<3>(&gens);
        let l4 = exact_layer::<4>(&gens);
        eprintln!("|L2|={} |L3|={} |L4|={}", l2.len(), l3.len(), l4.len());
        let mut n_l2 = 0usize;
        let mut n_l4 = 0usize;
        let mut n_l6 = 0usize;
        let mut still = Vec::new();
        for &idx in &leftover {
            let t = pack_pi_otimes_i({
                let mut pi = [0u8; 8];
                unrank_s8(idx, &mut pi);
                pi
            });
            if let Some(w) = l2.get(&t) {
                circuits[idx] = Some(replace_first_then_pad(w, &six));
                source[idx] = "even-L2+6word";
                n_l2 += 1;
                continue;
            }
            if let Some(w) = l4.get(&t) {
                circuits[idx] = Some(replace_first_then_pad(w, &six));
                source[idx] = "even-L4+6word";
                n_l4 += 1;
                continue;
            }
            let mut hit6 = None;
            for (&p3, w3) in &l3 {
                let q = compose_perm16(invert_perm16(p3), t);
                if let Some(w3b) = l3.get(&q) {
                    let mut word = Vec::with_capacity(6);
                    word.extend_from_slice(w3b);
                    word.extend_from_slice(w3);
                    hit6 = Some(word);
                    break;
                }
            }
            if let Some(w) = hit6 {
                circuits[idx] = Some(replace_first_then_pad(&w, &six));
                source[idx] = "even-L6+6word";
                n_l6 += 1;
                continue;
            }
            still.push(idx);
        }
        eprintln!(
            "L2 {} L4 {} L6 {}; MITM leftover {}",
            n_l2, n_l4, n_l6, still.len()
        );

        if !still.is_empty() {
            eprintln!("building exact L5 ({} targets)…", still.len());
            let l5 = exact_layer::<5>(&gens);
            eprintln!("|L5|={}", l5.len());
            for (k, &idx) in still.iter().enumerate() {
                let t = pack_pi_otimes_i({
                    let mut pi = [0u8; 8];
                    unrank_s8(idx, &mut pi);
                    pi
                });
                let mut found = None;
                'outer: for (&p5, w5) in &l5 {
                    let p6 = compose_perm16(invert_perm16(p5), t);
                    for (gi, &gate) in gens.iter().enumerate() {
                        let s = apply_gate_perm16(p6, gate);
                        if let Some(ws) = l5.get(&s) {
                            let mut word = Vec::with_capacity(11);
                            word.extend_from_slice(ws);
                            word.push(gi as u8);
                            word.extend_from_slice(w5);
                            found = Some(word);
                            break 'outer;
                        }
                    }
                }
                let word = found.unwrap_or_else(|| {
                    panic!("no 11-gate factorization for leftover #{k} idx={idx}")
                });
                assert_eq!(word.len(), 11);
                let mut arr = [0u8; 11];
                arr.copy_from_slice(&word);
                circuits[idx] = Some(arr);
                source[idx] = "MITM-L5-L6";
                if (k + 1) % 4 == 0 || k + 1 == still.len() {
                    eprintln!("  MITM {}/{}", k + 1, still.len());
                }
            }
        }
    }

    eprintln!("verifying 40320 circuits…");
    let mut seen = vec![false; S8_ORDER];
    for idx in 0..S8_ORDER {
        let word = circuits[idx].expect("missing circuit");
        let packed = eval_word(&word, &gens);
        let pi = induced_pi_otimes_i(packed)
            .unwrap_or_else(|| panic!("circuit {idx} is not π⊗I: {}", format_circuit(&word, &gens)));
        let got = lehmer_rank_s8(pi);
        assert_eq!(got, idx, "circuit for rank {idx} realized rank {got}");
        assert!(!seen[idx], "duplicate rank {idx}");
        seen[idx] = true;
        let mut expect = [0u8; 8];
        unrank_s8(idx, &mut expect);
        assert_eq!(pi, expect);
    }
    assert!(seen.iter().all(|&b| b));
    eprintln!("ok: 40320 distinct π⊗I, all of S8 covered");

    let path = "s8_exact11_fourwire.txt";
    let mut out = BufWriter::new(File::create(path).expect("create output"));
    writeln!(
        out,
        "# Exact-length 11-gate 4-wire r57 circuits for every π in S8 as π⊗I (ancilla wire 3)."
    )
    .unwrap();
    writeln!(
        out,
        "# Gate abc = target a, positive control b, negative control c. Points 0..7 are wires 0,1,2."
    )
    .unwrap();
    writeln!(
        out,
        "# Columns: reduced cycle form of π, then the 11-gate word."
    )
    .unwrap();
    for idx in 0..S8_ORDER {
        let mut pi = [0u8; 8];
        unrank_s8(idx, &mut pi);
        let word = circuits[idx].unwrap();
        writeln!(
            out,
            "{}\t{}",
            reduced_cycle_form(pi),
            format_circuit(&word, &gens)
        )
        .unwrap();
    }
    out.flush().unwrap();
    eprintln!("wrote {path}");

    let mut src_counts = FxHashMap::default();
    for s in &source {
        *src_counts.entry(*s).or_insert(0usize) += 1;
    }
    eprintln!("sources: {src_counts:?}");
}
