use local_mixing::replace::transpositions::Transpositions;
use std::collections::{HashMap, HashSet};
use std::{env, fs};

type Gate = [u16; 3];
type Sig = Vec<Gate>;

const SG: [Gate; 6] = [
    [4, 5, 6],
    [0, 4, 6],
    [0, 5, 4],
    [4, 5, 6],
    [0, 6, 3],
    [0, 3, 5],
];

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

const RG3: [Gate; 2] = [[0, 2, 3], [1, 2, 3]];

#[derive(Default)]
struct MatchStats {
    raw: usize,
    greedy: usize,
    by_len: HashMap<usize, usize>,
    examples: Vec<usize>,
}

#[derive(Default)]
struct NearStats {
    missing_one_gate: usize,
    extra_one_gate: usize,
    substituted_one_gate: usize,
    any: usize,
    examples: Vec<String>,
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

fn normalize(gates: &[Gate]) -> Sig {
    let mut map: HashMap<u16, u16> = HashMap::new();
    let mut next = 0u16;
    let mut out = Vec::with_capacity(gates.len());
    for gate in gates {
        let mut ng = [0u16; 3];
        for (i, &wire) in gate.iter().enumerate() {
            let id = *map.entry(wire).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
            ng[i] = id;
        }
        out.push(ng);
    }
    out
}

fn count_matches(gates: &[Gate], sets: &HashMap<&'static str, HashSet<Sig>>) -> HashMap<&'static str, MatchStats> {
    let mut lengths: Vec<usize> = sets
        .values()
        .flat_map(|set| set.iter().map(|sig| sig.len()))
        .collect();
    lengths.sort_unstable();
    lengths.dedup();

    let mut stats: HashMap<&'static str, MatchStats> =
        sets.keys().copied().map(|name| (name, MatchStats::default())).collect();
    let mut starts: HashMap<&'static str, Vec<(usize, usize)>> =
        sets.keys().copied().map(|name| (name, Vec::new())).collect();

    for i in 0..gates.len() {
        for &len in &lengths {
            if i + len > gates.len() {
                continue;
            }
            let sig = normalize(&gates[i..i + len]);
            for (&name, set) in sets {
                if set.contains(&sig) {
                    let stat = stats.get_mut(name).unwrap();
                    stat.raw += 1;
                    *stat.by_len.entry(len).or_insert(0) += 1;
                    if stat.examples.len() < 8 {
                        stat.examples.push(i);
                    }
                    starts.get_mut(name).unwrap().push((i, len));
                }
            }
        }
    }

    for (&name, hits) in &mut starts {
        hits.sort_by_key(|&(start, len)| (start, std::cmp::Reverse(len)));
        let mut next_allowed = 0usize;
        let mut count = 0usize;
        for &(start, len) in hits.iter() {
            if start >= next_allowed {
                count += 1;
                next_allowed = start + len;
            }
        }
        stats.get_mut(name).unwrap().greedy = count;
    }

    stats
}

fn without_gate(gates: &[Gate], skip: usize) -> Sig {
    let mut out = Vec::with_capacity(gates.len().saturating_sub(1));
    for (idx, gate) in gates.iter().enumerate() {
        if idx != skip {
            out.push(*gate);
        }
    }
    normalize(&out)
}

fn count_near_matches(
    gates: &[Gate],
    sets: &HashMap<&'static str, HashSet<Sig>>,
) -> HashMap<&'static str, NearStats> {
    let mut stats: HashMap<&'static str, NearStats> =
        sets.keys().copied().map(|name| (name, NearStats::default())).collect();

    for (&name, set) in sets {
        let mut exact_by_len: HashMap<usize, HashSet<Sig>> = HashMap::new();
        let mut delete_by_len: HashMap<usize, HashSet<Sig>> = HashMap::new();
        let mut sub_by_pos: HashMap<(usize, usize), HashSet<Sig>> = HashMap::new();
        let mut scan_lengths = HashSet::new();

        for sig in set {
            let len = sig.len();
            exact_by_len.entry(len).or_default().insert(sig.clone());
            scan_lengths.insert(len);
            if len > 1 {
                scan_lengths.insert(len - 1);
            }
            scan_lengths.insert(len + 1);
            for pos in 0..len {
                let shortened = without_gate(sig, pos);
                delete_by_len
                    .entry(len - 1)
                    .or_default()
                    .insert(shortened.clone());
                sub_by_pos.entry((len, pos)).or_default().insert(shortened);
            }
        }

        let mut scan_lengths: Vec<usize> = scan_lengths.into_iter().collect();
        scan_lengths.sort_unstable();

        for &len in &scan_lengths {
            if len == 0 || len > gates.len() {
                continue;
            }
            for i in 0..=gates.len() - len {
                let window = &gates[i..i + len];
                let sig = normalize(window);
                let exact = exact_by_len
                    .get(&len)
                    .is_some_and(|set| set.contains(&sig));
                if exact {
                    continue;
                }

                let missing = delete_by_len
                    .get(&len)
                    .is_some_and(|set| set.contains(&sig));

                let mut extra = false;
                if len > 1 {
                    if let Some(exact_shorter) = exact_by_len.get(&(len - 1)) {
                        for pos in 0..len {
                            let shortened = without_gate(window, pos);
                            if exact_shorter.contains(&shortened) {
                                extra = true;
                                break;
                            }
                        }
                    }
                }

                let mut substituted = false;
                for pos in 0..len {
                    let shortened = without_gate(window, pos);
                    if sub_by_pos
                        .get(&(len, pos))
                        .is_some_and(|set| set.contains(&shortened))
                    {
                        substituted = true;
                        break;
                    }
                }

                if missing || extra || substituted {
                    let stat = stats.get_mut(name).unwrap();
                    if missing {
                        stat.missing_one_gate += 1;
                    }
                    if extra {
                        stat.extra_one_gate += 1;
                    }
                    if substituted {
                        stat.substituted_one_gate += 1;
                    }
                    stat.any += 1;
                    if stat.examples.len() < 8 {
                        let mut cats = Vec::new();
                        if missing {
                            cats.push("missing");
                        }
                        if extra {
                            cats.push("extra");
                        }
                        if substituted {
                            cats.push("sub");
                        }
                        stat.examples.push(format!("{i}:{len}:{}", cats.join("+")));
                    }
                }
            }
        }
    }

    stats
}

fn insert_template(sets: &mut HashMap<&'static str, HashSet<Sig>>, name: &'static str, gates: &[Gate]) {
    sets.entry(name).or_default().insert(normalize(gates));
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: count_templates <circuit> [samf_samples=200000] [all|exact|near:<family>]");
        std::process::exit(2);
    }
    let samples: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let mode = args.get(3).map(String::as_str).unwrap_or("all");

    let gates = parse(&args[1]);
    let wires = gates
        .iter()
        .flat_map(|gate| gate.iter())
        .copied()
        .max()
        .map(|w| w as usize + 1)
        .unwrap_or(0);
    println!("circuit {} gates {} wires {}", args[1], gates.len(), wires);

    let mut sets: HashMap<&'static str, HashSet<Sig>> = HashMap::new();
    insert_template(&mut sets, "SG", &SG);
    insert_template(&mut sets, "RG1", &RG1);
    insert_template(&mut sets, "RG2", &RG2);
    insert_template(&mut sets, "RG3_like", &RG3);

    for neg in 0..=3u16 {
        for _ in 0..samples {
            let swap3 = Transpositions::gen_gates_swap(3, (1, 2, neg));
            let swap4 = Transpositions::gen_gates_swap(4, (1, 2, neg));
            insert_template(&mut sets, "SAMF_swap", &swap3);
            insert_template(&mut sets, "SAMF_swap", &swap4);
        }
    }
    for _ in 0..samples {
        let not4 = Transpositions::gen_gates_not(4, 1);
        insert_template(&mut sets, "SAMF_not", &not4);
    }

    if let Some(family) = mode.strip_prefix("near:") {
        sets.retain(|name, _| *name == family);
        if sets.is_empty() {
            eprintln!("unknown family {family:?}");
            std::process::exit(2);
        }
    }

    println!("template_sets");
    let mut names: Vec<_> = sets.keys().copied().collect();
    names.sort_unstable();
    for name in &names {
        let mut by_len: HashMap<usize, usize> = HashMap::new();
        for sig in &sets[name] {
            *by_len.entry(sig.len()).or_insert(0) += 1;
        }
        let mut parts: Vec<_> = by_len.into_iter().collect();
        parts.sort_unstable();
        let lens = parts
            .into_iter()
            .map(|(len, count)| format!("{len}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        println!("  {name} templates {} by_len {lens}", sets[name].len());
    }

    let stats = count_matches(&gates, &sets);
    let near = if mode == "exact" {
        HashMap::new()
    } else {
        count_near_matches(&gates, &sets)
    };
    println!("matches");
    for name in names {
        let stat = &stats[name];
        let mut by_len: Vec<_> = stat.by_len.iter().map(|(&k, &v)| (k, v)).collect();
        by_len.sort_unstable();
        let lens = by_len
            .into_iter()
            .map(|(len, count)| format!("{len}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        let examples = stat
            .examples
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "  {name} raw {} greedy_nonoverlap {} by_len {} examples {}",
            stat.raw, stat.greedy, lens, examples
        );
        if mode != "exact" {
            let near_stat = &near[name];
            println!(
                "  {name} near_edit1_any {} missing_one_gate {} extra_one_gate {} substituted_one_gate {} examples {}",
                near_stat.any,
                near_stat.missing_one_gate,
                near_stat.extra_one_gate,
                near_stat.substituted_one_gate,
                near_stat.examples.join(",")
            );
        }
    }
}
