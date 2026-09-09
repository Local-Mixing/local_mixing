//! Enumerate diverse residual-CNF models at a weight upper bound, then
//! crossover + polish. Gray-box (chunk tables → CNF).

use clap::Parser;
use local_mixing::sandwich::{hamming_weight, ChunkBody, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight128_obf.txt")]
    challenge: PathBuf,
    #[arg(long, default_value = "challenges/work_enum")]
    work_dir: PathBuf,
    #[arg(long, default_value = "third_party/kissat/build/kissat")]
    kissat: PathBuf,
    #[arg(long, default_value_t = 35)]
    weight_ub: usize,
    #[arg(long, default_value_t = 40)]
    n_models: usize,
    #[arg(long, default_value_t = 90)]
    kissat_seconds: u64,
    #[arg(long, default_value_t = 500_000)]
    polish_steps: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Abs {
    Zero,
    One,
    Unk,
}

struct Challenge {
    prefix_zeros: usize,
    circuit: ChunkedCircuit,
}

#[derive(Clone)]
struct ResidualPerm {
    in_wires: Vec<u16>,
    out_wires: Vec<u16>,
    data: Vec<usize>,
}

fn load_challenge(path: &Path) -> Challenge {
    let s = fs::read_to_string(path).unwrap();
    let mut prefix_zeros = 64usize;
    let mut body = String::new();
    for line in s.lines() {
        let t = line.trim();
        if let Some(r) = t.strip_prefix("challenge_prefix_zeros ") {
            prefix_zeros = r.parse().unwrap();
            continue;
        }
        if t.starts_with("challenge_seed ") {
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    Challenge {
        prefix_zeros,
        circuit: ChunkedCircuit::from_readable_string(&body).unwrap(),
    }
}

fn residual_circuit(ch: &Challenge) -> (Vec<ResidualPerm>, Vec<Abs>) {
    let c = &ch.circuit;
    let mut st = vec![Abs::Unk; c.total_wires];
    for w in 0..ch.prefix_zeros {
        st[w] = Abs::Zero;
    }
    for w in c.data_wires..c.total_wires {
        st[w] = Abs::Zero;
    }
    let mut residuals = Vec::new();
    for chunk in &c.chunks {
        match &chunk.body {
            ChunkBody::Zero => {
                for &w in &chunk.wires {
                    st[w as usize] = Abs::Zero;
                }
            }
            ChunkBody::Perm(perm) => {
                let k = chunk.wires.len();
                let mut in_idx = Vec::new();
                let mut in_wires = Vec::new();
                for (i, &w) in chunk.wires.iter().enumerate() {
                    if st[w as usize] == Abs::Unk {
                        in_idx.push(i);
                        in_wires.push(w);
                    }
                }
                let mut pairs = Vec::new();
                if in_wires.is_empty() {
                    let mut pat = 0usize;
                    for (i, &w) in chunk.wires.iter().enumerate() {
                        if st[w as usize] == Abs::One {
                            pat |= 1 << i;
                        }
                    }
                    pairs.push((pat, perm.data[pat]));
                } else {
                    let uk = in_wires.len();
                    for u in 0..(1 << uk) {
                        let mut pat = 0usize;
                        for (i, &w) in chunk.wires.iter().enumerate() {
                            if st[w as usize] == Abs::One {
                                pat |= 1 << i;
                            }
                        }
                        for (j, &i) in in_idx.iter().enumerate() {
                            if ((u >> j) & 1) != 0 {
                                pat |= 1 << i;
                            }
                        }
                        pairs.push((pat, perm.data[pat]));
                    }
                }
                let mut and_out = (1usize << k) - 1;
                let mut or_out = 0usize;
                for &(_, outv) in &pairs {
                    and_out &= outv;
                    or_out |= outv;
                }
                let mut out_idx = Vec::new();
                let mut out_wires = Vec::new();
                for (i, &w) in chunk.wires.iter().enumerate() {
                    let a = ((and_out >> i) & 1) != 0;
                    let o = ((or_out >> i) & 1) != 0;
                    if a == o {
                        st[w as usize] = if a { Abs::One } else { Abs::Zero };
                    } else {
                        st[w as usize] = Abs::Unk;
                        out_idx.push(i);
                        out_wires.push(w);
                    }
                }
                if out_wires.is_empty() {
                    continue;
                }
                let uk = in_wires.len();
                let mut data = vec![0usize; 1 << uk];
                for u in 0..(1 << uk) {
                    let fout = pairs
                        .iter()
                        .find(|(fin, _)| {
                            in_idx.iter().enumerate().all(|(j, &i)| {
                                (((fin >> i) & 1) != 0) == (((u >> j) & 1) != 0)
                            })
                        })
                        .map(|(_, o)| *o)
                        .unwrap();
                    let mut out_u = 0usize;
                    for (j, &i) in out_idx.iter().enumerate() {
                        if ((fout >> i) & 1) != 0 {
                            out_u |= 1 << j;
                        }
                    }
                    data[u] = out_u;
                }
                residuals.push(ResidualPerm {
                    in_wires,
                    out_wires,
                    data,
                });
            }
        }
    }
    (residuals, st)
}

struct CnfCore {
    free_vars: Vec<i32>,
    out_vars: Vec<Option<i32>>,
    clauses: Vec<Vec<i32>>,
    nvars: i32,
}

fn build_core(ch: &Challenge, residuals: &[ResidualPerm], final_abs: &[Abs]) -> CnfCore {
    let free_n = ch.circuit.data_wires - ch.prefix_zeros;
    let mut next = 1i32;
    let mut free_vars = Vec::with_capacity(free_n);
    for _ in 0..free_n {
        free_vars.push(next);
        next += 1;
    }
    let mut cur: Vec<Option<i32>> = vec![None; ch.circuit.total_wires];
    for (i, &v) in free_vars.iter().enumerate() {
        cur[ch.prefix_zeros + i] = Some(v);
    }
    let mut clauses = Vec::new();
    for (ri, rp) in residuals.iter().enumerate() {
        let k_in = rp.in_wires.len();
        let mut outs = Vec::with_capacity(rp.out_wires.len());
        for _ in &rp.out_wires {
            outs.push(next);
            next += 1;
        }
        for pat in 0..(1 << k_in) {
            let mut ante = Vec::with_capacity(k_in);
            for (i, &w) in rp.in_wires.iter().enumerate() {
                let src = cur[w as usize].unwrap_or_else(|| panic!("undef {w} @ {ri}"));
                if ((pat >> i) & 1) != 0 {
                    ante.push(-src);
                } else {
                    ante.push(src);
                }
            }
            let outv = rp.data[pat];
            for (i, &ov) in outs.iter().enumerate() {
                let mut c = ante.clone();
                if ((outv >> i) & 1) != 0 {
                    c.push(ov);
                } else {
                    c.push(-ov);
                }
                clauses.push(c);
            }
        }
        for (i, &w) in rp.out_wires.iter().enumerate() {
            cur[w as usize] = Some(outs[i]);
        }
    }
    let mut out_vars = Vec::with_capacity(ch.circuit.data_wires);
    for w in 0..ch.circuit.data_wires {
        match final_abs[w] {
            Abs::Zero => out_vars.push(None),
            Abs::One => panic!("forced one"),
            Abs::Unk => out_vars.push(Some(cur[w].unwrap())),
        }
    }
    CnfCore {
        free_vars,
        out_vars,
        clauses,
        nvars: next - 1,
    }
}

fn cardinality_at_most(core: &CnfCore, ub: usize) -> (i32, Vec<Vec<i32>>) {
    let out_lits: Vec<i32> = core.out_vars.iter().filter_map(|x| *x).collect();
    let m = out_lits.len();
    let mut next = core.nvars + 1;
    let mut clauses = Vec::new();
    if ub >= m {
        return (core.nvars, clauses);
    }
    let kmax = ub;
    let mut s = vec![vec![0i32; kmax + 2]; m + 1];
    for i in 1..=m {
        for j in 1..=(kmax + 1) {
            s[i][j] = next;
            next += 1;
        }
    }
    for i in 1..=m {
        let x = out_lits[i - 1];
        if i == 1 {
            clauses.push(vec![-s[1][1], x]);
            clauses.push(vec![s[1][1], -x]);
        } else {
            clauses.push(vec![-s[i][1], s[i - 1][1], x]);
            clauses.push(vec![-s[i - 1][1], s[i][1]]);
            clauses.push(vec![-x, s[i][1]]);
        }
        for j in 2..=(kmax + 1) {
            if j > i {
                clauses.push(vec![-s[i][j]]);
                continue;
            }
            clauses.push(vec![-s[i][j], s[i - 1][j], s[i - 1][j - 1]]);
            clauses.push(vec![-s[i][j], s[i - 1][j], x]);
            clauses.push(vec![-s[i - 1][j], s[i][j]]);
            clauses.push(vec![-s[i - 1][j - 1], -x, s[i][j]]);
        }
    }
    clauses.push(vec![-s[m][kmax + 1]]);
    (next - 1, clauses)
}

fn write_cnf(path: &Path, nvars: i32, clauses: &[Vec<i32>]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "p cnf {nvars} {}", clauses.len())?;
    for cl in clauses {
        for lit in cl {
            write!(f, "{lit} ")?;
        }
        writeln!(f, "0")?;
    }
    f.flush()
}

fn run_kissat(bin: &Path, cnf: &Path, seconds: u64, seed: u64) -> String {
    let out = Command::new(bin)
        .arg(format!("--time={seconds}"))
        .arg(format!("--seed={seed}"))
        .arg("--quiet=1")
        .arg(cnf)
        .output()
        .expect("kissat");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

fn parse_free(out: &str, free_vars: &[i32]) -> Option<Vec<bool>> {
    if !out.lines().any(|l| l.starts_with("s SATISFIABLE")) {
        return None;
    }
    let mut free = vec![false; free_vars.len()];
    for line in out.lines() {
        let line = line.trim();
        if !line.starts_with('v') {
            continue;
        }
        for tok in line[1..].split_whitespace() {
            if tok == "0" {
                break;
            }
            let lit: i32 = tok.parse().ok()?;
            let v = lit.unsigned_abs() as i32;
            if let Some(i) = free_vars.iter().position(|&x| x == v) {
                free[i] = lit > 0;
            }
        }
    }
    Some(free)
}

fn bits_to_input(ch: &Challenge, free: &[bool]) -> Vec<bool> {
    let mut input = vec![false; ch.circuit.data_wires];
    input[ch.prefix_zeros..].copy_from_slice(free);
    input
}

fn free_key(free: &[bool]) -> u128 {
    let mut x = 0u128;
    for (i, &b) in free.iter().enumerate() {
        if b {
            x |= 1u128 << i;
        }
    }
    x
}

fn free_to_hex(free: &[bool]) -> u64 {
    let mut x = 0u64;
    for i in 0..free.len().min(64) {
        if free[i] {
            x |= 1 << i;
        }
    }
    x
}

fn polish(ch: &Challenge, start: &[bool], steps: usize, seed: u64) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = start.len();
    let mut best = start.to_vec();
    let mut best_wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &best)));
    let mut cur = best.clone();
    let mut cur_wt = best_wt;
    let mut temp = 2.5f64;
    for s in 0..steps {
        let nflip = 1 + (rng.random_bool(0.3) as usize) + (rng.random_bool(0.1) as usize);
        let mut flipped = Vec::new();
        for _ in 0..nflip {
            let i = rng.random_range(0..n);
            cur[i] = !cur[i];
            flipped.push(i);
        }
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &cur)));
        if wt <= cur_wt || rng.random::<f64>() < ((cur_wt as f64 - wt as f64) / temp).exp() {
            cur_wt = wt;
            if wt < best_wt {
                best_wt = wt;
                best = cur.clone();
                println!("  [polish] wt={best_wt}");
            }
        } else {
            for i in flipped {
                cur[i] = !cur[i];
            }
        }
        if s % 10000 == 0 && s > 0 {
            temp *= 0.92;
            if temp < 0.05 {
                temp = 1.0;
                cur = best.clone();
                cur_wt = best_wt;
            }
        }
    }
    (best_wt, best)
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.work_dir).unwrap();
    let ch = load_challenge(&args.challenge);
    let (residuals, final_abs) = residual_circuit(&ch);
    let core = build_core(&ch, &residuals, &final_abs);
    let (nvars, card) = cardinality_at_most(&core, args.weight_ub);
    println!(
        "core clauses={} +card={} nvars={} ub={}",
        core.clauses.len(),
        card.len(),
        nvars,
        args.weight_ub
    );

    let mut base_clauses = core.clauses.clone();
    base_clauses.extend(card);

    let mut models: Vec<(usize, Vec<bool>)> = Vec::new();
    let mut seen: HashSet<u128> = HashSet::new();
    let mut blocks: Vec<Vec<i32>> = Vec::new();
    let mut rng = StdRng::seed_from_u64(11);

    for i in 0..args.n_models {
        let mut clauses = base_clauses.clone();
        clauses.extend(blocks.iter().cloned());
        // Random phase hint via a few free-bit unit preferences? Use blocking + seed.
        // Add a random XOR-ish cut: force a random free bit to a random value sometimes
        // to diversify (retractable by not adding to blocks permanently if fails).
        let mut probe = clauses.clone();
        if i > 0 && rng.random_bool(0.7) {
            let b = rng.random_range(0..core.free_vars.len());
            let v = core.free_vars[b];
            if rng.random_bool(0.5) {
                probe.push(vec![v]);
            } else {
                probe.push(vec![-v]);
            }
        }
        let cnf = args.work_dir.join(format!("m{i}.cnf"));
        write_cnf(&cnf, nvars, &probe).unwrap();
        let t0 = Instant::now();
        let out = run_kissat(&args.kissat, &cnf, args.kissat_seconds, 1000 + i as u64);
        let dt = t0.elapsed().as_secs_f64();
        if let Some(free) = parse_free(&out, &core.free_vars) {
            let key = free_key(&free);
            let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &free)));
            if seen.insert(key) {
                println!(
                    "model {i}: wt={wt} x=0x{:016x} ({dt:.1}s) unique={}",
                    free_to_hex(&free),
                    seen.len()
                );
                models.push((wt, free.clone()));
                // Block this free assignment
                let mut block = Vec::new();
                for (j, &v) in core.free_vars.iter().enumerate() {
                    if free[j] {
                        block.push(-v);
                    } else {
                        block.push(v);
                    }
                }
                blocks.push(block);
            } else {
                println!("model {i}: duplicate ({dt:.1}s)");
            }
        } else if out.contains("s UNSATISFIABLE") {
            println!("model {i}: UNSAT with blocks — stop ({dt:.1}s)");
            break;
        } else {
            println!("model {i}: timeout ({dt:.1}s)");
            // drop probe bias and try pure once
        }
    }

    if models.is_empty() {
        println!("no models");
        return;
    }
    models.sort_by_key(|(w, _)| *w);
    let mut best_wt = models[0].0;
    let mut best = models[0].1.clone();
    println!("pool={} best_so_far={best_wt}", models.len());

    // Backbone of free bits among models with wt == best
    let elites: Vec<_> = models.iter().filter(|(w, _)| *w == best_wt).collect();
    let n = best.len();
    let mut agree = vec![0i32; n];
    for (_, f) in &elites {
        for i in 0..n {
            if f[i] {
                agree[i] += 1;
            } else {
                agree[i] -= 1;
            }
        }
    }
    let backbone: Vec<Option<bool>> = agree
        .iter()
        .map(|&a| {
            if a == elites.len() as i32 {
                Some(true)
            } else if a == -(elites.len() as i32) {
                Some(false)
            } else {
                None
            }
        })
        .collect();
    let n_bb = backbone.iter().filter(|b| b.is_some()).count();
    println!("elite backbone bits={n_bb}/{} among {} elites", n, elites.len());

    // Crossover: uniform between random elite pairs, then polish
    for t in 0..200 {
        let a = &elites[rng.random_range(0..elites.len())].1;
        let b = &elites[rng.random_range(0..elites.len())].1;
        let mut child = vec![false; n];
        for i in 0..n {
            child[i] = if rng.random_bool(0.5) { a[i] } else { b[i] };
        }
        // Respect backbone
        for i in 0..n {
            if let Some(v) = backbone[i] {
                child[i] = v;
            }
        }
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &child)));
        if wt < best_wt {
            best_wt = wt;
            best = child;
            println!("crossover best wt={best_wt}");
        }
    }

    println!("polish...");
    let (pwt, pfree) = polish(&ch, &best, args.polish_steps, 5);
    if pwt < best_wt {
        best_wt = pwt;
        best = pfree;
    }
    // Also polish each unique model briefly
    for (i, (w, f)) in models.iter().take(15).enumerate() {
        let (pw, pf) = polish(&ch, f, args.polish_steps / 10, 20 + i as u64);
        if pw < best_wt {
            best_wt = pw;
            best = pf;
            println!("from model {i} (was {w}) -> {best_wt}");
        }
    }

    println!("DONE best_wt={best_wt} x=0x{:016x}", free_to_hex(&best));
}
