//! Gray-box: iterative output-bit killing on the residual CNF.
//! Start from a known low-weight model (or random), then repeatedly try to force
//! each currently-1 output data bit to 0 and re-solve. Uses chunk truth tables.

use clap::Parser;
use local_mixing::sandwich::{hamming_weight, ChunkBody, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight128_obf.txt")]
    challenge: PathBuf,
    #[arg(long, default_value = "challenges/work_kill")]
    work_dir: PathBuf,
    #[arg(long, default_value = "third_party/kissat/build/kissat")]
    kissat: PathBuf,
    #[arg(long, default_value_t = 90)]
    kissat_seconds: u64,
    /// Seed free bits as hex (lo 64). Empty = use SAT model file or anneal seed.
    #[arg(long, default_value = "a8237049ae982e16")]
    seed_x_lo: String,
    #[arg(long, default_value = "0")]
    seed_x_hi: String,
    #[arg(long, default_value_t = 40)]
    rounds: usize,
    #[arg(long, default_value_t = 200_000)]
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

                let mut pairs: Vec<(usize, usize)> = Vec::new();
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
                    let mut found = None;
                    for &(fin, fout) in &pairs {
                        let mut ok = true;
                        for (j, &i) in in_idx.iter().enumerate() {
                            let bit = ((fin >> i) & 1) != 0;
                            let want = ((u >> j) & 1) != 0;
                            if bit != want {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            found = Some(fout);
                            break;
                        }
                    }
                    let fout = found.expect("pattern");
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

/// Build residual CNF once. Returns free vars, output data vars (None if const 0),
/// and the clause list (without cardinality / kill units).
struct CnfCore {
    free_vars: Vec<i32>,
    /// For each data wire: Some(var) if unknown, None if const 0, (forced ones panic).
    out_vars: Vec<Option<i32>>,
    /// All hard residual clauses.
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
                let src = cur[w as usize].unwrap_or_else(|| {
                    panic!("undefined wire {w} at residual {ri}")
                });
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
            Abs::One => panic!("forced one at out {w} — unexpected"),
            Abs::Unk => out_vars.push(Some(cur[w].expect("unk"))),
        }
    }

    CnfCore {
        free_vars,
        out_vars,
        clauses,
        nvars: next - 1,
    }
}

/// Sequential counter: at most `ub` ones among the unknown output vars.
/// Returns (new_nvars, extra_clauses). `ub` counts only unknown outs (no forced ones).
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

fn write_cnf(
    path: &Path,
    core: &CnfCore,
    extra: &[Vec<i32>],
    nvars: i32,
) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let nclauses = core.clauses.len() + extra.len();
    writeln!(f, "c kill-bits gray residual")?;
    writeln!(f, "p cnf {nvars} {nclauses}")?;
    for cl in core.clauses.iter().chain(extra.iter()) {
        for lit in cl {
            write!(f, "{lit} ")?;
        }
        writeln!(f, "0")?;
    }
    f.flush()
}

fn run_kissat(bin: &Path, cnf: &Path, seconds: u64) -> String {
    let out = Command::new(bin)
        .arg(format!("--time={seconds}"))
        .arg("--quiet=1")
        .arg(cnf)
        .output()
        .expect("kissat");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

fn parse_model(out: &str, nvars: i32) -> Option<Vec<bool>> {
    if !out.lines().any(|l| l.starts_with("s SATISFIABLE")) {
        return None;
    }
    let mut assign = vec![false; (nvars as usize) + 1];
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
            let v = lit.unsigned_abs() as usize;
            if v < assign.len() {
                assign[v] = lit > 0;
            }
        }
    }
    Some(assign)
}

fn free_from_assign(core: &CnfCore, assign: &[bool]) -> Vec<bool> {
    core.free_vars
        .iter()
        .map(|&v| assign[v as usize])
        .collect()
}

fn bits_to_input(ch: &Challenge, free: &[bool]) -> Vec<bool> {
    let mut input = vec![false; ch.circuit.data_wires];
    input[ch.prefix_zeros..].copy_from_slice(free);
    input
}

fn parse_hex_u64(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
}

fn free_from_hex(lo: u64, hi: u64, n: usize) -> Vec<bool> {
    let mut free = vec![false; n];
    for i in 0..n.min(64) {
        free[i] = ((lo >> i) & 1) != 0;
    }
    for i in 64..n {
        free[i] = ((hi >> (i - 64)) & 1) != 0;
    }
    free
}

fn free_to_hex(free: &[bool]) -> (u64, u64) {
    let mut x0 = 0u64;
    let mut x1 = 0u64;
    for i in 0..free.len().min(64) {
        if free[i] {
            x0 |= 1 << i;
        }
    }
    for i in 64..free.len() {
        if free[i] {
            x1 |= 1 << (i - 64);
        }
    }
    (x0, x1)
}

fn polish(ch: &Challenge, start: &[bool], steps: usize, seed: u64) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let free_n = start.len();
    let mut best = start.to_vec();
    let mut best_wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &best)));
    let mut cur = best.clone();
    let mut cur_wt = best_wt;
    let mut temp = 3.0f64;
    for s in 0..steps {
        let nflip = if rng.random_bool(0.2) {
            3
        } else if rng.random_bool(0.35) {
            2
        } else {
            1
        };
        let mut flipped = Vec::with_capacity(nflip);
        for _ in 0..nflip {
            let i = rng.random_range(0..free_n);
            cur[i] = !cur[i];
            flipped.push(i);
        }
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &cur)));
        let accept = wt <= cur_wt
            || rng.random::<f64>() < ((cur_wt as f64 - wt as f64) / temp).exp();
        if accept {
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
        if s % 8000 == 0 && s > 0 {
            temp *= 0.9;
            if temp < 0.04 {
                temp = 1.2;
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
    println!(
        "data={} chunks={} prefix={}",
        ch.circuit.data_wires,
        ch.circuit.chunks.len(),
        ch.prefix_zeros
    );

    let t0 = Instant::now();
    let (residuals, final_abs) = residual_circuit(&ch);
    let core = build_core(&ch, &residuals, &final_abs);
    println!(
        "core: residuals={} clauses={} nvars={} in {:.2}s",
        residuals.len(),
        core.clauses.len(),
        core.nvars,
        t0.elapsed().as_secs_f64()
    );

    let free_n = ch.circuit.data_wires - ch.prefix_zeros;
    let mut best_free = free_from_hex(
        parse_hex_u64(&args.seed_x_lo),
        parse_hex_u64(&args.seed_x_hi),
        free_n,
    );
    let mut best_wt =
        hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &best_free)));
    println!("seed wt={best_wt}");

    // First: get a full SAT assignment consistent with seed free bits (unit-assume free).
    let mut assume_free: Vec<Vec<i32>> = Vec::new();
    for (i, &v) in core.free_vars.iter().enumerate() {
        if best_free[i] {
            assume_free.push(vec![v]);
        } else {
            assume_free.push(vec![-v]);
        }
    }
    let cnf0 = args.work_dir.join("seed.cnf");
    write_cnf(&cnf0, &core, &assume_free, core.nvars).unwrap();
    let out0 = run_kissat(&args.kissat, &cnf0, args.kissat_seconds.min(60));
    let mut assign = parse_model(&out0, core.nvars).unwrap_or_else(|| {
        panic!(
            "seed model UNSAT — encoding bug? {}",
            out0.lines().take(8).collect::<Vec<_>>().join(" | ")
        );
    });
    for (i, &v) in core.free_vars.iter().enumerate() {
        assign[v as usize] = best_free[i];
    }

    let mut rng = StdRng::seed_from_u64(123);
    for round in 0..args.rounds {
        let y = ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &best_free));
        let ones: Vec<usize> = y
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        println!("\n=== round {round} wt={} ===", ones.len());
        if ones.is_empty() {
            break;
        }

        let mut targets = ones.clone();
        for i in (1..targets.len()).rev() {
            let j = rng.random_range(0..=i);
            targets.swap(i, j);
        }

        let target_ub = best_wt - 1;
        let (nvars_card, mut card_clauses) = cardinality_at_most(&core, target_ub);

        let mut improved = false;
        for &bit in &targets {
            let ov = match core.out_vars[bit] {
                Some(v) => v,
                None => continue,
            };
            // Monotone: keep existing zeros, kill `bit`, and weight <= best-1.
            let mut extra = card_clauses.clone();
            extra.push(vec![-ov]);
            for (i, &b) in y.iter().enumerate() {
                if !b {
                    if let Some(v) = core.out_vars[i] {
                        extra.push(vec![-v]);
                    }
                }
            }

            let cnf = args.work_dir.join(format!("r{round}_b{bit}.cnf"));
            write_cnf(&cnf, &core, &extra, nvars_card).unwrap();
            let t1 = Instant::now();
            let out = run_kissat(&args.kissat, &cnf, args.kissat_seconds);
            let dt = t1.elapsed().as_secs_f64();
            if let Some(new_assign) = parse_model(&out, nvars_card) {
                let free = free_from_assign(&core, &new_assign);
                let wt =
                    hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &free)));
                println!("  kill out[{bit}] SAT wt={wt} ({dt:.1}s)");
                if wt < best_wt {
                    best_wt = wt;
                    best_free = free;
                    assign = new_assign;
                    improved = true;
                    let (x0, x1) = free_to_hex(&best_free);
                    println!("  NEW BEST wt={best_wt} x_lo=0x{x0:016x} x_hi=0x{x1:016x}");
                    break;
                }
            } else if out.contains("s UNSATISFIABLE") {
                println!("  kill out[{bit}] UNSAT ({dt:.1}s)");
            } else {
                println!("  kill out[{bit}] timeout ({dt:.1}s)");
            }
        }

        if !improved {
            println!("  monotone stuck; try kill+card without freezing zeros");
            let mut any = false;
            for &bit in targets.iter().take(24) {
                let ov = match core.out_vars[bit] {
                    Some(v) => v,
                    None => continue,
                };
                let mut extra = card_clauses.clone();
                extra.push(vec![-ov]);
                let cnf = args.work_dir.join(format!("r{round}_loose_b{bit}.cnf"));
                write_cnf(&cnf, &core, &extra, nvars_card).unwrap();
                let t1 = Instant::now();
                let out = run_kissat(&args.kissat, &cnf, args.kissat_seconds);
                let dt = t1.elapsed().as_secs_f64();
                if let Some(new_assign) = parse_model(&out, nvars_card) {
                    let free = free_from_assign(&core, &new_assign);
                    let wt = hamming_weight(
                        &ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &free)),
                    );
                    println!("  loose kill out[{bit}] SAT wt={wt} ({dt:.1}s)");
                    if wt < best_wt {
                        best_wt = wt;
                        best_free = free;
                        assign = new_assign;
                        any = true;
                        let (x0, x1) = free_to_hex(&best_free);
                        println!("  NEW BEST wt={best_wt} x_lo=0x{x0:016x} x_hi=0x{x1:016x}");
                        break;
                    }
                } else if out.contains("s UNSATISFIABLE") {
                    println!("  loose kill out[{bit}] UNSAT ({dt:.1}s)");
                } else {
                    println!("  loose kill out[{bit}] timeout ({dt:.1}s)");
                }
            }
            let _ = &mut card_clauses;
            if !any {
                println!("no improvement this round; stop kills");
                break;
            }
        }
    }

    println!("\npolish from wt={best_wt}...");
    let (pwt, pfree) = polish(&ch, &best_free, args.polish_steps, 7);
    if pwt < best_wt {
        best_wt = pwt;
        best_free = pfree;
    }
    let (x0, x1) = free_to_hex(&best_free);
    println!("DONE best_wt={best_wt} x_lo=0x{x0:016x} x_hi=0x{x1:016x}");
    let _ = assign;
}
