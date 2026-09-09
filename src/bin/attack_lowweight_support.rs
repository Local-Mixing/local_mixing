//! Gray-box: random output-support forcing (no sequential counter).
//! Force `n - ub` random output bits to 0 ⇒ any model has weight ≤ ub.
//! Often easier for CDCL than cardinality networks.

use clap::Parser;
use local_mixing::sandwich::{hamming_weight, ChunkBody, ChunkedCircuit};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng, SeedableRng};
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
    #[arg(long, default_value = "challenges/work_support")]
    work_dir: PathBuf,
    #[arg(long, default_value = "third_party/kissat/build/kissat")]
    kissat: PathBuf,
    #[arg(long, default_value_t = 34)]
    weight_ub: usize,
    #[arg(long, default_value_t = 80)]
    trials: usize,
    #[arg(long, default_value_t = 45)]
    kissat_seconds: u64,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 300_000)]
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
    out_vars: Vec<i32>, // only unknown outs
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
                let src = cur[w as usize].unwrap_or_else(|| panic!("undef {w}@{ri}"));
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
    let mut out_vars = Vec::new();
    for w in 0..ch.circuit.data_wires {
        match final_abs[w] {
            Abs::Zero => {}
            Abs::One => panic!("forced one"),
            Abs::Unk => out_vars.push(cur[w].unwrap()),
        }
    }
    CnfCore {
        free_vars,
        out_vars,
        clauses,
        nvars: next - 1,
    }
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
    let mut temp = 2.0f64;
    for s in 0..steps {
        let i = rng.random_range(0..n);
        cur[i] = !cur[i];
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &cur)));
        if wt <= cur_wt || rng.random::<f64>() < ((cur_wt as f64 - wt as f64) / temp).exp() {
            cur_wt = wt;
            if wt < best_wt {
                best_wt = wt;
                best = cur.clone();
                println!("  [polish] wt={best_wt}");
            }
        } else {
            cur[i] = !cur[i];
        }
        if s % 8000 == 0 && s > 0 {
            temp = (temp * 0.9).max(0.05);
            if s % 40000 == 0 {
                cur = best.clone();
                cur_wt = best_wt;
                temp = 1.5;
            }
        }
    }
    (best_wt, best)
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.work_dir).unwrap();
    let ch = load_challenge(&args.challenge);
    let t0 = Instant::now();
    let (residuals, final_abs) = residual_circuit(&ch);
    let core = build_core(&ch, &residuals, &final_abs);
    println!(
        "core clauses={} outs={} nvars={} in {:.2}s",
        core.clauses.len(),
        core.out_vars.len(),
        core.nvars,
        t0.elapsed().as_secs_f64()
    );

    let n_out = core.out_vars.len();
    let n_force = n_out.saturating_sub(args.weight_ub);
    println!(
        "force {n_force}/{n_out} outs to 0 (ub={})",
        args.weight_ub
    );

    // Biased supports from known good model zeros
    let seed_x: u64 = 0xa8237049ae982e16;
    let mut seed_free = vec![false; core.free_vars.len()];
    for i in 0..seed_free.len().min(64) {
        seed_free[i] = ((seed_x >> i) & 1) != 0;
    }
    let seed_y = ch
        .circuit
        .evaluate_data_bits(&bits_to_input(&ch, &seed_free));
    let seed_zeros: Vec<usize> = seed_y
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if !b { Some(i) } else { None })
        .collect();
    println!(
        "seed wt={} zeros={}",
        seed_y.iter().filter(|&&b| b).count(),
        seed_zeros.len()
    );

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut best_wt = seed_y.iter().filter(|&&b| b).count();
    let mut best_free = seed_free.clone();
    let mut seen = HashSet::new();
    seen.insert(seed_x);

    for trial in 0..args.trials {
        // Build support: start from seed zeros, then mutate; or fully random
        let mut force_idx: Vec<usize> = if trial % 3 == 0 {
            // fully random
            let mut idx: Vec<usize> = (0..n_out).collect();
            idx.shuffle(&mut rng);
            idx.truncate(n_force);
            idx
        } else {
            // keep most seed zeros, swap a few
            let mut z = seed_zeros.clone();
            z.shuffle(&mut rng);
            let mut chosen: Vec<usize> = z.into_iter().take(n_force.min(seed_zeros.len())).collect();
            // if need more, add random ones-bits from seed
            while chosen.len() < n_force {
                let i = rng.random_range(0..n_out);
                if !chosen.contains(&i) {
                    chosen.push(i);
                }
            }
            // mutate: replace a few
            for _ in 0..(2 + trial % 5) {
                if chosen.is_empty() {
                    break;
                }
                let j = rng.random_range(0..chosen.len());
                let mut neu = rng.random_range(0..n_out);
                while chosen.contains(&neu) {
                    neu = rng.random_range(0..n_out);
                }
                chosen[j] = neu;
            }
            chosen.truncate(n_force);
            chosen
        };

        force_idx.sort_unstable();
        let mut clauses = core.clauses.clone();
        for &i in &force_idx {
            // Map data-wire index i → out var. out_vars is parallel to unknown data wires
            // in order 0..data_wires skipping const — here all are unk, so out_vars[i].
            clauses.push(vec![-core.out_vars[i]]);
        }

        let cnf = args.work_dir.join(format!("t{trial}.cnf"));
        write_cnf(&cnf, core.nvars, &clauses).unwrap();
        let t1 = Instant::now();
        let out = run_kissat(
            &args.kissat,
            &cnf,
            args.kissat_seconds,
            args.seed + trial as u64,
        );
        let dt = t1.elapsed().as_secs_f64();
        if let Some(free) = parse_free(&out, &core.free_vars) {
            let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &free)));
            let x = free_to_hex(&free);
            let neu = seen.insert(x);
            println!(
                "trial {trial}: SAT wt={wt} x=0x{x:016x} new={neu} ({dt:.1}s)"
            );
            if wt < best_wt {
                best_wt = wt;
                best_free = free;
                println!("  NEW BEST {best_wt}");
            }
        } else if out.contains("s UNSATISFIABLE") {
            println!("trial {trial}: UNSAT ({dt:.1}s)");
        } else {
            println!("trial {trial}: timeout ({dt:.1}s)");
        }
    }

    println!("polish from {best_wt}...");
    let (pwt, pfree) = polish(&ch, &best_free, args.polish_steps, 3);
    if pwt < best_wt {
        best_wt = pwt;
        best_free = pfree;
    }
    println!(
        "DONE best_wt={best_wt} x=0x{:016x}",
        free_to_hex(&best_free)
    );
}
