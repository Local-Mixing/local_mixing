//! Gray-box low-weight attack via residual CNF + kissat.
//!
//! Constant-propagate through chunk truth tables under the known zero prefix
//! and zero ancilla, then encode only the residual unknown fragment as CNF
//! with a sequential-counter cardinality constraint on output weight.

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
    #[arg(long, default_value = "challenges/work_cnf")]
    work_dir: PathBuf,
    #[arg(long, default_value = "third_party/kissat/build/kissat")]
    kissat: PathBuf,
    /// Starting weight upper bound for descent (0 = run a short black-box baseline).
    #[arg(long, default_value_t = 0)]
    start_ub: usize,
    #[arg(long, default_value_t = 2)]
    step: usize,
    #[arg(long, default_value_t = 120)]
    kissat_seconds: u64,
    #[arg(long, default_value_t = 2000)]
    baseline_samples: usize,
    #[arg(long, default_value_t = 300_000)]
    polish_steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Abs {
    Zero,
    One,
    Unk,
}

struct Challenge {
    prefix_zeros: usize,
    circuit: ChunkedCircuit,
}

#[derive(Clone, Debug)]
struct ResidualPerm {
    in_wires: Vec<u16>,
    out_wires: Vec<u16>,
    /// data[in_pat] = out_pat on out_wires
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
                assert!(!in_wires.is_empty(), "varying outputs need unknown inputs");

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

/// Emit CNF. Returns (nvars, free_var_ids starting at 1).
fn emit_cnf(
    ch: &Challenge,
    residuals: &[ResidualPerm],
    final_abs: &[Abs],
    path: &Path,
    ub: usize,
) -> std::io::Result<(usize, Vec<i32>)> {
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

    let mut clauses: Vec<Vec<i32>> = Vec::new();

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
                // matching pat: for bit=1 use ~src in ante of (~ante \/ out)
                // clause: \/_{bit=0} src  \/  \/_{bit=1} ~src  \/ out_lit
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

    // Collect output literals contributing to weight.
    let mut out_lits = Vec::new();
    let mut forced_ones = 0usize;
    for w in 0..ch.circuit.data_wires {
        match final_abs[w] {
            Abs::Zero => {}
            Abs::One => forced_ones += 1,
            Abs::Unk => {
                let v = cur[w].unwrap_or_else(|| panic!("unk out {w} has no var"));
                out_lits.push(v);
            }
        }
    }
    assert!(
        forced_ones <= ub,
        "forced ones {forced_ones} already exceed ub {ub}"
    );
    let rem_ub = ub - forced_ones;

    // Sequential counter: at most rem_ub ones among out_lits.
    let m = out_lits.len();
    if rem_ub < m {
        let kmax = rem_ub;
        // s[i][j] (1-based i over outs, j=1..=kmax+1): among first i, at least j ones
        let mut s = vec![vec![0i32; kmax + 2]; m + 1];
        for i in 1..=m {
            for j in 1..=(kmax + 1) {
                s[i][j] = next;
                next += 1;
            }
        }
        for i in 1..=m {
            let x = out_lits[i - 1];
            // j = 1
            if i == 1 {
                // s[1][1] <=> x
                clauses.push(vec![-s[1][1], x]);
                clauses.push(vec![s[1][1], -x]);
            } else {
                // s[i][1] <=> s[i-1][1] \/ x
                clauses.push(vec![-s[i][1], s[i - 1][1], x]);
                clauses.push(vec![-s[i - 1][1], s[i][1]]);
                clauses.push(vec![-x, s[i][1]]);
            }
            for j in 2..=(kmax + 1) {
                if j > i {
                    // impossible to have j ones in i bits — leave var unconstrained / force false
                    clauses.push(vec![-s[i][j]]);
                    continue;
                }
                if i == 1 {
                    clauses.push(vec![-s[i][j]]);
                    continue;
                }
                // s[i][j] <=> s[i-1][j] \/ (s[i-1][j-1] /\ x)
                clauses.push(vec![-s[i][j], s[i - 1][j], s[i - 1][j - 1]]);
                clauses.push(vec![-s[i][j], s[i - 1][j], x]);
                clauses.push(vec![-s[i - 1][j], s[i][j]]);
                clauses.push(vec![-s[i - 1][j - 1], -x, s[i][j]]);
            }
        }
        // Forbid at least rem_ub+1 ones.
        clauses.push(vec![-s[m][kmax + 1]]);
    }

    let mut f = BufWriter::new(File::create(path)?);
    writeln!(
        f,
        "c gray residual sandwich CNF weight<={ub} forced_ones={forced_ones}"
    )?;
    writeln!(f, "p cnf {} {}", next - 1, clauses.len())?;
    for cl in &clauses {
        for lit in cl {
            write!(f, "{lit} ")?;
        }
        writeln!(f, "0")?;
    }
    f.flush()?;
    Ok((free_n, free_vars))
}

fn run_kissat(bin: &Path, cnf: &Path, seconds: u64) -> String {
    let out = Command::new(bin)
        .arg(format!("--time={seconds}"))
        .arg("--walkinitially")
        .arg("--target=2")
        .arg("--quiet=1")
        .arg(cnf)
        .output();
    match out {
        Ok(o) => {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            s
        }
        Err(e) => format!("kissat spawn failed: {e}"),
    }
}

fn parse_model(out: &str, free_vars: &[i32]) -> Option<Vec<bool>> {
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

fn baseline(ch: &Challenge, samples: usize) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(2);
    let free_n = ch.circuit.data_wires - ch.prefix_zeros;
    let mut best_wt = usize::MAX;
    let mut best = vec![false; free_n];
    for _ in 0..samples {
        let mut free = vec![false; free_n];
        for b in &mut free {
            *b = rng.random();
        }
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &free)));
        if wt < best_wt {
            best_wt = wt;
            best = free;
            println!("  [baseline] wt={best_wt}");
        }
    }
    (best_wt, best)
}

fn polish(ch: &Challenge, start: &[bool], steps: usize) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(99);
    let free_n = start.len();
    let mut best = start.to_vec();
    let mut best_wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(ch, &best)));
    let mut cur = best.clone();
    let mut cur_wt = best_wt;
    // Simulated annealing polish after SAT model
    let mut temp = 2.5f64;
    for s in 0..steps {
        let nflip = if rng.random_bool(0.15) { 2 } else { 1 };
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
        if s % 5000 == 0 && s > 0 {
            temp *= 0.92;
            if temp < 0.05 {
                temp = 0.8;
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
        "data={} total={} chunks={} prefix={} kissat={}",
        ch.circuit.data_wires,
        ch.circuit.total_wires,
        ch.circuit.chunks.len(),
        ch.prefix_zeros,
        args.kissat.display()
    );

    println!("black-box baseline...");
    let (base_wt, base_free) = baseline(&ch, args.baseline_samples);
    println!("baseline wt={base_wt}");

    println!("building residual...");
    let t0 = Instant::now();
    let (residuals, final_abs) = residual_circuit(&ch);
    let n_one = final_abs[..ch.circuit.data_wires]
        .iter()
        .filter(|&&a| a == Abs::One)
        .count();
    let n_zero = final_abs[..ch.circuit.data_wires]
        .iter()
        .filter(|&&a| a == Abs::Zero)
        .count();
    let n_unk = final_abs[..ch.circuit.data_wires]
        .iter()
        .filter(|&&a| a == Abs::Unk)
        .count();
    let pats: usize = residuals.iter().map(|r| r.data.len()).sum();
    let max_kin = residuals
        .iter()
        .map(|r| r.in_wires.len())
        .max()
        .unwrap_or(0);
    println!(
        "residual in {:.2}s: perms={} patterns={} max_kin={}  out 0/1/unk={n_zero}/{n_one}/{n_unk}",
        t0.elapsed().as_secs_f64(),
        residuals.len(),
        pats,
        max_kin
    );
    let forced = n_one;
    println!("forced ones lower bound = {forced}");

    let mut best_wt = base_wt;
    let mut best_free = base_free;
    let mut ub = if args.start_ub == 0 {
        base_wt.saturating_sub(1)
    } else {
        args.start_ub
    };
    if ub < forced {
        ub = forced;
    }

    while ub >= forced {
        let cnf = args.work_dir.join(format!("w{ub}.cnf"));
        println!("\n=== try weight <= {ub} ===");
        let t1 = Instant::now();
        let (free_n, free_vars) = emit_cnf(&ch, &residuals, &final_abs, &cnf, ub).unwrap();
        let nbytes = fs::metadata(&cnf).unwrap().len();
        println!(
            "CNF {} ({:.1} MB) free_n={free_n} build={:.2}s",
            cnf.display(),
            nbytes as f64 / (1024.0 * 1024.0),
            t1.elapsed().as_secs_f64()
        );

        let t2 = Instant::now();
        let out = run_kissat(&args.kissat, &cnf, args.kissat_seconds);
        fs::write(cnf.with_extension("kissatout"), &out).unwrap();
        println!("kissat {:.1}s", t2.elapsed().as_secs_f64());

        if let Some(free) = parse_model(&out, &free_vars) {
            let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&bits_to_input(&ch, &free)));
            println!("SAT verified wt={wt}");
            let (pwt, pfree) = polish(&ch, &free, args.polish_steps);
            println!("after polish wt={pwt}");
            if pwt < best_wt {
                best_wt = pwt;
                best_free = pfree;
            }
            if ub == forced {
                break;
            }
            ub = ub.saturating_sub(args.step).max(forced);
        } else if out.contains("s UNSATISFIABLE") {
            println!("UNSAT — stopping descent");
            break;
        } else {
            println!("UNKNOWN/TIMEOUT — stop descent");
            for l in out.lines().take(12) {
                println!("  {l}");
            }
            break;
        }
    }

    // Final long polish from best
    println!("\nfinal polish from best={best_wt}...");
    let (pwt, pfree) = polish(&ch, &best_free, args.polish_steps * 2);
    if pwt < best_wt {
        best_wt = pwt;
        best_free = pfree;
    }
    let (x0, x1) = free_to_hex(&best_free);
    println!("DONE best_wt={best_wt} x_lo=0x{x0:016x} x_hi=0x{x1:016x} forced={forced}");
}
