//! SAT attack on the *plaintext* r57 circuit (unsealed for comparison).
//! Encode gate-57 as CNF with SSA wire versions + sequential-counter weight bound.
//! Also runs a short black-box baseline for comparison.

use clap::Parser;
use local_mixing::circuit::CircuitSeq;
use local_mixing::sandwich::hamming_weight;
use primitive_types::U256 as u256;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "challenges/lowweight128_plain.SEALED.txt")]
    plain: PathBuf,
    #[arg(long, default_value_t = 64)]
    prefix_zeros: usize,
    #[arg(long, default_value = "challenges/work_plain_cnf")]
    work_dir: PathBuf,
    #[arg(long, default_value = "third_party/kissat/build/kissat")]
    kissat: PathBuf,
    #[arg(long, default_value_t = 0)]
    start_ub: usize,
    #[arg(long, default_value_t = 1)]
    step: usize,
    #[arg(long, default_value_t = 300)]
    kissat_seconds: u64,
    #[arg(long, default_value_t = 20_000)]
    baseline_samples: usize,
}

fn load_plain(path: &Path) -> (usize, CircuitSeq) {
    let s = fs::read_to_string(path).unwrap();
    let mut n = 0usize;
    let mut gates = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(r) = t.strip_prefix("n ") {
            n = r.parse().unwrap();
            continue;
        }
        if t.starts_with("m ") {
            continue;
        }
        let parts: Vec<u16> = t.split(',').map(|x| x.parse().unwrap()).collect();
        assert_eq!(parts.len(), 3);
        gates.push([parts[0], parts[1], parts[2]]);
    }
    assert!(n > 0);
    (n, CircuitSeq { gates })
}

fn eval_wt(c: &CircuitSeq, n: usize, prefix: usize, free: &[bool]) -> usize {
    let mut state = u256::zero();
    for (i, &b) in free.iter().enumerate() {
        if b {
            state |= u256::one() << (prefix + i);
        }
    }
    let out = c.evaluate_256(state);
    let mut bits = Vec::with_capacity(n);
    for i in 0..n {
        bits.push(((out >> i) & u256::one()) == u256::one());
    }
    hamming_weight(&bits)
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

/// Emit CNF. Returns (nvars, free_var_ids).
fn emit_cnf(
    c: &CircuitSeq,
    n: usize,
    prefix: usize,
    path: &Path,
    ub: usize,
) -> std::io::Result<(usize, Vec<i32>)> {
    let free_n = n - prefix;
    let mut next = 1i32;
    let mut free_vars = Vec::with_capacity(free_n);
    for _ in 0..free_n {
        free_vars.push(next);
        next += 1;
    }

    // Current SSA var for each wire; None means constant 0 (prefix / never written as 1 yet with const).
    // We use Option: Some(v) unknown, and a parallel Abs for const-prop.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Abs {
        Zero,
        One,
        Unk(i32),
    }

    let mut cur: Vec<Abs> = vec![Abs::Unk(0); n];
    for w in 0..prefix {
        cur[w] = Abs::Zero;
    }
    for (i, &v) in free_vars.iter().enumerate() {
        cur[prefix + i] = Abs::Unk(v);
    }

    let mut clauses: Vec<Vec<i32>> = Vec::new();

    let lit = |a: Abs| -> Option<i32> {
        match a {
            Abs::Zero => None, // const false — handled specially
            Abs::One => None,
            Abs::Unk(v) => Some(v),
        }
    };

    for g in &c.gates {
        let a = g[0] as usize;
        let b = g[1] as usize;
        let cwire = g[2] as usize;
        let ain = cur[a];
        let bin = cur[b];
        let cin = cur[cwire];

        // flip = b | ~c
        let flip_abs = match (bin, cin) {
            (Abs::One, _) | (_, Abs::Zero) => Abs::One,
            (Abs::Zero, Abs::One) => Abs::Zero,
            _ => {
                // need a flip var
                let fv = next;
                next += 1;
                // Encode flip <=> b \/ ~c  (with consts folded)
                match (bin, cin) {
                    (Abs::Unk(bv), Abs::Unk(cv)) => {
                        // flip = b | ~c
                        clauses.push(vec![-fv, bv, -cv]);
                        clauses.push(vec![-bv, fv]);
                        clauses.push(vec![cv, fv]);
                    }
                    (Abs::Unk(bv), Abs::One) => {
                        // flip = b | 0 = b
                        clauses.push(vec![-fv, bv]);
                        clauses.push(vec![-bv, fv]);
                    }
                    (Abs::Zero, Abs::Unk(cv)) => {
                        // flip = 0 | ~c = ~c
                        clauses.push(vec![-fv, -cv]);
                        clauses.push(vec![cv, fv]);
                    }
                    (Abs::One, _) | (_, Abs::Zero) => unreachable!(),
                    (Abs::Zero, Abs::One) => unreachable!(),
                    (Abs::Unk(_), Abs::Zero) => unreachable!(), // flip=1
                    (Abs::One, Abs::Unk(_)) => unreachable!(),
                }
                Abs::Unk(fv)
            }
        };

        // a_out = ain XOR flip
        let aout = match (ain, flip_abs) {
            (Abs::Zero, Abs::Zero) | (Abs::One, Abs::One) => Abs::Zero,
            (Abs::Zero, Abs::One) | (Abs::One, Abs::Zero) => Abs::One,
            (Abs::Unk(av), Abs::Zero) => Abs::Unk(av), // XOR 0
            (Abs::Unk(av), Abs::One) => {
                // XOR 1 = NOT av
                let ov = next;
                next += 1;
                clauses.push(vec![-ov, -av]);
                clauses.push(vec![ov, av]);
                Abs::Unk(ov)
            }
            (Abs::Zero, Abs::Unk(fv)) => Abs::Unk(fv), // 0 XOR f = f
            (Abs::One, Abs::Unk(fv)) => {
                let ov = next;
                next += 1;
                clauses.push(vec![-ov, -fv]);
                clauses.push(vec![ov, fv]);
                Abs::Unk(ov)
            }
            (Abs::Unk(av), Abs::Unk(fv)) => {
                let ov = next;
                next += 1;
                // ov <=> av XOR fv
                clauses.push(vec![-av, -fv, -ov]);
                clauses.push(vec![av, fv, -ov]);
                clauses.push(vec![-av, fv, ov]);
                clauses.push(vec![av, -fv, ov]);
                Abs::Unk(ov)
            }
        };
        cur[a] = aout;
        let _ = lit;
    }

    // Collect output lits / forced ones
    let mut out_lits = Vec::new();
    let mut forced_ones = 0usize;
    for w in 0..n {
        match cur[w] {
            Abs::Zero => {}
            Abs::One => forced_ones += 1,
            Abs::Unk(v) => out_lits.push(v),
        }
    }
    assert!(
        forced_ones <= ub,
        "forced ones {forced_ones} already exceed ub {ub}"
    );
    let rem_ub = ub - forced_ones;
    let m = out_lits.len();
    if rem_ub < m {
        let kmax = rem_ub;
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
    }

    let mut f = BufWriter::new(File::create(path)?);
    writeln!(
        f,
        "c plaintext r57 CNF weight<={ub} forced_ones={forced_ones} outs={}",
        out_lits.len()
    )?;
    writeln!(f, "p cnf {} {}", next - 1, clauses.len())?;
    for cl in &clauses {
        for lit in cl {
            write!(f, "{lit} ")?;
        }
        writeln!(f, "0")?;
    }
    f.flush()?;
    println!(
        "  forced_ones={forced_ones} unk_outs={} nvars={} clauses={}",
        out_lits.len(),
        next - 1,
        clauses.len()
    );
    Ok((free_n, free_vars))
}

fn run_kissat(bin: &Path, cnf: &Path, seconds: u64) -> String {
    let out = Command::new(bin)
        .arg(format!("--time={seconds}"))
        .arg("--walkinitially")
        .arg("--target=2")
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

fn baseline(c: &CircuitSeq, n: usize, prefix: usize, samples: usize) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(2);
    let free_n = n - prefix;
    let mut best_wt = usize::MAX;
    let mut best = vec![false; free_n];
    for _ in 0..samples {
        let free: Vec<bool> = (0..free_n).map(|_| rng.random()).collect();
        let wt = eval_wt(c, n, prefix, &free);
        if wt < best_wt {
            best_wt = wt;
            best = free;
            println!("  [baseline] wt={best_wt}");
        }
    }
    // short hill
    let mut cur = best.clone();
    let mut cur_wt = best_wt;
    for _ in 0..samples {
        let i = rng.random_range(0..free_n);
        cur[i] = !cur[i];
        let wt = eval_wt(c, n, prefix, &cur);
        if wt <= cur_wt {
            cur_wt = wt;
            if wt < best_wt {
                best_wt = wt;
                best = cur.clone();
                println!("  [hill] wt={best_wt}");
            }
        } else {
            cur[i] = !cur[i];
        }
    }
    (best_wt, best)
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.work_dir).unwrap();
    let (n, circ) = load_plain(&args.plain);
    let prefix = args.prefix_zeros;
    println!(
        "PLAINTEXT SAT: n={n} m={} prefix={prefix} free={}",
        circ.gates.len(),
        n - prefix
    );

    println!("black-box baseline...");
    let (base_wt, _) = baseline(&circ, n, prefix, args.baseline_samples);
    println!("baseline best wt={base_wt}");

    let mut ub = if args.start_ub == 0 {
        base_wt.saturating_sub(1)
    } else {
        args.start_ub
    };
    let mut best_wt = base_wt;
    let mut best_free = vec![false; n - prefix];

    while ub > 0 {
        let cnf = args.work_dir.join(format!("w{ub}.cnf"));
        println!("\n=== try weight <= {ub} ===");
        let t0 = Instant::now();
        let (free_n, free_vars) = emit_cnf(&circ, n, prefix, &cnf, ub).unwrap();
        println!(
            "CNF {:.2} MB in {:.2}s",
            fs::metadata(&cnf).unwrap().len() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64()
        );
        let t1 = Instant::now();
        let out = run_kissat(&args.kissat, &cnf, args.kissat_seconds);
        fs::write(cnf.with_extension("kissatout"), &out).unwrap();
        println!("kissat {:.1}s", t1.elapsed().as_secs_f64());

        if let Some(free) = parse_free(&out, &free_vars) {
            let wt = eval_wt(&circ, n, prefix, &free);
            println!("SAT verified wt={wt} x=0x{:016x}", free_to_hex(&free));
            assert_eq!(free.len(), free_n);
            if wt < best_wt {
                best_wt = wt;
                best_free = free;
            }
            if ub <= 1 {
                break;
            }
            ub = ub.saturating_sub(args.step).max(1);
        } else if out.contains("s UNSATISFIABLE") {
            println!("UNSAT — stopping");
            break;
        } else {
            println!("TIMEOUT/UNKNOWN — stop");
            for l in out.lines().take(8) {
                println!("  {l}");
            }
            break;
        }
    }

    println!(
        "DONE plaintext best_wt={best_wt} x=0x{:016x} (baseline={base_wt})",
        free_to_hex(&best_free)
    );
}
