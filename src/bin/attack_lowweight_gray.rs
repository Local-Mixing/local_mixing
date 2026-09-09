//! Gray-box low-weight attack: constant-propagate through chunk truth tables
//! (zero prefix + zero ancilla), then SAT-encode only the residual unknown
//! fragment and binary-search Hamming weight with Z3.

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
    #[arg(long, default_value = "challenges/work_gray")]
    work_dir: PathBuf,
    #[arg(long, default_value_t = 300_000)]
    timeout_ms: u64,
    #[arg(long, default_value_t = 1000)]
    baseline_samples: usize,
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

                // Enumerate consistent full patterns under constants.
                let mut pairs: Vec<(usize, usize)> = Vec::new(); // (full_in, full_out)
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
                            match st[w as usize] {
                                Abs::One => pat |= 1 << i,
                                Abs::Zero | Abs::Unk => {}
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
                    continue; // fully constant chunk under this prefix
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

fn emit_smt(
    ch: &Challenge,
    residuals: &[ResidualPerm],
    final_abs: &[Abs],
    path: &Path,
    ub: usize,
) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "(set-option :produce-models true)")?;
    writeln!(f, "(set-logic ALL)")?;

    let free: Vec<usize> = (ch.prefix_zeros..ch.circuit.data_wires).collect();
    for &w in &free {
        writeln!(f, "(declare-const in{w} Bool)")?;
    }

    let mut cur: Vec<Option<String>> = vec![None; ch.circuit.total_wires];
    for &w in &free {
        cur[w] = Some(format!("in{w}"));
    }

    for (ri, rp) in residuals.iter().enumerate() {
        let k_in = rp.in_wires.len();
        let mut outs = Vec::new();
        for &w in &rp.out_wires {
            let name = format!("n{ri}_{w}");
            writeln!(f, "(declare-const {name} Bool)")?;
            outs.push(name);
        }
        for pat in 0..(1 << k_in) {
            write!(f, "(assert (=> (and")?;
            for (i, &w) in rp.in_wires.iter().enumerate() {
                let src = cur[w as usize]
                    .as_ref()
                    .unwrap_or_else(|| panic!("missing def for wire {w} at residual {ri}"));
                if ((pat >> i) & 1) != 0 {
                    write!(f, " {src}")?;
                } else {
                    write!(f, " (not {src})")?;
                }
            }
            // empty and is true
            if k_in == 0 {
                write!(f, " true")?;
            }
            write!(f, ") (and")?;
            let outv = rp.data[pat];
            for (i, oname) in outs.iter().enumerate() {
                if ((outv >> i) & 1) != 0 {
                    write!(f, " {oname}")?;
                } else {
                    write!(f, " (not {oname})")?;
                }
            }
            writeln!(f, ")))")?;
        }
        for (i, &w) in rp.out_wires.iter().enumerate() {
            cur[w as usize] = Some(outs[i].clone());
        }
    }

    write!(f, "(define-fun weight () Int (+")?;
    for w in 0..ch.circuit.data_wires {
        match final_abs[w] {
            Abs::Zero => write!(f, " 0")?,
            Abs::One => write!(f, " 1")?,
            Abs::Unk => {
                let v = cur[w]
                    .as_ref()
                    .unwrap_or_else(|| panic!("unk out {w} has no symbol"));
                write!(f, " (ite {v} 1 0)")?;
            }
        }
    }
    writeln!(f, "))")?;
    writeln!(f, "(assert (<= weight {ub}))")?;
    writeln!(f, "(check-sat)")?;
    write!(f, "(get-value (weight")?;
    for &w in &free {
        write!(f, " in{w}")?;
    }
    writeln!(f, "))")?;
    f.flush()
}

fn run_z3(smt: &Path, timeout_ms: u64) -> String {
    let mut cmd = Command::new("z3");
    if timeout_ms > 0 {
        cmd.arg(format!("-t:{timeout_ms}"));
    }
    cmd.arg("-smt2").arg(smt.as_os_str());
    let out = cmd.output().expect("z3");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.is_empty() {
        s.push_str("\n;stderr\n");
        s.push_str(&err);
    }
    s
}

fn parse_sat(out: &str) -> Option<bool> {
    for line in out.lines() {
        match line.trim() {
            "sat" => return Some(true),
            "unsat" => return Some(false),
            "timeout" | "unknown" => return None,
            _ => {}
        }
    }
    None
}

fn baseline(ch: &Challenge, samples: usize) -> usize {
    let mut rng = StdRng::seed_from_u64(2);
    let free = ch.circuit.data_wires - ch.prefix_zeros;
    let mut best = usize::MAX;
    for _ in 0..samples {
        let mut input = vec![false; ch.circuit.data_wires];
        for i in 0..free {
            input[ch.prefix_zeros + i] = rng.random();
        }
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&input));
        if wt < best {
            best = wt;
            println!("  [baseline] wt={best}");
        }
    }
    best
}

fn verify_model(ch: &Challenge, z3out: &str) -> Option<(usize, Vec<bool>)> {
    // Parse inW true/false from get-value lines
    let mut input = vec![false; ch.circuit.data_wires];
    for line in z3out.lines() {
        // (in64 true) or (in64 false) possibly nested
        let t = line.trim().trim_start_matches('(').trim_end_matches(')');
        let parts: Vec<_> = t.split_whitespace().collect();
        if parts.len() == 2 && parts[0].starts_with("in") {
            if let Ok(w) = parts[0][2..].parse::<usize>() {
                if w < input.len() {
                    input[w] = parts[1].starts_with('t');
                }
            }
        }
    }
    // If nothing parsed, fail
    if input.iter().skip(ch.prefix_zeros).all(|&b| !b)
        && !z3out.contains(" in")
    {
        // might be all-zero free which is valid; check weight line
    }
    let y = ch.circuit.evaluate_data_bits(&input);
    Some((hamming_weight(&y), input))
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.work_dir).unwrap();
    let ch = load_challenge(&args.challenge);
    println!(
        "data={} total={} chunks={} prefix={}",
        ch.circuit.data_wires,
        ch.circuit.total_wires,
        ch.circuit.chunks.len(),
        ch.prefix_zeros
    );

    println!("black-box baseline...");
    let base = baseline(&ch, args.baseline_samples);
    println!("baseline wt={base}");

    println!("building residual gray-box circuit...");
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
    println!(
        "residual in {:.2}s: perms={} patterns={}  out 0/1/unk={n_zero}/{n_one}/{n_unk}",
        t0.elapsed().as_secs_f64(),
        residuals.len(),
        pats
    );
    let forced = n_one;
    println!("forced ones lower bound = {forced}");

    // Descend from baseline: get SAT models at decreasing ub, polish each.
    let mut ub = base;
    let mut best_wt = base;
    let mut best_input = vec![false; ch.circuit.data_wires];
    let step = 2usize;
    while ub >= forced {
        let smt = args.work_dir.join(format!("ub_{ub}.smt2"));
        println!("\n=== try weight <= {ub} ===");
        emit_smt(&ch, &residuals, &final_abs, &smt, ub).unwrap();
        println!("smt size {} KB", fs::metadata(&smt).unwrap().len() / 1024);
        let t1 = Instant::now();
        let out = run_z3(&smt, args.timeout_ms);
        fs::write(smt.with_extension("z3out"), &out).unwrap();
        println!("z3 {:.1}s", t1.elapsed().as_secs_f64());
        match parse_sat(&out) {
            Some(true) => {
                println!("SAT");
                if let Some((wt, input)) = verify_model(&ch, &out) {
                    println!("verified eval weight={wt}");
                    // Polish with 1-flip / random-restart hill using gray eval
                    let (pwt, pinput) = polish(&ch, &input, 20_000);
                    println!("after polish wt={pwt}");
                    if pwt < best_wt {
                        best_wt = pwt;
                        best_input = pinput;
                    }
                }
                if ub == forced {
                    break;
                }
                ub = ub.saturating_sub(step).max(forced);
            }
            Some(false) => {
                println!("UNSAT — stopping descent");
                break;
            }
            None => {
                println!("UNKNOWN/TIMEOUT — try slightly higher or stop");
                for line in out.lines().take(8) {
                    println!("  {line}");
                }
                break;
            }
        }
    }

    let mut x = 0u128;
    for i in 0..(ch.circuit.data_wires - ch.prefix_zeros) {
        if best_input[ch.prefix_zeros + i] {
            x |= 1u128 << i;
        }
    }
    println!(
        "DONE gray-box best_wt={best_wt} x=0x{x:x} (baseline={base}, forced={forced})"
    );
}

fn polish(ch: &Challenge, start: &[bool], samples: usize) -> (usize, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(99);
    let mut best = start.to_vec();
    let mut best_wt = hamming_weight(&ch.circuit.evaluate_data_bits(&best));
    let free0 = ch.prefix_zeros;
    let free_bits = ch.circuit.data_wires - free0;
    let mut cur = best.clone();
    let mut cur_wt = best_wt;
    for s in 0..samples {
        let i = free0 + rng.random_range(0..free_bits);
        cur[i] = !cur[i];
        let wt = hamming_weight(&ch.circuit.evaluate_data_bits(&cur));
        if wt <= cur_wt {
            cur_wt = wt;
            if wt < best_wt {
                best_wt = wt;
                best = cur.clone();
                println!("  [polish] wt={best_wt}");
            }
        } else {
            cur[i] = !cur[i]; // revert
        }
        if s % 5000 == 0 && s > 0 {
            // random restart from best
            cur = best.clone();
            cur_wt = best_wt;
        }
    }
    (best_wt, best)
}
