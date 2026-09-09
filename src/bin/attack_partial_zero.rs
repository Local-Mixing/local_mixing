//! Partial-zero challenge: find x,y with C(x || 0^k) = y || 0^k (k = n/2 by default).
//!
//! Generates a random r57 circuit, saves `repr()`, exports an MQ system over F2, and
//! solves via forward search or meet-in-the-middle (same asymptotic time; MITM needs
//! 2^{n-k} memory so forward is used for larger instances).

use clap::Parser;
use local_mixing::circuit::{CircuitSeq, Gate};
use primitive_types::U256 as u256;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn parse_u64_auto(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        t.parse::<u64>().map_err(|e| e.to_string())
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Method {
    Auto,
    Forward,
    Mitm,
}

#[derive(Parser, Debug)]
#[command(about = "Partial-zero attack on a random r57 circuit (forward / MITM + MQ export)")]
struct Args {
    #[arg(short = 'n', long, default_value_t = 16)]
    wires: usize,
    /// Gate count. If omitted, uses n * ceil(log2(n)).
    #[arg(short = 'm', long)]
    gates: Option<usize>,
    /// Suffix zeros on input and output. Default: n/2.
    #[arg(short = 'k', long)]
    zeros: Option<usize>,
    #[arg(long, default_value = "0xC0FFEE", value_parser = parse_u64_auto)]
    seed: u64,
    #[arg(long, default_value = "challenges/partial_zero")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = false)]
    export_only: bool,
    #[arg(long, value_enum, default_value_t = Method::Auto)]
    method: Method,
    /// Wall-clock timeout for the solve phase (seconds). 0 = no limit.
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,
    /// Also write a DIMACS CNF encoding (for SAT solvers).
    #[arg(long, default_value_t = false)]
    export_cnf: bool,
}

fn default_gates(n: usize) -> usize {
    let lg = (n as f64).log2().ceil() as usize;
    n * lg
}

fn random_circuit(n: usize, m: usize, rng: &mut impl Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut g = [0u16; 3];
            let mut used = vec![false; n];
            for j in 0..3 {
                loop {
                    let v = rng.random_range(0..n) as u16;
                    if !used[v as usize] {
                        used[v as usize] = true;
                        g[j] = v;
                        break;
                    }
                }
            }
            if gates.last() == Some(&g) {
                continue;
            }
            gates.push(g);
            break;
        }
    }
    CircuitSeq { gates }
}

fn pack_input_u64(free: u64, k: usize) -> u64 {
    free << k
}

fn pack_input_u256(free: u64, k: usize) -> u256 {
    u256::from(free) << k
}

fn unpack_free_u64(state: u64, k: usize, n: usize) -> u64 {
    let free_bits = n - k;
    (state >> k) & ((1u64 << free_bits) - 1)
}

fn unpack_free_u256(state: u256, k: usize, n: usize) -> u64 {
    let mut free = 0u64;
    let free_bits = n - k;
    for i in 0..free_bits {
        if ((state >> (k + i)) & u256::one()) == u256::one() {
            free |= 1u64 << i;
        }
    }
    free
}

fn low_k_zero_u64(state: u64, k: usize) -> bool {
    let mask = if k >= 64 {
        u64::MAX
    } else {
        (1u64 << k) - 1
    };
    state & mask == 0
}

fn low_k_zero_u256(state: u256, k: usize) -> bool {
    let mask = (u256::one() << k) - u256::one();
    (state & mask).is_zero()
}

fn eval_u64(state: u64, gates: &[[u16; 3]]) -> u64 {
    let mut s = state as usize;
    for &g in gates {
        s = Gate::evaluate_index(s, g);
    }
    s as u64
}

fn eval_u64_inv(state: u64, gates: &[[u16; 3]]) -> u64 {
    let mut s = state as usize;
    for &g in gates.iter().rev() {
        s = Gate::evaluate_index(s, g);
    }
    s as u64
}

fn eval_u256(state: u256, gates: &[[u16; 3]]) -> u256 {
    let mut s = state;
    for &g in gates {
        s = Gate::evaluate_index_256(s, g);
    }
    s
}

fn eval_u256_inv(state: u256, gates: &[[u16; 3]]) -> u256 {
    let mut s = state;
    for &g in gates.iter().rev() {
        s = Gate::evaluate_index_256(s, g);
    }
    s
}

struct SolveResult {
    hit: Option<(u64, u64)>,
    method: &'static str,
    evals: u64,
    timed_out: bool,
    elapsed: Duration,
}

fn forward_u64(
    gates: &[[u16; 3]],
    n: usize,
    k: usize,
    timeout: Option<Duration>,
) -> SolveResult {
    let free_bits = n - k;
    let n_free = 1u64 << free_bits;
    let stop = Arc::new(AtomicBool::new(false));
    let evals = AtomicU64::new(0);
    let t0 = Instant::now();

    // Timeout watcher: park in short slices so we can join quickly after a hit.
    let watcher = timeout.map(|lim| {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let deadline = Instant::now() + lim;
            while Instant::now() < deadline {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            stop.store(true, Ordering::Relaxed);
        })
    });

    let chunk = (1u64 << free_bits.saturating_sub(8).min(16)).max(1);
    let n_chunks = n_free.div_ceil(chunk);
    let found = (0..n_chunks)
        .into_par_iter()
        .find_map_any(|ci| {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            let start = ci * chunk;
            let end = (start + chunk).min(n_free);
            let mut local = 0u64;
            for x in start..end {
                if (local & 0xfff) == 0 && stop.load(Ordering::Relaxed) {
                    evals.fetch_add(local, Ordering::Relaxed);
                    return None;
                }
                let out = eval_u64(pack_input_u64(x, k), gates);
                local += 1;
                if low_k_zero_u64(out, k) {
                    evals.fetch_add(local, Ordering::Relaxed);
                    stop.store(true, Ordering::Relaxed);
                    let y = unpack_free_u64(out, k, n);
                    return Some((x, y));
                }
            }
            evals.fetch_add(local, Ordering::Relaxed);
            None
        });

    // Unblock watcher before join.
    stop.store(true, Ordering::Relaxed);
    let elapsed = t0.elapsed();
    if let Some(h) = watcher {
        let _ = h.join();
    }
    let timed_out = found.is_none()
        && timeout.is_some_and(|lim| elapsed >= lim.saturating_sub(Duration::from_millis(100)));
    SolveResult {
        hit: found,
        method: "forward",
        evals: evals.load(Ordering::Relaxed),
        timed_out,
        elapsed,
    }
}

fn mitm_u64(gates: &[[u16; 3]], n: usize, k: usize) -> Option<(u64, u64, u64)> {
    let free_bits = n - k;
    assert!(free_bits <= 28, "MITM table too large");
    let n_free = 1u64 << free_bits;
    let mid = gates.len() / 2;
    let (left, right) = gates.split_at(mid);

    let mut table: FxHashMap<u64, u64> =
        FxHashMap::with_capacity_and_hasher(n_free as usize, Default::default());
    for x in 0..n_free {
        table.insert(eval_u64(pack_input_u64(x, k), left), x);
    }
    for y in 0..n_free {
        let mid_state = eval_u64_inv(pack_input_u64(y, k), right);
        if let Some(&x) = table.get(&mid_state) {
            return Some((x, y, eval_u64(pack_input_u64(x, k), gates)));
        }
    }
    None
}

/// Export MQ over F2 in a Sage-friendly ANF text format.
fn export_mq(c: &CircuitSeq, n: usize, k: usize, path: &Path) -> std::io::Result<()> {
    let m = c.gates.len();
    let mut f = File::create(path)?;

    writeln!(f, "# Partial-zero MQ over GF(2)")?;
    writeln!(f, "# n={n} m={m} k={k}")?;
    writeln!(f, "# vars: x0..x{} (inputs), z0..z{} (gate outs)", n - 1, m - 1)?;
    writeln!(f, "# each line: monomials of an equation summed to 0 (XOR)")?;
    writeln!(f, "# monomials are products of vars, or '1' for constant")?;
    writeln!(f, "n_vars {}", n + m)?;
    writeln!(f, "n_eqs {}", m + 2 * k)?;

    let xin = |w: usize| format!("x{w}");
    let zin = |i: usize| format!("z{i}");
    let mut live: Vec<String> = (0..n).map(xin).collect();

    for w in 0..k {
        writeln!(f, "{}", xin(w))?;
    }

    for (i, &[a, c1, c2]) in c.gates.iter().enumerate() {
        let (a, c1, c2) = (a as usize, c1 as usize, c2 as usize);
        writeln!(
            f,
            "{} {} {}*{} {} 1",
            zin(i),
            live[a],
            live[c1],
            live[c2],
            live[c2]
        )?;
        live[a] = zin(i);
    }

    for w in 0..k {
        writeln!(f, "{}", live[w])?;
    }
    writeln!(f, "# END_EQS")?;
    writeln!(f, "# live_final {}", live.join(" "))?;
    Ok(())
}

/// DIMACS CNF for the same system. r57: a' = a XOR (c1 OR NOT c2).
/// Tseitin on SSA wires. Vars: 1..=n inputs, then gate outs.
fn export_cnf(c: &CircuitSeq, n: usize, k: usize, path: &Path) -> std::io::Result<()> {
    let m = c.gates.len();
    // var ids: input w -> w+1; gate i out -> n+i+1
    let xin = |w: usize| w + 1;
    let zin = |i: usize| n + i + 1;
    let mut live: Vec<usize> = (0..n).map(xin).collect();
    let mut clauses: Vec<Vec<i32>> = Vec::new();

    // input zeros
    for w in 0..k {
        clauses.push(vec![-(xin(w) as i32)]);
    }

    for (i, &[a, c1, c2]) in c.gates.iter().enumerate() {
        let a = a as usize;
        let c1 = c1 as usize;
        let c2 = c2 as usize;
        let av = live[a] as i32;
        let p = live[c1] as i32; // c1
        let q = live[c2] as i32; // c2
        let z = zin(i) as i32;
        // f = c1 OR NOT c2: introduce t <=> (p ∨ ¬q)
        // Use direct encoding of z <=> av XOR f with f = p∨¬q without extra var:
        // z = av XOR (p OR NOT q)
        // Cases on q:
        //  q=0 => f=1 => z = NOT av
        //  q=1 => f=p => z = av XOR p
        // CNF for z ↔ av ⊕ (p ∨ ¬q):
        // Expand: f = p∨¬q, z ↔ av⊕f.
        // Clauses for f = p ∨ ¬q (aux tf = n+m+1+i):
        let tf = (n + m + 1 + i) as i32;
        // tf => p∨¬q : (¬tf ∨ p ∨ ¬q)
        clauses.push(vec![-tf, p, -q]);
        // p => tf : (¬p ∨ tf)
        clauses.push(vec![-p, tf]);
        // ¬q => tf : (q ∨ tf)
        clauses.push(vec![q, tf]);
        // z <=> av XOR tf
        // (¬z ∨ av ∨ tf) (z ∨ ¬av ∨ tf) (z ∨ av ∨ ¬tf) (¬z ∨ ¬av ∨ ¬tf)
        clauses.push(vec![-z, av, tf]);
        clauses.push(vec![z, -av, tf]);
        clauses.push(vec![z, av, -tf]);
        clauses.push(vec![-z, -av, -tf]);
        live[a] = zin(i);
    }

    for w in 0..k {
        clauses.push(vec![-(live[w] as i32)]);
    }

    let n_vars = n + m + m; // inputs + gate outs + tf aux
    let mut f = File::create(path)?;
    writeln!(
        f,
        "c partial-zero r57 n={n} m={m} k={k}"
    )?;
    writeln!(f, "p cnf {n_vars} {}", clauses.len())?;
    for cl in &clauses {
        for lit in cl {
            write!(f, "{lit} ")?;
        }
        writeln!(f, "0")?;
    }
    Ok(())
}

fn ensure_dir(p: &Path) {
    fs::create_dir_all(p).expect("create out_dir");
}

fn main() {
    let args = Args::parse();
    let n = args.wires;
    let m = args.gates.unwrap_or_else(|| default_gates(n));
    let k = args.zeros.unwrap_or(n / 2);
    assert!(k < n, "k must be < n");
    assert!(n <= 256, "n too large");
    let free_bits = n - k;

    ensure_dir(&args.out_dir);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let circuit = random_circuit(n, m, &mut rng);

    let circ_path = args.out_dir.join(format!("circuit_n{n}_m{m}_s{:x}.repr", args.seed));
    fs::write(&circ_path, circuit.repr()).expect("write circuit repr");
    println!("wrote circuit {}", circ_path.display());
    println!(
        "n={n} m={m} (=n*ceil(lg n) default) k={k} free_bits={free_bits} seed=0x{:x}",
        args.seed
    );

    let mq_path = args.out_dir.join(format!("mq_n{n}_m{m}_s{:x}.anf", args.seed));
    export_mq(&circuit, n, k, &mq_path).expect("export mq");
    println!(
        "wrote MQ {} ({} vars, {} eqs)",
        mq_path.display(),
        n + m,
        m + 2 * k
    );

    if args.export_cnf {
        let cnf_path = args.out_dir.join(format!("mq_n{n}_m{m}_s{:x}.cnf", args.seed));
        export_cnf(&circuit, n, k, &cnf_path).expect("export cnf");
        println!("wrote CNF {}", cnf_path.display());
    }

    let meta_path = args.out_dir.join(format!("meta_n{n}_m{m}_s{:x}.txt", args.seed));
    let mut meta = File::create(&meta_path).expect("meta");
    writeln!(meta, "n={n}").unwrap();
    writeln!(meta, "m={m}").unwrap();
    writeln!(meta, "k={k}").unwrap();
    writeln!(meta, "seed={:#x}", args.seed).unwrap();

    if args.export_only {
        println!("export_only: skipping solve");
        return;
    }

    let timeout = (args.timeout_secs > 0).then(|| Duration::from_secs(args.timeout_secs));
    let method = match args.method {
        Method::Auto => {
            if free_bits <= 24 && n <= 64 {
                Method::Mitm
            } else {
                Method::Forward
            }
        }
        other => other,
    };

    println!(
        "solve method={method:?} timeout={:?} (MITM≠faster here; same ~2^{free_bits} time, MITM needs ~2^{free_bits} mem)",
        timeout
    );

    let result = match method {
        Method::Mitm if n <= 64 && free_bits <= 28 => {
            let t0 = Instant::now();
            match mitm_u64(&circuit.gates, n, k) {
                Some((x, y, _)) => SolveResult {
                    hit: Some((x, y)),
                    method: "mitm",
                    evals: 2u64 << free_bits,
                    timed_out: false,
                    elapsed: t0.elapsed(),
                },
                None => {
                    println!("MITM: no hit; running forward to confirm...");
                    let mut r = forward_u64(&circuit.gates, n, k, timeout);
                    r.elapsed = t0.elapsed();
                    r
                }
            }
        }
        Method::Mitm => {
            println!("MITM refused (need n<=64 and free_bits<=28); using forward");
            assert!(n <= 64, "n>64: use --export-only");
            forward_u64(&circuit.gates, n, k, timeout)
        }
        Method::Forward | Method::Auto => {
            assert!(n <= 64, "forward fast path requires n<=64");
            forward_u64(&circuit.gates, n, k, timeout)
        }
    };

    report_result(&circuit, n, k, &args.out_dir, args.seed, m, result);
}

fn report_result(
    circuit: &CircuitSeq,
    n: usize,
    k: usize,
    out_dir: &Path,
    seed: u64,
    m: usize,
    result: SolveResult,
) {
    let rate = result.evals as f64 / result.elapsed.as_secs_f64().max(1e-9);
    if result.timed_out {
        println!(
            "TIMEOUT method={} elapsed={:?} evals={} ({:.2e}/s)",
            result.method, result.elapsed, result.evals, rate
        );
        return;
    }

    let Some((x, y)) = result.hit else {
        println!(
            "NO SOLUTION method={} elapsed={:?} evals={} ({:.2e}/s)",
            result.method, result.elapsed, result.evals, rate
        );
        return;
    };

    let out = eval_u64(pack_input_u64(x, k), &circuit.gates);
    assert!(low_k_zero_u64(out, k));
    assert_eq!(unpack_free_u64(out, k, n), y);

    println!(
        "FOUND method={} x=0x{x:x} y=0x{y:x} elapsed={:?} evals={} ({:.2e}/s)",
        result.method, result.elapsed, result.evals, rate
    );
    let sol_path = out_dir.join(format!("solution_n{n}_m{m}_s{seed:x}.txt"));
    let mut sf = File::create(&sol_path).unwrap();
    writeln!(sf, "x=0x{x:x}").unwrap();
    writeln!(sf, "y=0x{y:x}").unwrap();
    writeln!(sf, "k={k}").unwrap();
    writeln!(sf, "method={}", result.method).unwrap();
    writeln!(sf, "elapsed_ms={}", result.elapsed.as_millis()).unwrap();
    writeln!(sf, "evals={}", result.evals).unwrap();
    println!("wrote {}", sol_path.display());
}
