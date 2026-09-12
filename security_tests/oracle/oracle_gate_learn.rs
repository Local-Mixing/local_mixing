use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use clap::Parser;
use local_mixing::circuit::CircuitSeq;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Learn a hidden random gate circuit from oracle examples via SAT."
)]
struct Args {
    #[arg(short = 'n', long = "n", default_value_t = 8)]
    wires: usize,

    #[arg(short = 'm', long = "m", default_value_t = 10)]
    gates: usize,

    #[arg(long, default_value_t = 3)]
    initial_queries: usize,

    #[arg(long, default_value_t = 10)]
    max_rounds: usize,

    #[arg(long, default_value_t = 500)]
    validation_queries: usize,

    #[arg(long, default_value_t = 0x0faca_de2026_u64)]
    seed: u64,

    #[arg(long, default_value = "work/oracle_gate_learn")]
    out_dir: PathBuf,

    #[arg(
        long,
        default_value = "work/sss_challenge/kissat_src_verbose/build/kissat"
    )]
    kissat: PathBuf,

    #[arg(long, default_value_t = false)]
    keep_cnf: bool,

    #[arg(long, default_value_t = false)]
    save_private_circuit: bool,
}

#[derive(Clone, Copy, Debug)]
struct Example {
    input: u128,
    output: u128,
}

struct HiddenCircuit {
    n: usize,
    circuit: CircuitSeq,
    reverse_gates: Vec<[u16; 3]>,
}

struct Layout {
    n: usize,
    m: usize,
    seq_base: i32,
    trace_base: i32,
    pos_base: i32,
    neg_base: i32,
    max_var: i32,
}

struct CnfWriter {
    body_path: PathBuf,
    writer: BufWriter<File>,
    clauses: u64,
    max_var: i32,
}

#[derive(Debug)]
struct SolveResult {
    sat: bool,
    elapsed_s: f64,
    status: Option<i32>,
}

impl HiddenCircuit {
    fn new(n: usize, m: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let circuit = random_circuit_seeded(n, m, &mut rng);
        let mut reverse_gates = circuit.gates.clone();
        reverse_gates.reverse();
        Self {
            n,
            circuit,
            reverse_gates,
        }
    }

    fn query(&self, q: u128) -> (u128, u128) {
        (
            eval_gates_u128(q, &self.circuit.gates, self.n),
            eval_gates_u128(q, &self.reverse_gates, self.n),
        )
    }
}

impl Layout {
    fn new(n: usize, m: usize, examples: usize) -> Self {
        let selector_count = 3usize * m * n;
        let seq_count = 3usize * m * (n - 1);
        let trace_count = examples * (m + 1) * n;
        let pos_count = examples * m;
        let neg_count = examples * m;
        let seq_base = 1 + selector_count as i32;
        let trace_base = seq_base + seq_count as i32;
        let pos_base = trace_base + trace_count as i32;
        let neg_base = pos_base + pos_count as i32;
        let max_var = neg_base + neg_count as i32 - 1;
        Self {
            n,
            m,
            seq_base,
            trace_base,
            pos_base,
            neg_base,
            max_var,
        }
    }

    fn selector(&self, kind: usize, gate: usize, wire: usize) -> i32 {
        1 + (kind * self.m * self.n + gate * self.n + wire) as i32
    }

    fn seq_aux(&self, kind: usize, gate: usize, index: usize) -> i32 {
        self.seq_base + ((kind * self.m + gate) * (self.n - 1) + index) as i32
    }

    fn trace(&self, example: usize, step: usize, wire: usize) -> i32 {
        self.trace_base + (example * (self.m + 1) * self.n + step * self.n + wire) as i32
    }

    fn pos(&self, example: usize, gate: usize) -> i32 {
        self.pos_base + (example * self.m + gate) as i32
    }

    fn neg(&self, example: usize, gate: usize) -> i32 {
        self.neg_base + (example * self.m + gate) as i32
    }
}

impl CnfWriter {
    fn new(body_path: PathBuf, max_var: i32) -> std::io::Result<Self> {
        let writer = BufWriter::new(File::create(&body_path)?);
        Ok(Self {
            body_path,
            writer,
            clauses: 0,
            max_var,
        })
    }

    fn add_clause(&mut self, lits: &[i32]) -> std::io::Result<()> {
        for &lit in lits {
            debug_assert_ne!(lit, 0);
            write!(self.writer, "{} ", lit)?;
        }
        writeln!(self.writer, "0")?;
        self.clauses += 1;
        Ok(())
    }

    fn finish(mut self, cnf_path: &Path) -> std::io::Result<u64> {
        self.writer.flush()?;
        let mut out = BufWriter::new(File::create(cnf_path)?);
        writeln!(out, "p cnf {} {}", self.max_var, self.clauses)?;
        let mut body = BufReader::new(File::open(&self.body_path)?);
        std::io::copy(&mut body, &mut out)?;
        out.flush()?;
        Ok(self.clauses)
    }
}

fn random_circuit_seeded(n: usize, m: usize, rng: &mut fastrand::Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut used = vec![false; n];
            let mut gate = [0u16; 3];
            for pin in &mut gate {
                loop {
                    let wire = rng.usize(..n);
                    if !used[wire] {
                        used[wire] = true;
                        *pin = wire as u16;
                        break;
                    }
                }
            }
            if gates.last() != Some(&gate) {
                gates.push(gate);
                break;
            }
        }
    }
    CircuitSeq { gates }
}

#[inline(always)]
fn mask_to_n(x: u128, n: usize) -> u128 {
    if n == 128 { x } else { x & ((1u128 << n) - 1) }
}

#[inline(always)]
fn eval_gates_u128(mut state: u128, gates: &[[u16; 3]], n: usize) -> u128 {
    state = mask_to_n(state, n);
    for &[a, b, c] in gates {
        let c1 = (state >> b) & 1;
        let c2 = (state >> c) & 1;
        state ^= (c1 | (1 ^ c2)) << a;
    }
    mask_to_n(state, n)
}

fn random_bits(rng: &mut fastrand::Rng, bits: usize) -> u128 {
    if bits == 0 {
        return 0;
    }
    let raw = ((rng.u64(..) as u128) << 64) | rng.u64(..) as u128;
    mask_to_n(raw, bits)
}

fn bit_lit(var: i32, value: bool) -> i32 {
    if value { var } else { -var }
}

fn add_exactly_one(cnf: &mut CnfWriter, xs: &[i32], aux: &[i32]) -> std::io::Result<()> {
    debug_assert!(xs.len() >= 2);
    debug_assert_eq!(aux.len(), xs.len() - 1);
    cnf.add_clause(xs)?;
    cnf.add_clause(&[-xs[0], aux[0]])?;
    for i in 1..xs.len() - 1 {
        cnf.add_clause(&[-xs[i], aux[i]])?;
        cnf.add_clause(&[-aux[i - 1], aux[i]])?;
        cnf.add_clause(&[-xs[i], -aux[i - 1]])?;
    }
    cnf.add_clause(&[-xs[xs.len() - 1], -aux[aux.len() - 1]])?;
    Ok(())
}

fn add_update_clauses(
    cnf: &mut CnfWriter,
    guard: i32,
    old_a: i32,
    b: i32,
    c: i32,
    new_a: i32,
) -> std::io::Result<()> {
    cnf.add_clause(&[-guard, -old_a, -b, -new_a])?;
    cnf.add_clause(&[-guard, -old_a, c, -new_a])?;
    cnf.add_clause(&[-guard, old_a, -b, new_a])?;
    cnf.add_clause(&[-guard, old_a, c, new_a])?;
    cnf.add_clause(&[-guard, -old_a, b, -c, new_a])?;
    cnf.add_clause(&[-guard, old_a, b, -c, -new_a])?;
    Ok(())
}

fn build_learning_cnf(
    n: usize,
    m: usize,
    examples: &[Example],
    out_dir: &Path,
    stem: &str,
) -> std::io::Result<(PathBuf, PathBuf, Layout, u64)> {
    let layout = Layout::new(n, m, examples.len());
    let body_path = out_dir.join(format!("{stem}.body"));
    let cnf_path = out_dir.join(format!("{stem}.cnf"));
    let mut cnf = CnfWriter::new(body_path.clone(), layout.max_var)?;

    for gate in 0..m {
        for kind in 0..3 {
            let xs: Vec<i32> = (0..n)
                .map(|wire| layout.selector(kind, gate, wire))
                .collect();
            let aux: Vec<i32> = (0..n - 1)
                .map(|idx| layout.seq_aux(kind, gate, idx))
                .collect();
            add_exactly_one(&mut cnf, &xs, &aux)?;
        }

        for wire in 0..n {
            cnf.add_clause(&[
                -layout.selector(0, gate, wire),
                -layout.selector(1, gate, wire),
            ])?;
            cnf.add_clause(&[
                -layout.selector(0, gate, wire),
                -layout.selector(2, gate, wire),
            ])?;
            cnf.add_clause(&[
                -layout.selector(1, gate, wire),
                -layout.selector(2, gate, wire),
            ])?;
        }
    }

    for (example_idx, example) in examples.iter().enumerate() {
        for wire in 0..n {
            cnf.add_clause(&[bit_lit(
                layout.trace(example_idx, 0, wire),
                ((example.input >> wire) & 1) != 0,
            )])?;
            cnf.add_clause(&[bit_lit(
                layout.trace(example_idx, m, wire),
                ((example.output >> wire) & 1) != 0,
            )])?;
        }

        for gate in 0..m {
            let pos = layout.pos(example_idx, gate);
            let neg = layout.neg(example_idx, gate);
            for wire in 0..n {
                let prev = layout.trace(example_idx, gate, wire);
                let next = layout.trace(example_idx, gate + 1, wire);
                let a = layout.selector(0, gate, wire);
                let b_sel = layout.selector(1, gate, wire);
                let d_sel = layout.selector(2, gate, wire);

                cnf.add_clause(&[-b_sel, -prev, pos])?;
                cnf.add_clause(&[-b_sel, prev, -pos])?;
                cnf.add_clause(&[-d_sel, -prev, neg])?;
                cnf.add_clause(&[-d_sel, prev, -neg])?;

                cnf.add_clause(&[a, -prev, next])?;
                cnf.add_clause(&[a, prev, -next])?;
                add_update_clauses(&mut cnf, a, prev, pos, neg, next)?;
            }
        }
    }

    let clauses = cnf.finish(&cnf_path)?;
    Ok((cnf_path, body_path, layout, clauses))
}

fn solve_cnf(kissat: &Path, cnf: &Path, out: &Path) -> std::io::Result<SolveResult> {
    let start = Instant::now();
    let stdout = File::create(out)?;
    let status = Command::new(kissat)
        .arg(cnf)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::null())
        .status()?;
    let code = status.code();
    Ok(SolveResult {
        sat: code == Some(10),
        elapsed_s: start.elapsed().as_secs_f64(),
        status: code,
    })
}

fn parse_candidate_model(model_path: &Path, layout: &Layout) -> std::io::Result<CircuitSeq> {
    let mut gates = vec![[u16::MAX; 3]; layout.m];
    let file = BufReader::new(File::open(model_path)?);
    for line in file.lines() {
        let line = line?;
        if !line.starts_with('v') {
            continue;
        }
        for tok in line[1..].split_whitespace() {
            let Ok(lit) = tok.parse::<i32>() else {
                continue;
            };
            if lit <= 0 {
                continue;
            }
            let v0 = lit - 1;
            let selector_count = (3 * layout.m * layout.n) as i32;
            if v0 >= selector_count {
                continue;
            }
            let idx = v0 as usize;
            let kind = idx / (layout.m * layout.n);
            let rest = idx % (layout.m * layout.n);
            let gate = rest / layout.n;
            let wire = rest % layout.n;
            gates[gate][kind] = wire as u16;
        }
    }
    for (idx, gate) in gates.iter().enumerate() {
        if gate.iter().any(|&w| w == u16::MAX) {
            panic!("model did not assign all pins for gate {idx}: {gate:?}");
        }
    }
    Ok(CircuitSeq { gates })
}

fn add_oracle_query(hidden: &HiddenCircuit, examples: &mut Vec<Example>, q: u128) {
    let (forward, inverse) = hidden.query(q);
    examples.push(Example {
        input: mask_to_n(q, hidden.n),
        output: forward,
    });
    examples.push(Example {
        input: inverse,
        output: mask_to_n(q, hidden.n),
    });
}

fn format_hex_n(x: u128, n: usize) -> String {
    let nybbles = n.div_ceil(4);
    format!("0x{:0width$x}", mask_to_n(x, n), width = nybbles)
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    assert!(args.wires >= 3 && args.wires <= 128);
    fs::create_dir_all(&args.out_dir)?;

    let hidden = HiddenCircuit::new(args.wires, args.gates, args.seed ^ 0x9e37_79b9_7f4a_7c15);
    if args.save_private_circuit {
        let mut private = BufWriter::new(File::create(args.out_dir.join("private_circuit.txt"))?);
        writeln!(private, "{}", hidden.circuit.repr())?;
    }

    let mut rng = fastrand::Rng::with_seed(args.seed ^ 0xbf58_476d_1ce4_e5b9);
    let mut examples = Vec::new();
    for _ in 0..args.initial_queries {
        let q = random_bits(&mut rng, args.wires);
        add_oracle_query(&hidden, &mut examples, q);
    }

    let mut ledger = BufWriter::new(File::create(args.out_dir.join("learner_ledger.tsv"))?);
    writeln!(
        ledger,
        "round\toracle_queries\texamples\tvars\tclauses\tsat\tstatus\tsolve_s\tvalidated\tcounterexample\tcandidate_matches_hidden"
    )?;

    for round in 0..args.max_rounds {
        let stem = format!("round_{round:03}");
        let (cnf_path, body_path, layout, clauses) =
            build_learning_cnf(args.wires, args.gates, &examples, &args.out_dir, &stem)?;
        let model_path = args.out_dir.join(format!("{stem}.model"));
        let solve = solve_cnf(&args.kissat, &cnf_path, &model_path)?;
        if !args.keep_cnf {
            let _ = fs::remove_file(&cnf_path);
            let _ = fs::remove_file(&body_path);
        }

        if !solve.sat {
            writeln!(
                ledger,
                "{round}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:.6}\t0\t\tfalse",
                examples.len() / 2,
                examples.len(),
                layout.max_var,
                clauses,
                solve.sat,
                solve.status,
                solve.elapsed_s
            )?;
            ledger.flush()?;
            break;
        }

        let candidate = parse_candidate_model(&model_path, &layout)?;
        let mut counterexample = None;
        let mut validated = 0usize;
        for _ in 0..args.validation_queries {
            let q = random_bits(&mut rng, args.wires);
            let real = hidden.query(q);
            let predicted = eval_gates_u128(q, &candidate.gates, args.wires);
            validated += 1;
            if predicted != real.0 {
                counterexample = Some(q);
                break;
            }
            let mut reverse = candidate.gates.clone();
            reverse.reverse();
            let predicted_inverse = eval_gates_u128(q, &reverse, args.wires);
            if predicted_inverse != real.1 {
                counterexample = Some(real.1);
                break;
            }
        }

        let candidate_matches_hidden = candidate.gates == hidden.circuit.gates;
        writeln!(
            ledger,
            "{round}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:.6}\t{}\t{}\t{}",
            examples.len() / 2,
            examples.len(),
            layout.max_var,
            clauses,
            solve.sat,
            solve.status,
            solve.elapsed_s,
            validated,
            counterexample
                .map(|q| format_hex_n(q, args.wires))
                .unwrap_or_default(),
            candidate_matches_hidden
        )?;
        ledger.flush()?;

        println!(
            "round={round} examples={} vars={} clauses={} solve={:.3}s validated={} counterexample={} exact={}",
            examples.len(),
            layout.max_var,
            clauses,
            solve.elapsed_s,
            validated,
            counterexample.is_some(),
            candidate_matches_hidden
        );

        if let Some(q) = counterexample {
            add_oracle_query(&hidden, &mut examples, q);
        } else {
            break;
        }
    }

    println!(
        "ledger\t{}",
        args.out_dir.join("learner_ledger.tsv").display()
    );
    Ok(())
}
