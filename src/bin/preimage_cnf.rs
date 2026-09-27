//! Emit a DIMACS preimage instance for a mixed circuit, with the reductions
//! that actually shrink the solver's job:
//!
//!   * CONE OF INFLUENCE  — only 128 of 512 output wires are constrained, so
//!     every gate that cannot reach the payload block is deleted (backward pass).
//!   * CONSTANT PROPAGATION — 384 of 512 input wires are pinned 0, so early
//!     gates fold to constants and cost no variables or clauses at all.
//!   * optional INVARIANT INJECTION (--inject) — the verified cross-cut affine
//!     relations from invariant_classify, as redundant XOR-chain lemmas.
//!
//! Encoding: SSA over wires. wire -> Const(b) | Var(lit). Per live gate at most
//! one AND var (k+1 clauses) and one XOR var (4 clauses); folds emit nothing.
//!
//!   preimage_cnf <circuit.mpmct1> <Yhex> -o out.cnf [--inject rels.txt]
//!                [--selfcheck <Xhex>] [--stats-only]
//!
//! --selfcheck X evaluates the circuit on X, assigns every emitted variable its
//! true value, and asserts every clause is satisfied — a complete validation of
//! the encoding (use with the Y that X actually produces).
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::XGate;
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum W {
    Const(bool),
    Var(i32), // dimacs literal, sign = polarity
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: preimage_cnf <circuit.mpmct1> <Yhex> -o out.cnf [opts]");
    let yhex = a.next().expect("need Y hex");
    let (mut out_path, mut inject, mut selfcheck) = (String::new(), None::<String>, None::<u128>);
    let (mut xw0, mut yw0, mut blk) = (0usize, 128usize, 128usize);
    let mut stats_only = false;
    let mut free_bits: Option<usize> = None;
    let mut it = a.peekable();
    while let Some(t) = it.next() {
        match t.as_str() {
            "-o" | "--out" => out_path = it.next().unwrap(),
            "--inject" => inject = Some(it.next().unwrap()),
            "--selfcheck" => {
                selfcheck =
                    Some(u128::from_str_radix(it.next().unwrap().trim_start_matches("0x"), 16).unwrap())
            }
            "--xw0" => xw0 = it.next().unwrap().parse().unwrap(),
            "--yw0" => yw0 = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            "--stats-only" => stats_only = true,
            // hardness ladder: leave only the low N input bits free, pin the rest
            // to their true values from --selfcheck X*. Instance stays SAT with a
            // known unique solution; difficulty scales with N.
            "--free-bits" => free_bits = Some(it.next().unwrap().parse::<usize>().unwrap()),
            o => panic!("unknown arg {o}"),
        }
    }
    let free_bits = free_bits.unwrap_or(blk);
    if free_bits < blk {
        assert!(selfcheck.is_some(), "--free-bits needs --selfcheck X* to pin the remaining bits");
    }
    let y = u128::from_str_radix(yhex.trim_start_matches("0x"), 16).expect("bad Y hex");

    let inject_requested = inject.is_some();
    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    let ng = gates.len();
    eprintln!("loaded {ng} gates / {num_wires} wires ({:.1}s)", t0.elapsed().as_secs_f64());

    // ---------- cone of influence (backward) ----------
    let mut needed = vec![false; num_wires];
    for j in 0..blk {
        needed[yw0 + j] = true;
    }
    let mut live = vec![false; ng];
    for gi in (0..ng).rev() {
        let g = &gates[gi];
        if needed[g.target as usize] {
            live[gi] = true;
            for &(w, _) in &g.ctrls {
                needed[w as usize] = true;
            }
        }
    }
    let nlive = live.iter().filter(|&&l| l).count();
    eprintln!(
        "cone of influence: {nlive}/{ng} gates live ({:.1}% removed), {} wires relevant",
        100.0 * (ng - nlive) as f64 / ng as f64,
        needed.iter().filter(|&&n| n).count()
    );
    // Injected relations reference wire values at cut points, including wires
    // written only by cone-dead gates. Skipping those gates would leave their
    // snapshot entries stale and inject FALSE lemmas (observed: a known-SAT
    // instance turned UNSAT). Cone pruning is worth ~0.1% here, so drop it.
    if inject_requested {
        live.iter_mut().for_each(|l| *l = true);
        eprintln!("  (cone pruning disabled: --inject needs exact wire snapshots)");
    }

    // ---------- cut positions, if injecting ----------
    let mut cut_gates: Vec<usize> = Vec::new();
    let mut rels: Vec<(bool, Vec<(usize, usize)>)> = Vec::new();
    if let Some(p) = &inject {
        let s = std::fs::read_to_string(p).expect("read relations");
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("# cuts") {
                cut_gates = rest.split_whitespace().map(|t| t.parse().unwrap()).collect();
            } else if let Some(rest) = line.strip_prefix("rel ") {
                let mut t = rest.split_whitespace();
                let c: u8 = t.next().unwrap().parse().unwrap();
                let n: usize = t.next().unwrap().parse().unwrap();
                let mut sup = Vec::with_capacity(n);
                for tok in t {
                    let (a, b) = tok.split_once(':').unwrap();
                    sup.push((a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()));
                }
                rels.push((c == 1, sup));
            }
        }
        eprintln!("injecting {} relations across {} cuts", rels.len(), cut_gates.len());
    }

    // ---------- emit ----------
    let mut wire: Vec<W> = (0..num_wires).map(|_| W::Const(false)).collect();
    let mut truth: Vec<bool> = vec![false]; // truth[0] unused; index by var id
    let mut sim: Vec<bool> = vec![false; num_wires]; // real values under selfcheck X
    let mut nvar = 0i32;
    for i in 0..blk {
        let b = selfcheck.map(|x| (x >> i) & 1 == 1).unwrap_or(false);
        sim[xw0 + i] = b;
        if i < free_bits {
            nvar += 1;
            wire[xw0 + i] = W::Var(nvar);
            truth.push(b);
        } else {
            wire[xw0 + i] = W::Const(b); // pinned to X* for the ladder
        }
    }
    if free_bits < blk {
        eprintln!("ladder: {free_bits}/{blk} input bits free, {} pinned to X*", blk - free_bits);
    }

    let mut clauses: Vec<Vec<i32>> = Vec::new(); // buffered; written at the end
    let mut nclause = 0usize;
    let mut bad = 0usize;

    // value of a literal under the selfcheck assignment
    macro_rules! lit_true {
        ($l:expr) => {{
            let l: i32 = $l;
            let v = truth[l.unsigned_abs() as usize];
            if l > 0 {
                v
            } else {
                !v
            }
        }};
    }
    macro_rules! emit {
        ($c:expr) => {{
            let c: Vec<i32> = $c;
            if selfcheck.is_some() && !c.iter().any(|&l| lit_true!(l)) {
                bad += 1;
            }
            nclause += 1;
            if !stats_only {
                clauses.push(c);
            }
        }};
    }

    let mut snaps: Vec<Vec<W>> = Vec::new();
    let mut sim_snaps: Vec<Vec<bool>> = Vec::new();
    let mut ci = 0usize;
    let mut and_vars = 0usize;
    let mut xor_vars = 0usize;
    let mut folded = 0usize;

    for gi in 0..ng {
        while ci < cut_gates.len() && cut_gates[ci] == gi {
            snaps.push(wire.clone());
            sim_snaps.push(sim.clone());
            ci += 1;
        }
        let g: &XGate = &gates[gi];
        // keep the simulator exact over ALL gates
        if selfcheck.is_some() {
            let mut f = true;
            for &(w, p) in &g.ctrls {
                f &= sim[w as usize] == p;
            }
            f ^= g.comp;
            if f {
                sim[g.target as usize] ^= true;
            }
        }
        if !live[gi] {
            continue;
        }

        // fold controls
        let mut and_lits: Vec<i32> = Vec::with_capacity(g.ctrls.len());
        let mut is_false = false;
        for &(w, p) in &g.ctrls {
            match wire[w as usize] {
                W::Const(b) => {
                    if b != p {
                        is_false = true;
                        break;
                    }
                }
                W::Var(l) => and_lits.push(if p { l } else { -l }),
            }
        }
        let and_w = if is_false {
            W::Const(false)
        } else if and_lits.is_empty() {
            W::Const(true)
        } else if and_lits.len() == 1 {
            W::Var(and_lits[0])
        } else {
            nvar += 1;
            let av = nvar;
            let tv = and_lits.iter().all(|&l| lit_true!(l));
            truth.push(tv);
            for &l in &and_lits {
                emit!(vec![-av, l]);
            }
            let mut c = vec![av];
            for &l in &and_lits {
                c.push(-l);
            }
            emit!(c);
            and_vars += 1;
            W::Var(av)
        };

        let fires = match and_w {
            W::Const(b) => W::Const(b ^ g.comp),
            W::Var(l) => W::Var(if g.comp { -l } else { l }),
        };

        let t = g.target as usize;
        wire[t] = match (wire[t], fires) {
            (cur, W::Const(false)) => {
                folded += 1;
                cur
            }
            (W::Const(b), W::Const(true)) => {
                folded += 1;
                W::Const(!b)
            }
            (W::Var(l), W::Const(true)) => {
                folded += 1;
                W::Var(-l)
            }
            (W::Const(b), W::Var(f)) => {
                folded += 1;
                W::Var(if b { -f } else { f })
            }
            (W::Var(l), W::Var(f)) => {
                nvar += 1;
                let n = nvar;
                truth.push(lit_true!(l) ^ lit_true!(f));
                emit!(vec![-n, l, f]);
                emit!(vec![-n, -l, -f]);
                emit!(vec![n, -l, f]);
                emit!(vec![n, l, -f]);
                xor_vars += 1;
                W::Var(n)
            }
        };
    }
    while ci < cut_gates.len() {
        snaps.push(wire.clone());
        sim_snaps.push(sim.clone());
        ci += 1;
    }

    // ---------- target constraint ----------
    let mut unsat_by_const = false;
    for j in 0..blk {
        let want = (y >> j) & 1 == 1;
        match wire[yw0 + j] {
            W::Const(b) => {
                if b != want {
                    unsat_by_const = true;
                }
            }
            W::Var(l) => emit!(vec![if want { l } else { -l }]),
        }
    }

    // ---------- injected invariants (redundant XOR-chain lemmas) ----------
    let mut inj_clauses = 0usize;
    if !rels.is_empty() {
        let before = nclause;
        // Guard: every injected relation must hold under the known-true assignment.
        // A violation means the lemma is false and would turn a SAT instance UNSAT.
        if selfcheck.is_some() {
            let mut viol = 0usize;
            for (cst, sup) in &rels {
                let mut acc = false;
                for &(cut, w) in sup {
                    if cut < sim_snaps.len() {
                        acc ^= sim_snaps[cut][w];
                    }
                }
                if acc != *cst {
                    viol += 1;
                }
            }
            if viol > 0 {
                eprintln!("FATAL: {viol}/{} injected relations are FALSE under X* — aborting.", rels.len());
                std::process::exit(2);
            }
            eprintln!("relation guard: all {} injected relations hold under X*", rels.len());
        }
        for (cst, sup) in &rels {
            let mut acc: Option<W> = None;
            let mut ok = true;
            for &(cut, w) in sup {
                if cut >= snaps.len() {
                    ok = false;
                    break;
                }
                let cur = snaps[cut][w];
                acc = Some(match (acc, cur) {
                    (None, v) => v,
                    (Some(W::Const(a)), W::Const(b)) => W::Const(a ^ b),
                    (Some(W::Const(a)), W::Var(l)) => W::Var(if a { -l } else { l }),
                    (Some(W::Var(l)), W::Const(b)) => W::Var(if b { -l } else { l }),
                    (Some(W::Var(l)), W::Var(f)) => {
                        nvar += 1;
                        let n = nvar;
                        truth.push(lit_true!(l) ^ lit_true!(f));
                        emit!(vec![-n, l, f]);
                        emit!(vec![-n, -l, -f]);
                        emit!(vec![n, -l, f]);
                        emit!(vec![n, l, -f]);
                        W::Var(n)
                    }
                });
            }
            if !ok {
                continue;
            }
            match acc {
                Some(W::Var(l)) => emit!(vec![if *cst { l } else { -l }]),
                Some(W::Const(b)) => {
                    if b != *cst {
                        eprintln!("WARNING: injected relation contradicts constant fold");
                    }
                }
                None => {}
            }
        }
        inj_clauses = nclause - before;
    }

    eprintln!(
        "vars {nvar} ({and_vars} AND, {xor_vars} XOR), clauses {nclause} ({inj_clauses} injected), {folded} gates folded free ({:.1}s)",
        t0.elapsed().as_secs_f64()
    );
    if let Some(x) = selfcheck {
        let mut got = 0u128;
        for j in 0..blk {
            if sim[yw0 + j] {
                got |= 1u128 << j;
            }
        }
        eprintln!("SELFCHECK: X={x:032x} -> payload {got:032x}; target {y:032x}");
        eprintln!(
            "SELFCHECK: forward match {}; unsatisfied clauses under true assignment: {}",
            if got == y { "YES" } else { "NO" },
            bad
        );
        if got == y && bad == 0 {
            eprintln!("SELFCHECK: PASS — encoding is faithful and the instance is SAT.");
        } else {
            eprintln!("SELFCHECK: FAIL");
        }
    }
    if unsat_by_const {
        eprintln!("WARNING: target contradicts a constant-folded output bit -> UNSAT by construction");
    }

    if stats_only {
        return;
    }
    let f = std::fs::File::create(&out_path).expect("create cnf");
    let mut w = BufWriter::with_capacity(1 << 22, f);
    writeln!(w, "p cnf {nvar} {nclause}").unwrap();
    let mut buf = String::with_capacity(1 << 16);
    for c in &clauses {
        for l in c {
            use std::fmt::Write as _;
            let _ = write!(buf, "{l} ");
        }
        buf.push_str("0\n");
        if buf.len() > (1 << 15) {
            w.write_all(buf.as_bytes()).unwrap();
            buf.clear();
        }
    }
    w.write_all(buf.as_bytes()).unwrap();
    w.flush().unwrap();
    eprintln!("wrote {out_path} ({:.1}s)", t0.elapsed().as_secs_f64());
}
