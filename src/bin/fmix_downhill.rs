//! Find locally profitable inverse-fmix conjugations using only the released
//! `mpmct1` circuit.  No mixer provenance, seed, or source circuit is used.
//!
//! For a same-target block A and an adjacent gate h that does not read A's
//! target, moving A across h conjugates A's ESOP control function by h.  If h
//! targets b, this is exactly the substitution b <- b XOR fire(h).  Forward
//! fmix R1 crossings expand one cube into a case-split ladder; the reverse
//! substitution can therefore collapse that ladder again.  This tool scans
//! both sides of every maximal same-target run and reports substitutions that
//! reduce the number of ESOP cubes after exact catalogue reduction.

use clap::Parser;
use local_mixing::postmix::format;
use local_mixing::postmix::mix::{Merge, merge_result};
use local_mixing::postmix::xgate::{Lits, XGate, eval_lanes};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fmix_downhill")]
struct Args {
    /// Released mpmct1 challenge circuit.
    #[arg(long)]
    input: String,
    /// Print at most this many best candidates.
    #[arg(long, default_value_t = 30)]
    top: usize,
    /// Apply this many non-overlapping downhill passes (zero = scan only).
    #[arg(long, default_value_t = 0)]
    passes: usize,
    /// Write the demixed mpmct1 circuit after applying passes.
    #[arg(long)]
    output: Option<String>,
    /// Random 64-lane equivalence checks after rewriting.
    #[arg(long, default_value_t = 16)]
    verify_rounds: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Clone)]
struct Esop {
    parity: bool,
    cubes: Vec<XGate>,
}

#[derive(Clone, Debug)]
struct Candidate {
    lo: usize,
    hi: usize,
    neighbor: usize,
    side: &'static str,
    block_target: u16,
    neighbor_target: u16,
    before_gates: usize,
    after_gates: usize,
    before_lits: usize,
    after_lits: usize,
    span_lo: usize,
    span_hi: usize,
    replacement: Vec<XGate>,
}

fn xor_add(esop: &mut Esop, mut cube: XGate) {
    debug_assert!(!cube.comp);
    if cube.ctrls.is_empty() {
        esop.parity ^= true;
        return;
    }
    cube.comp = false;
    if let Some(i) = esop.cubes.iter().position(|g| g == &cube) {
        esop.cubes.swap_remove(i);
    } else {
        esop.cubes.push(cube);
    }
}

fn from_block(block: &[XGate], target: u16) -> Esop {
    let mut out = Esop {
        parity: false,
        cubes: Vec::new(),
    };
    for g in block {
        debug_assert_eq!(g.target, target);
        out.parity ^= g.comp;
        xor_add(
            &mut out,
            XGate {
                target,
                comp: false,
                ctrls: g.ctrls.clone(),
            },
        );
    }
    reduce(out)
}

fn reduce(mut esop: Esop) -> Esop {
    loop {
        let mut found = None;
        'outer: for i in 0..esop.cubes.len() {
            for j in i + 1..esop.cubes.len() {
                if let Some(m) = merge_result(&esop.cubes[i], &esop.cubes[j]) {
                    found = Some((i, j, m));
                    break 'outer;
                }
            }
        }
        let Some((i, j, merge)) = found else { break };
        esop.cubes.swap_remove(j);
        esop.cubes.swap_remove(i);
        match merge {
            Merge::Cancel => {}
            Merge::XFuse(g) | Merge::DropLit(g) | Merge::Subsume(g) => xor_add(&mut esop, g),
        }
    }
    esop
}

fn product(target: u16, a: &Lits, b: &Lits) -> Option<XGate> {
    XGate::conj(target, a.iter().copied().chain(b.iter().copied()))
}

/// Conjugate `phi` by h.  This is an involution because h is an involution.
fn conjugate(phi: &Esop, h: &XGate, target: u16) -> Esop {
    debug_assert!(!h.reads(target));
    let b = h.target;
    let mut out = Esop {
        parity: phi.parity,
        cubes: Vec::new(),
    };
    for cube in &phi.cubes {
        xor_add(&mut out, cube.clone());
        if !cube.reads(b) {
            continue;
        }
        let stripped: Lits = cube
            .ctrls
            .iter()
            .copied()
            .filter(|&(w, _)| w != b)
            .collect();
        // fire(h) = h.comp XOR product(h.ctrls).  Substituting b XOR fire(h)
        // into either polarity of b changes the literal by exactly fire(h).
        if h.comp {
            xor_add(
                &mut out,
                XGate {
                    target,
                    comp: false,
                    ctrls: stripped.clone(),
                },
            );
        }
        if let Some(g) = product(target, &stripped, &h.ctrls) {
            xor_add(&mut out, g);
        }
    }
    reduce(out)
}

fn gate_cost(esop: &Esop) -> usize {
    esop.cubes.len() + usize::from(esop.parity)
}

fn lit_cost(esop: &Esop) -> usize {
    esop.cubes.iter().map(XGate::width).sum()
}

fn gates_of(esop: Esop, target: u16) -> Vec<XGate> {
    let mut out = esop.cubes;
    if esop.parity {
        out.push(XGate::x_gate(target));
    }
    out
}

fn consider(
    gates: &[XGate],
    lo: usize,
    hi: usize,
    neighbor: usize,
    side: &'static str,
    out: &mut Vec<Candidate>,
) {
    let target = gates[lo].target;
    let h = &gates[neighbor];
    if h.target == target || h.reads(target) {
        return;
    }
    let before = from_block(&gates[lo..hi], target);
    if !before.cubes.iter().any(|g| g.reads(h.target)) {
        return;
    }
    let after = conjugate(&before, h, target);
    let bg = gate_cost(&before);
    let ag = gate_cost(&after);
    let bl = lit_cost(&before);
    let al = lit_cost(&after);
    if ag < bg || (ag == bg && al < bl) {
        let mut replacement = gates_of(after, target);
        let (span_lo, span_hi) = if side == "left" {
            replacement.push(h.clone());
            (neighbor, hi)
        } else {
            replacement.insert(0, h.clone());
            (lo, neighbor + 1)
        };
        out.push(Candidate {
            lo,
            hi,
            neighbor,
            side,
            block_target: target,
            neighbor_target: h.target,
            before_gates: bg,
            after_gates: ag,
            before_lits: bl,
            after_lits: al,
            span_lo,
            span_hi,
            replacement,
        });
    }
}

fn scan(gates: &[XGate]) -> (usize, usize, Vec<Candidate>) {
    let mut candidates = Vec::new();
    let mut runs = 0usize;
    let mut multi_runs = 0usize;
    let mut i = 0usize;
    while i < gates.len() {
        let mut j = i + 1;
        while j < gates.len() && gates[j].target == gates[i].target {
            j += 1;
        }
        runs += 1;
        multi_runs += usize::from(j - i > 1);
        if i > 0 {
            consider(&gates, i, j, i - 1, "left", &mut candidates);
        }
        if j < gates.len() {
            consider(&gates, i, j, j, "right", &mut candidates);
        }
        i = j;
    }
    candidates.sort_unstable_by_key(|c| {
        (
            std::cmp::Reverse(c.before_gates - c.after_gates),
            std::cmp::Reverse(c.before_lits.saturating_sub(c.after_lits)),
            c.lo,
        )
    });
    (runs, multi_runs, candidates)
}

fn main() {
    let args = Args::parse();
    let (mut gates, wires) = format::read_mpmct(&args.input).expect("read mpmct1 circuit");
    let original = gates.clone();
    for pass in 0..=args.passes {
        let (runs, multi_runs, candidates) = scan(&gates);
        let gate_savings: usize = candidates
            .iter()
            .map(|c| c.before_gates - c.after_gates)
            .sum();
        println!(
            "[downhill] pass={} wires={} gates={} runs={} multi_runs={} profitable={} opportunity_gate_savings={}",
            pass,
            wires,
            gates.len(),
            runs,
            multi_runs,
            candidates.len(),
            gate_savings
        );
        for c in candidates.iter().take(args.top) {
            println!(
                "[candidate] side={} block={}..{} neighbor={} targets={}/{} gates={}->{} lits={}->{}",
                c.side,
                c.lo,
                c.hi,
                c.neighbor,
                c.block_target,
                c.neighbor_target,
                c.before_gates,
                c.after_gates,
                c.before_lits,
                c.after_lits
            );
        }
        if pass == args.passes || candidates.is_empty() {
            break;
        }

        // Prefer the strongest reductions, then take a maximal non-overlapping
        // subset.  Rebuild the vector once, so a pass stays linear in circuit
        // size even when it contains thousands of opportunities.
        let mut ranked: Vec<&Candidate> = candidates.iter().collect();
        ranked.sort_unstable_by_key(|c| {
            (
                std::cmp::Reverse(c.before_gates - c.after_gates),
                std::cmp::Reverse(c.before_lits.saturating_sub(c.after_lits)),
                c.span_lo,
            )
        });
        let mut occupied = vec![false; gates.len()];
        let mut chosen: Vec<&Candidate> = Vec::new();
        for c in ranked {
            if occupied[c.span_lo..c.span_hi].iter().any(|&x| x) {
                continue;
            }
            occupied[c.span_lo..c.span_hi].fill(true);
            chosen.push(c);
        }
        chosen.sort_unstable_by_key(|c| c.span_lo);
        let before = gates.len();
        let mut next = Vec::with_capacity(before);
        let mut cursor = 0usize;
        for c in chosen {
            next.extend_from_slice(&gates[cursor..c.span_lo]);
            next.extend(c.replacement.iter().cloned());
            cursor = c.span_hi;
        }
        next.extend_from_slice(&gates[cursor..]);
        gates = next;
        println!(
            "[downhill] applied pass={} gates {} -> {}",
            pass + 1,
            before,
            gates.len()
        );
    }

    if args.passes > 0 {
        let mut rng = StdRng::seed_from_u64(args.seed);
        for round in 0..args.verify_rounds {
            let state: Vec<u64> = (0..wires).map(|_| rng.random()).collect();
            let mut a = state.clone();
            let mut b = state;
            eval_lanes(original.iter(), &mut a);
            eval_lanes(gates.iter(), &mut b);
            assert_eq!(a, b, "global equivalence failed in round {round}");
        }
        println!(
            "[downhill] verified {} rounds x64 lanes; gates {} -> {}",
            args.verify_rounds,
            original.len(),
            gates.len()
        );
    }
    if let Some(path) = &args.output {
        format::write_mpmct(path, &gates, wires).expect("write output circuit");
        println!("[downhill] wrote {}", path);
    }
}
